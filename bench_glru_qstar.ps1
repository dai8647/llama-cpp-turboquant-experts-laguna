# Validation harness for the global-LRU slot pool (coder A) and the q*
# bandwidth-adaptive split. Fires only once the shared tree builds.
#
# What it proves:
#   glru  : the collection-mode gate lift works end-to-end -> residents /
#           cross_evict counters MOVE, and hit-rate is NOT statically pinned
#           (contrast with whitelist mode, see frequency-placement-findings.md).
#   qstar : the decode split engages -> [q*] lines show a non-trivial
#           xfer-vs-cpu decision per layer/step.
#
# Graphs are forced OFF: the eval-time remap/planning ops make per-step
# residency decisions that a captured HIP graph would replay stale.
#
# Usage:
#   .\bench_glru_qstar.ps1 -Mode glru  -Slots 96 -Rounds 3
#   .\bench_glru_qstar.ps1 -Mode qstar -Slots 96 -Rounds 2
param(
    [int]$Port = 8096,
    [int]$Ctx = 32768,
    [int]$Predict = 128,
    [int]$Slots = 96,
    [int]$Rounds = 3,
    [ValidateSet('glru','qstar')]
    [string]$Mode = 'glru',
    [string]$Tag = "",
    [int]$IdleAfter = 2,
    # -Mode qstar passes --moe-qstar, which only the round-2 branch binary accepts;
    # point this at that build (omitted = main-tree build-hip binary as before)
    [string]$BinaryPath = ''
)
$ErrorActionPreference = 'Continue'
if (-not $Tag) { $Tag = "{0}-slot{1}" -f $Mode, $Slots }

$root  = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna'
$model = 'C:\Users\dai86\.lmstudio\models\huihui-ai\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf'

$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:LLAMA_MOE_SLOT_STATS = '1'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
Remove-Item Env:LLAMA_MOE_PREFETCH_MS -ErrorAction SilentlyContinue
# q* per-step per-layer decision log (verbose; parsed below)
if ($Mode -eq 'qstar') { $env:LLAMA_MOE_QSTAR_STATS = '1' } else { Remove-Item Env:LLAMA_MOE_QSTAR_STATS -ErrorAction SilentlyContinue }

Set-Location $root
$outLog = "$root\glru_out.log"
$errLog = "$root\glru_err.log"
Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue

$baseArgs = @('-m', $model, '--host', '127.0.0.1', '--port', "$Port", '--no-webui',
    '-lv', '4', '-c', "$Ctx", '-np', '1', '-t', '6', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '--cpu-moe',
    '--moe-gpu-expert-slot-num', "$Slots")
if ($Mode -eq 'glru') { $baseArgs += @('--moe-gpu-expert-global-lru') }
if ($Mode -eq 'qstar') { $baseArgs += @('--moe-qstar') }

$sw = [Diagnostics.Stopwatch]::StartNew()
$serverExe = if ($BinaryPath) { $BinaryPath } else { "$root\build-hip\bin\llama-server.exe" }
$p = Start-Process -FilePath $serverExe -ArgumentList $baseArgs `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -WindowStyle Hidden

$ready = $false
while ($sw.Elapsed.TotalSeconds -lt 300) {
    Start-Sleep -Milliseconds 700
    if ($p.HasExited) { break }
    $tail = Get-Content $errLog -Tail 3 -ErrorAction SilentlyContinue
    if ($tail -match 'listening') { $ready = $true; break }
}
$loadS = [math]::Round($sw.Elapsed.TotalSeconds, 1)
if (-not $ready) {
    Write-Output ("{0}: LOAD-FAILED t={1}s alive={2}" -f $Tag, $loadS, (-not $p.HasExited))
    Get-Content $errLog -Tail 12 -ErrorAction SilentlyContinue
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    exit 1
}

# wait until /health reports ok (weights fully mapped)
$hz = [Diagnostics.Stopwatch]::StartNew()
while ($hz.Elapsed.TotalSeconds -lt 180) {
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5; if ($h.status -eq 'ok') { break } } catch {}
    Start-Sleep -Milliseconds 900
}

# warmup
$null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' `
    -Body (@{ prompt = 'Say hello.'; n_predict = 16; temperature = 0 } | ConvertTo-Json) -TimeoutSec 180

function Make-Prompt([int]$variant) {
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add(("Review the following C++ code (variant {0}) and summarize potential issues concisely." -f $variant))
    for ($i = 0; $i -lt 40; $i++) {
        $lines.Add("static void worker_stage_$($variant)_$i(context_t * ctx, const batch_t * batch, uint32_t flags) {")
        $lines.Add("    assert(ctx != NULL && ctx->magic == CTX_MAGIC_$($variant)_$i);")
        $lines.Add("    for (uint32_t j = 0; j < batch->count; ++j) { accumulate(&ctx->acc, batch->items[j].value, flags & FLAG_MASK_$i); }")
        $lines.Add("    if (ctx->pending >= THRESHOLD) { flush_stage(ctx, $i); ctx->pending = 0; }")
        $lines.Add("}")
    }
    return ($lines -join "`n") + "`n"
}

# Multi-round: alternate two prompt variants so the router working set shifts a
# little between rounds -> a live LRU should show hit-rate MOVING, not pinned.
for ($rd = 1; $rd -le $Rounds; $rd++) {
    $prompt = Make-Prompt ($rd % 2)
    $body = @{ prompt = $prompt; n_predict = $Predict; temperature = 0.3 } | ConvertTo-Json -Compress
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
        $t = $r.timings
        Write-Output ("{0} r{1}: pp={2:N1} t/s ({3} tok) | tg={4:N2} t/s ({5} tok)" -f `
            $Tag, $rd, $t.prompt_per_second, $r.tokens_evaluated, $t.predicted_per_second, $r.tokens_predicted)
    } catch {
        Write-Output ("{0} r{1}: REQUEST-FAILED {2}" -f $Tag, $rd, $_.Exception.Message)
    }
    Start-Sleep -Seconds 1
    $last = Select-String -Path $errLog -Pattern 'MoE GPU slot stats' -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($last) { Write-Output ("  SLOT: " + $last.Line.Trim()) }
}

Start-Sleep -Seconds $IdleAfter
Write-Output ("{0}: load={1}s alive={2}" -f $Tag, $loadS, (-not $p.HasExited))
$stats = Select-String -Path $errLog -Pattern 'MoE GPU slot stats' -ErrorAction SilentlyContinue | Select-Object -Last 2
foreach ($s in $stats) { Write-Output ("STATS: " + $s.Line.Trim()) }
if (-not $stats) { Write-Output "STATS: (none emitted)  <-- gate lift FAILED if global-LRU/qstar was requested" }

if ($Mode -eq 'qstar') {
    # aggregate the per-step q* decision log into total xfer vs cpu-deferred
    $q = Select-String -Path $errLog -Pattern '\[q\*\] layer=\d+ xfer=(\d+) cpu=(\d+)' -ErrorAction SilentlyContinue
    if ($q) {
        $totX = 0; $totC = 0
        foreach ($m in $q) { $totX += [int]$m.Matches[0].Groups[1].Value; $totC += [int]$m.Matches[0].Groups[2].Value }
        Write-Output ("QSTAR: steps_logged={0} total_xfer={1} total_cpu_deferred={2} cpu_share={3:P1}" -f `
            $q.Count, $totX, $totC, ($(if (($totX + $totC) -gt 0) { $totC / ($totX + $totC) } else { 0 })))
    } else {
        Write-Output "QSTAR: no [q*] decision lines  <-- q* split NOT engaging (check n_tokens==1 decode + calibration)"
    }
}

Stop-Process -Id $p.Id -Force
