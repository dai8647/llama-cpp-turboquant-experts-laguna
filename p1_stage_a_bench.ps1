# P1 baseline decisive bench (review-p1-baseline-decisive-bench-2026-08-26.md @d3663914e).
# Stage-a exact reproduction on the round-2 binary. DO NOT RUN until review side lifts the GO hold.
#
# Arms:
#   a (mandatory): stage-a repro   = slot96, graphs OFF (env), NO glru/q*
#   b (optional) : same as a but GGML_CUDA_DISABLE_GRAPHS removed (graphs ON)
#   c (optional) : A-control repro = slot32 + glru, -c 8192 -t 8, no q*
# Measurement: acceptance-template finalized prompts P1-P3 x3 rounds (r1 discarded)
#              + long 6575-tok generator. Judgment band on median tg of kept rounds.
param(
    [ValidateSet('a','b','c')]
    [string]$Arm = 'a',
    [string]$BinaryPath = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna\build-hip\bin\llama-server.exe',
    [string]$Model = 'C:\Users\dai86\.lmstudio\models\huihui-ai\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf',
    [int]$Port = 8091,
    [string]$BuildNote = ''   # record the confirmed build hash / provenance here
)
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna'
$tag = "P1$Arm"

$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:LLAMA_MOE_SLOT_STATS = '1'
if ($Arm -eq 'b') { Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue }
else { $env:GGML_CUDA_DISABLE_GRAPHS = '1' }

$outLog = "$root\p1_out.log"
$errLog = "$root\p1_err.log"
# preserve previous logs BEFORE truncation (cp-first lesson, @4af9b34ff)
foreach ($f in @($outLog, $errLog)) {
    if (Test-Path $f) { Copy-Item $f "$f.$(Get-Date -Format 'yyyyMMdd-HHmmss').bak" -ErrorAction SilentlyContinue }
}
Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue

$baseArgs = @('-m', $Model, '--host', '127.0.0.1', '--port', "$Port", '--no-webui',
    '-ngl', '999', '-ctk', 'q8_0', '-ctv', 'q8_0', '-fa', 'on', '--cpu-moe')
if ($Arm -in 'a','b') {
    $baseArgs += @('-c', '32768', '-t', '6', '--moe-gpu-expert-slot-num', '96')
} else { # arm c: A-control estimated config
    $baseArgs += @('-c', '8192', '-t', '8', '--moe-gpu-expert-slot-num', '32', '--moe-gpu-expert-global-lru')
}

$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $BinaryPath -ArgumentList $baseArgs `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -WindowStyle Hidden

function Save-Row($row) {
    $row | ConvertTo-Json -Depth 5 -Compress | Add-Content -Path "$root\p1_results.jsonl" -Encoding UTF8
}

$ready = $false
while ($sw.Elapsed.TotalSeconds -lt 300) {
    Start-Sleep -Milliseconds 700
    if ($p.HasExited) { break }
    if ((Get-Content $errLog -Tail 3 -ErrorAction SilentlyContinue) -match 'listening') { $ready = $true; break }
}
$loadS = [math]::Round($sw.Elapsed.TotalSeconds, 1)
$row = [ordered]@{ run = $tag; build_note = $BuildNote; port = $Port; ready = $ready; load_s = $loadS }
$row.slot_mode_lines = ((Select-String -Path $errLog -Pattern 'MoE GPU expert slot mode' -ErrorAction SilentlyContinue | ForEach-Object { $_.Line.Trim() }) -join ' || ')

if (-not $ready) {
    # ctx32K VRAM exhaustion lands HERE by spec: record as a finding, never as silence
    Write-Output ("{0}: LOAD-FAILED t={1}s alive={2}" -f $tag, $loadS, (-not $p.HasExited))
    Get-Content $errLog -Tail 15 -ErrorAction SilentlyContinue
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    $row.ok = $false; $row.err = 'load failed (ctx VRAM failure is itself a recorded finding)'
    Save-Row $row
    exit 1
}

$hz = [Diagnostics.Stopwatch]::StartNew()
while ($hz.Elapsed.TotalSeconds -lt 240) {
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5; if ($h.status -eq 'ok') { break } } catch {}
    Start-Sleep -Milliseconds 900
}

$null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' `
    -Body (@{ prompt = 'Say hello.'; n_predict = 16; temperature = 0 } | ConvertTo-Json) -TimeoutSec 180

# finalized prompts: verbatim from coder-b-qstar-acceptance-template-2026-08-26.md (@b5c1eebd3)
$P1 = @'
Review the following C++ function and point out any issues concisely.

std::vector<int> top_k(const float* scores, size_t n, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) { return scores[a] > scores[b]; });
    idx.resize(k);
    return idx;
}

List at most five findings.
'@
$P2 = 'You are a fixed-income analyst. Explain how a steepening of the 2s10s Treasury curve affects: (1) a leveraged duration-neutral curve trade, (2) negative convexity hedging costs for MBS, and (3) corporate bond relative value screening. Write three short paragraphs, each under four sentences.'
$P3 = "Translate the following Japanese technical passage into natural English, preserving terminology. Provide only the translation.`n`n「expert の一部のみを VRAM に常駐させ、残りは需要発生時にホストから転送する方式では、転送帯域と計算の overlap が性能を左右する。特に prefill 中の大量ページングはレイテンシ急増を招くため、LRU による常駐選択の精度が重要になる。」"
$prompts = @{ P1 = $P1; P2 = $P2; P3 = $P3 }

# long 6575-tok generator (verify_b.ps1 lineage, measured 6575 tok evaluated)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('Review the following C++ code and summarize potential issues concisely.')
for ($i = 0; $i -lt 55; $i++) {
    $lines.Add("static void worker_stage_$i(context_t * ctx, const batch_t * batch, uint32_t flags) {")
    $lines.Add("    assert(ctx != NULL && ctx->magic == CTX_MAGIC_$i);")
    $lines.Add("    for (uint32_t j = 0; j < batch->count; ++j) { accumulate(&ctx->acc, batch->items[j].value, flags & FLAG_MASK_$i); }")
    $lines.Add("    if (ctx->pending >= THRESHOLD) { flush_stage(ctx, $i); ctx->pending = 0; }")
    $lines.Add("}")
}
$LongPrompt = (($lines -join "`n") + "`n")

$kept = New-Object System.Collections.Generic.List[double]
foreach ($k in 'P1','P2','P3') {
    foreach ($rd in 1,2,3) {
        $body = @{ prompt = $prompts[$k]; n_predict = 128; temperature = 0.3 } | ConvertTo-Json -Compress
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
            $tg = [math]::Round($r.timings.predicted_per_second, 2)
            $pp = [math]::Round($r.timings.prompt_per_second, 2)
            Write-Output ("{0} {1} r{2}: pp={3} ({4} tok) tg={5} ({6} tok){7}" -f $tag, $k, $rd, $pp, $r.tokens_evaluated, $tg, $r.tokens_predicted, $(if ($rd -eq 1) {'  [discarded]'} else { '' }))
            $row["$k.r$rd"] = @{ pp = $pp; tg = $tg; ptok = $r.tokens_evaluated; ttok = $r.tokens_predicted }
            if ($rd -gt 1) { $kept.Add($tg) }
        } catch {
            Write-Output ("{0} {1} r{2}: REQUEST-FAILED {3}" -f $tag, $k, $rd, $_.Exception.Message)
            $row["$k.r$rd"] = 'REQUEST-FAILED'
        }
        Start-Sleep -Seconds 1
    }
}

try {
    $body = @{ prompt = $LongPrompt; n_predict = 128; temperature = 0.3 } | ConvertTo-Json -Compress
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
    $row.long = @{ pp = [math]::Round($r.timings.prompt_per_second,2); tg = [math]::Round($r.timings.predicted_per_second,2); ptok = $r.tokens_evaluated }
    Write-Output ("{0} LONG: pp={1:N1} ({2} tok) tg={3:N2} t/s" -f $tag, $r.timings.prompt_per_second, $r.tokens_evaluated, $r.timings.predicted_per_second)
} catch {
    $row.long = "REQUEST-FAILED"
    Write-Output ("{0} LONG: REQUEST-FAILED {1}" -f $tag, $_.Exception.Message)
}

Start-Sleep -Seconds 2
$row.alive = -not $p.HasExited
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue

# --- post-run evidence from stderr ---
$allMoe = Select-String -Path $errLog -Pattern 'MoE GPU' -ErrorAction SilentlyContinue | ForEach-Object { $_.Line.Trim() }
$row.moe_log_lines = ($allMoe | Select-Object -Last 6) -join ' || '
$row.qstar_line_count = (Select-String -Path $errLog -Pattern '\[QSTAR-' -ErrorAction SilentlyContinue | Measure-Object).Count
$lastStats = Select-String -Path $errLog -Pattern 'MoE GPU slot stats' -ErrorAction SilentlyContinue | Select-Object -Last 1
$row.last_slot_stats = if ($lastStats) { $lastStats.Line.Trim() } else { '(none emitted)' }

$s = $kept | Sort-Object
$n = $s.Count
$median = if ($n -eq 0) { 0 } elseif ($n % 2 -eq 1) { $s[[int][math]::Floor($n/2)] } else { [math]::Round(($s[$n/2 - 1] + $s[$n/2]) / 2, 2) }
$row.median_short_tg = $median
$row.judgment = if ($n -lt 6) { "INCOMPLETE (kept=$n)" }
    elseif ($median -ge 11) { 'PASS >=11: no regression; A 4.5 = control-config difference' }
    elseif ($median -le 6) { 'REGRESSION <=6: bisect first, Bench GO frozen indefinitely' }
    else { 'GRAY 6-11: run optional arms + inspect startup log' }

Write-Output ("{0}: median short tg (kept r2/r3, n={1}) = {2} t/s" -f $tag, $n, $median)
Write-Output ("JUDGMENT: {0}" -f $row.judgment)
Write-Output ("QSTAR lines: {0}  (must be 0 with q* off)" -f $row.qstar_line_count)
Write-Output ("last slot stats: {0}" -f $row.last_slot_stats)
Save-Row $row
