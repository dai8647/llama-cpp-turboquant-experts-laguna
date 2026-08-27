# coder A crash-triage runner, derived from verify_b.ps1.
# differences: records llama-server ExitCode + WER dump presence so a silent
# death can be classified as access-violation family (H1/H5) vs hang (H4)
# without admin rights / page heap. Same prompts and flags as verify_b so the
# results stay comparable with B's rows in bench_results.txt.
param(
    [string]$Tag = 'va',
    [int]$Port = 8101,
    [string]$ExtraArgs = '',
    [string]$EnvPF = '',
    [int]$Ctx = 8192,
    [int]$Predict = 128,
    [string]$PromptKind = 'long',
    [int]$TimeoutSec = 1500,
    [switch]$E1 = $false
)
$ErrorActionPreference = 'Continue'
$root  = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna'
$model = 'C:\Users\dai86\.lmstudio\models\huihui-ai\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf'

$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:LLAMA_MOE_SLOT_STATS = '1'
if ($EnvPF -eq '1') { $env:LLAMA_MOE_PREFILL_PF = '1' } else { Remove-Item Env:LLAMA_MOE_PREFILL_PF -ErrorAction SilentlyContinue }
if ($E1) { $env:LLAMA_MOE_PREFETCH_MS = '4.0' }
# NOTE: GGML_CUDA_DISABLE_GRAPHS is deliberately NOT set here (graphs ON)

Set-Location $root
$outLog = "$root\va_out.log"
$errLog = "$root\va_err.log"
Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue

$baseArgs = @('-m', $model, '--host', '127.0.0.1', '--port', "$Port", '--no-webui',
    '-lv', '4', '-c', "$Ctx", '-np', '1', '-t', '6', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '--cpu-moe')
$extra = if ($ExtraArgs) { $ExtraArgs -split ' ' } else { @() }
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath "$root\build-hip\bin\llama-server.exe" -ArgumentList ($baseArgs + $extra) `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -WindowStyle Hidden

$ready = $false
while ($sw.Elapsed.TotalSeconds -lt 300) {
    Start-Sleep -Milliseconds 700
    if ($p.HasExited) { break }
    if ((Get-Content $errLog -Tail 3 -ErrorAction SilentlyContinue) -match 'listening') { $ready = $true; break }
}
$loadS = [math]::Round($sw.Elapsed.TotalSeconds, 1)
$row = [ordered]@{ run = $Tag; extra = $ExtraArgs; pf = $EnvPF; prompt = $PromptKind; ctx = $Ctx; ready = $ready; load_s = $loadS }

if (-not $ready) {
    Write-Output ("{0}: LOAD-FAILED t={1}s alive={2}" -f $Tag, $loadS, (-not $p.HasExited))
    Get-Content $errLog -Tail 15 -ErrorAction SilentlyContinue
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    $row.ok = $false; $row.err = 'load failed'
    $row | ConvertTo-Json -Depth 4 | Set-Content "$root\va_last.json" -Encoding UTF8
    exit 1
}

# wait until /health reports ok (weights fully mapped)
$hz = [Diagnostics.Stopwatch]::StartNew()
while ($hz.Elapsed.TotalSeconds -lt 240) {
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5; if ($h.status -eq 'ok') { break } } catch {}
    Start-Sleep -Milliseconds 900
}

# warmup (short)
$null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' `
    -Body (@{ prompt = 'Say hello.'; n_predict = 16; temperature = 0 } | ConvertTo-Json) -TimeoutSec 180

# PromptKind naming footgun (matches verify_b.ps1 line 67-68 by design):
#   -PromptKind short -> 55 iters ~6575 tok  (B's "1a 確実再現" repro size, large prompt)
#   -PromptKind long  -> 38 iters ~4535 tok  (B's "再現保証なし" smaller size)
# The script inherits the counter-intuitive names from verify_b.ps1 unchanged
# (renaming here would break A/B tool symmetry; fix at verify_b if ever).
$iters = if ($PromptKind -eq 'long') { 38 } else { 55 }
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('Review the following C++ code and summarize potential issues concisely.')
for ($i = 0; $i -lt $iters; $i++) {
    $lines.Add("static void worker_stage_$i(context_t * ctx, const batch_t * batch, uint32_t flags) {")
    $lines.Add("    assert(ctx != NULL && ctx->magic == CTX_MAGIC_$i);")
    $lines.Add("    for (uint32_t j = 0; j < batch->count; ++j) { accumulate(&ctx->acc, batch->items[j].value, flags & FLAG_MASK_$i); }")
    $lines.Add("    if (ctx->pending >= THRESHOLD) { flush_stage(ctx, $i); ctx->pending = 0; }")
    $lines.Add("}")
}
$prompt = ($lines -join "`n") + "`n"

$body = @{ prompt = $prompt; n_predict = $Predict; temperature = 0.3 } | ConvertTo-Json -Compress
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec $TimeoutSec
    $t = $r.timings
    $row.pp_tps = [math]::Round($t.prompt_per_second, 2)
    $row.tg_tps = [math]::Round($t.predicted_per_second, 2)
    $row.pp_tokens = $r.tokens_evaluated
    $row.tg_tokens = $r.tokens_predicted
    $row.ok = $true
    Write-Output ("{0}: pp={1:N1} t/s ({2} tok) | tg={3:N2} t/s ({4} tok) | load={5}s" -f `
        $Tag, $t.prompt_per_second, $r.tokens_evaluated, $t.predicted_per_second, $r.tokens_predicted, $loadS)
} catch {
    $row.ok = $false; $row.err = $_.Exception.Message
    Write-Output ("{0}: REQUEST-FAILED {1}" -f $Tag, $_.Exception.Message)
}

Start-Sleep -Seconds 2
$row.alive = -not $p.HasExited
if ($p.HasExited) {
    # triage: 0xC0000005 = access violation (heap corruption family H1/H5),
    # other nonzero = see docs, process alive at timeout = hang family (H4)
    $row.exit_code = ('0x{0:X8}' -f ($p.ExitCode -band 0xFFFFFFFF))
}
$row.dumps = @(Get-ChildItem "$root\dumps" -Filter '*.dmp' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 2 |
    ForEach-Object { '{0} {1:N1}MB' -f $_.Name, ($_.Length/1MB) }) -join ', '
$err = Get-Content $errLog -Raw -ErrorAction SilentlyContinue
$row.capture_ok = ($err -match 'warmup complete') -and ($err -notmatch 'operation not permitted')
$row.crash_marker = [bool]($err -match 'operation not permitted|hipError|HIP error|abort')
$row.last_layer = (($err -split "`n") | Select-String -Pattern 'MoE GPU slot stats' |
    ForEach-Object { $_.Line.Trim() } | Select-Object -Last 1)
$row.stats = $row.last_layer
$e1 = $err | Select-String -Pattern '\[E1-PREFETCH-HIT\] h=([0-9.]+)' -AllMatches -ErrorAction SilentlyContinue
if ($e1) {
    $hs = @(); foreach ($m in $e1) { foreach ($mm in $m.Matches) { $hs += [double]$mm.Groups[1].Value } }
    $row.e1_h = [math]::Round(($hs | Measure-Object -Average).Average, 3)
    $row.e1_windows = $hs.Count
    Write-Output ("E1: windows={0} h_avg={1:F3}" -f $hs.Count, $row.e1_h)
} else { $row.e1_h = $null }

if (-not $row.alive) {
    Write-Output ("{0}: PROCESS-DIED exit_code={1} dumps=[{2}]" -f $Tag, $row.exit_code, $row.dumps)
    Write-Output ("{0}: last slot stats: {1}" -f $Tag, $row.last_layer)
}

Add-Content -Path "$root\bench_results.txt" -Value ("{0}: extra='{1}' pf={2} prompt={3} ctx={4} | pp={5} t/s ({6} tok) tg={7} t/s ({8} tok) load={9}s capture_ok={10} crash={11} alive={12} exit_code={13} e1={14}" -f `
    $Tag, $ExtraArgs, $EnvPF, $PromptKind, $Ctx, $row.pp_tps, $row.pp_tokens, $row.tg_tps, $row.tg_tokens, $loadS, $row.capture_ok, $row.crash_marker, $row.alive, $row.exit_code, $row.e1_h) -Encoding UTF8

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
$row | ConvertTo-Json -Depth 4 | Set-Content "$root\va_last.json" -Encoding UTF8
Write-Output ($row | ConvertTo-Json -Depth 4)
