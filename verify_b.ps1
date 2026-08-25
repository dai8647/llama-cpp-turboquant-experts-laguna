# Coder B verification: graphs ON (no GGML_CUDA_DISABLE_GRAPHS), long/short prompt,
# optional prefill double buffering and slot count. Appends to bench_results.txt/.jsonl.
param(
    [string]$Tag = 'vb',
    [int]$Port = 8101,
    [string]$ExtraArgs = '',
    [string]$EnvPF = '',
    [int]$Ctx = 8192,
    [int]$Predict = 128,
    [string]$PromptKind = 'long',
    [int]$TimeoutSec = 1500
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
# NOTE: GGML_CUDA_DISABLE_GRAPHS is deliberately NOT set here (graphs ON)

Set-Location $root
$outLog = "$root\vb_out.log"
$errLog = "$root\vb_err.log"
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
    $row | ConvertTo-Json -Depth 4 | Set-Content "$root\vb_last.json" -Encoding UTF8
    Add-Content -Path "$root\bench_results.jsonl" -Value ($row | ConvertTo-Json -Compress) -Encoding UTF8
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

# prompt: short ~2k tok (55 iters) or long ~4.7k tok (38 iters; ~24 tok/line)
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
$err = Get-Content $errLog -Raw -ErrorAction SilentlyContinue
$row.capture_ok = ($err -match 'warmup complete') -and ($err -notmatch 'operation not permitted')
$row.crash_marker = [bool]($err -match 'operation not permitted|hipError|HIP error|abort')
$row.graph_lines = (($err -split "`n") | Select-String -Pattern 'graph|warmup' | ForEach-Object { $_.Line.Trim() } | Select-Object -Last 4) -join ' || '
$row.pf_lines = (($err -split "`n") | Select-String -Pattern 'prefill|auto:|slot' | ForEach-Object { $_.Line.Trim() } | Select-Object -Last 6) -join ' || '
$row.stats = (Select-String -Path $errLog -Pattern 'MoE GPU slot stats' -ErrorAction SilentlyContinue | Select-Object -Last 1 | ForEach-Object { $_.Line.Trim() })

if ($row.ok -and $row.capture_ok) { Write-Output "GRAPH-FIX: OK (captured, no crash)" }
elseif (-not $row.ok) { Write-Output "GRAPH-FIX: REQUEST FAILED" }
else { Write-Output "GRAPH-FIX: capture=$($row.capture_ok) crash_marker=$($row.crash_marker)" }

Add-Content -Path "$root\bench_results.jsonl" -Value ($row | ConvertTo-Json -Compress) -Encoding UTF8
Add-Content -Path "$root\bench_results.txt" -Value ("{0}: extra='{1}' pf={2} prompt={3} ctx={4} | pp={5} t/s ({6} tok) tg={7} t/s ({8} tok) load={9}s capture_ok={10} crash={11} alive={12}" -f `
    $Tag, $ExtraArgs, $EnvPF, $PromptKind, $Ctx, $row.pp_tps, $row.pp_tokens, $row.tg_tps, $row.tg_tokens, $loadS, $row.capture_ok, $row.crash_marker, $row.alive) -Encoding UTF8

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
$row | ConvertTo-Json -Depth 4 | Set-Content "$root\vb_last.json" -Encoding UTF8
Write-Output ($row | ConvertTo-Json -Depth 4)
Write-Output '--- vb_err.log tail (graph/prefill lines) ---'
Get-Content $errLog -Tail 25 -ErrorAction SilentlyContinue | Select-String -Pattern 'graph|warmup|prefill|slot|error|Error|abort' | ForEach-Object { $_.Line }
