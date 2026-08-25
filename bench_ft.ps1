param(
    [int]$Port = 8095,
    [int]$Ctx = 32768,
    [int]$Predict = 256,
    [string]$ExtraArgs = "",
    [string]$Tag = "run",
    [int]$IdleAfter = 2,
    [System.Collections.IDictionary]$Env2 = @{}
)
$ErrorActionPreference = 'Continue'
$root  = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna'
$model = 'C:\Users\dai86\.lmstudio\models\huihui-ai\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf'

$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:LLAMA_MOE_SLOT_STATS = '1'
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
Remove-Item Env:LLAMA_MOE_PREFETCH_MS -ErrorAction SilentlyContinue
foreach ($k in $Env2.Keys) { Set-Item -Path ("Env:" + $k) -Value $Env2[$k] }

Set-Location $root
$outLog = "$root\ft_out.log"
$errLog = "$root\ft_err.log"
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

# realistic agent-style long prompt (~2k tokens of ASCII code)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('Review the following C++ code and summarize potential issues concisely.')
for ($i = 0; $i -lt 55; $i++) {
    $lines.Add("static void worker_stage_$i(context_t * ctx, const batch_t * batch, uint32_t flags) {")
    $lines.Add("    assert(ctx != NULL && ctx->magic == CTX_MAGIC_$i);")
    $lines.Add("    for (uint32_t j = 0; j < batch->count; ++j) { accumulate(&ctx->acc, batch->items[j].value, flags & FLAG_MASK_$i); }")
    $lines.Add("    if (ctx->pending >= THRESHOLD) { flush_stage(ctx, $i); ctx->pending = 0; }")
    $lines.Add("}")
}
$prompt = ($lines -join "`n") + "`n"

$body = @{ prompt = $prompt; n_predict = $Predict; temperature = 0.3 } | ConvertTo-Json -Compress
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
    $t = $r.timings
    Write-Output ("{0}: pp={1:N1} t/s ({2} tok) | tg={3:N2} t/s ({4} tok) | load={5}s alive={6}" -f `
        $Tag, $t.prompt_per_second, $r.tokens_evaluated, $t.predicted_per_second, $r.tokens_predicted, $loadS, (-not $p.HasExited))
    $head = $r.content
    if ($head.Length -gt 80) { $head = $head.Substring(0, 80) }
    Write-Output ("head: " + ($head -replace "\r?\n", ' | '))
} catch {
    Write-Output ("{0}: REQUEST-FAILED {1}" -f $Tag, $_.Exception.Message)
}

Start-Sleep -Seconds $IdleAfter
$stats = Select-String -Path $errLog -Pattern 'MoE GPU slot stats' -ErrorAction SilentlyContinue | Select-Object -Last 2
foreach ($s in $stats) { Write-Output ("STATS: " + $s.Line.Trim()) }
if (-not $stats) { Write-Output "STATS: (none emitted)" }

Stop-Process -Id $p.Id -Force
