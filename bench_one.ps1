param(
  [string]$Name = 'default',
  [string]$ExtraArgs = '',
  [int]$Predict = 200
)
$ErrorActionPreference = 'Stop'
$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH

$root = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna'
$model = 'C:\Users\dai86\.lmstudio\models\puwaer\DeepSeek-V4-Flash-0731-reap-150b-gguf\DeepSeek-V4-Flash-0731-reap-150b-Q3_K_M.gguf'
Set-Location $root

# --- build full server arg list ---
$baseArgs = @('-m', $model, '--host', '127.0.0.1', '--port', '8091', '--no-webui', '-lv', '4')
$extra = if ($ExtraArgs) { ($ExtraArgs -split ' ') } else { @() }
$allArgs = $baseArgs + $extra

$outLog = "$root\bench_out.log"; $errLog = "$root\bench_err.log"
Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue

# --- start server ---
$server = Start-Process -FilePath "$root\build-hip\bin\llama-server.exe" -ArgumentList $allArgs -WorkingDirectory $root -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$serverId = $server.Id

# --- wait for "listening" ---
$ready = $false
for ($i = 0; $i -lt 240; $i++) {
    if (-not (Get-Process -Id $serverId -ErrorAction SilentlyContinue)) { break }
    if (Test-Path $errLog) {
        if ((Get-Content $errLog -Raw -ErrorAction SilentlyContinue) -match 'listening on http') { $ready = $true; break }
    }
    Start-Sleep -Seconds 2
}

$row = [ordered]@{ run = $Name; extra = $ExtraArgs; ready = $ready }

if ($ready) {
    # --- send one Japanese completion ---
    $body = @{ prompt = '日本語のモデルとして正しく動作していますか？3文で自己紹介してください。'; n_predict = $Predict; temperature = 0.4 } | ConvertTo-Json
    try {
        $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8091/completion' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
        $t = $r.timings
        $row.pred_tps = [math]::Round($t.predicted_per_second, 3)
        $row.prompt_tps = [math]::Round($t.prompt_per_second, 3)
        $row.pred = $r.tokens_predicted
        $row.eval = $r.tokens_evaluated
        $row.ok = $true
    } catch {
        $row.ok = $false; $row.err = $_.Exception.Message
    }
} else {
    $row.ok = $false; $row.err = 'server not ready / load fail'
}

# --- capture fit/model info from log ---
$modelInfo = ''
if (Test-Path $errLog) {
    $g = Get-Content $errLog | Select-String -Pattern 'offload|n_layer|n_expert|memory breakdown|context size reduced|ROCm' | Select-Object -Last 6 | ForEach-Object { $_.Line.Trim() }
    $modelInfo = ($g -join ' | ')
}
$row.info = $modelInfo

# --- append to log ---
$line = ($row | ConvertTo-Json -Compress)
Add-Content -Path "$root\bench_results.jsonl" -Value $line -Encoding UTF8
# also human readable
Add-Content -Path "$root\bench_results.txt" -Value ("{0}`tpred={1} tps`tpp={2} tps`t{3}" -f $Name, $row.pred_tps, $row.prompt_tps, $ExtraArgs) -Encoding UTF8

# --- stop server ---
Stop-Process -Id $serverId -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# write result file for polling
$row | ConvertTo-Json -Depth 4 | Set-Content -Path "$root\bench_last.json" -Encoding UTF8
