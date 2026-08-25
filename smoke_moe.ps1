param(
  [string]$Name = 'smoke',
  [string]$Model = 'C:\Users\dai86\.lmstudio\models\gbuzhf\Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF\Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Mini-v2D-lite.gguf',
  [string]$ExtraArgs = '',
  [int]$PrefetchMs = 0,
  [int]$Predict = 128
)
$ErrorActionPreference = 'Stop'
$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:LLAMA_MOE_SLOT_STATS = '1'
if ($PrefetchMs -gt 0) { $env:LLAMA_MOE_PREFETCH_MS = "$PrefetchMs" } else { Remove-Item Env:LLAMA_MOE_PREFETCH_MS -ErrorAction SilentlyContinue }

$root = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna'
Set-Location $root
$baseArgs = @('-m', $Model, '--host', '127.0.0.1', '--port', '8093', '--no-webui', '-lv', '4',
              '-c', '4096', '-np', '1', '-t', '6',
              '--cpu-moe', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0')
$extra = if ($ExtraArgs) { ($ExtraArgs -split ' ') } else { @() }
$allArgs = $baseArgs + $extra

$outLog = "$root\smoke_out.log"; $errLog = "$root\smoke_err.log"
Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue
$server = Start-Process -FilePath "$root\build-hip\bin\llama-server.exe" -ArgumentList $allArgs -WorkingDirectory $root -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$sid = $server.Id

$ready = $false
for ($i = 0; $i -lt 240; $i++) {
    if (-not (Get-Process -Id $sid -ErrorAction SilentlyContinue)) { break }
    if ((Get-Content $errLog -Raw -ErrorAction SilentlyContinue) -match 'listening on http') { $ready = $true; break }
    Start-Sleep -Seconds 2
}
$row = [ordered]@{ run = $Name; extra = $ExtraArgs; prefetch_ms = $PrefetchMs; ready = $ready }

if ($ready) {
    # warm-up
    $w = @{ prompt = 'hello'; n_predict = 16; temperature = 0.0 } | ConvertTo-Json
    try { Invoke-RestMethod -Uri 'http://127.0.0.1:8093/completion' -Method Post -ContentType 'application/json' -Body $w -TimeoutSec 600 | Out-Null } catch {}
    # measured
    $body = @{ prompt = '日本語のモデルとして正しく動作していますか？3文で自己紹介してください。'; n_predict = $Predict; temperature = 0.4 } | ConvertTo-Json
    try {
        $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8093/completion' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
        $t = $r.timings
        $row.pred_tps = [math]::Round($t.predicted_per_second, 3)
        $row.prompt_tps = [math]::Round($t.prompt_per_second, 3)
        $row.content_head = $r.content.Substring(0, [Math]::Min(120, $r.content.Length))
        $row.ok = $true
    } catch { $row.ok = $false; $row.err = $_.Exception.Message }
} else { $row.ok = $false; $row.err = 'server not ready / load fail' }

# telemetry lines
$tele = Get-Content $errLog -ErrorAction SilentlyContinue | Select-String -Pattern 'slot|moe_gpu|hit.*miss|prefetch' | ForEach-Object { $_.Line.Trim() } | Select-Object -Last 8
$row.telemetry = ($tele -join ' || ')

Add-Content -Path "$root\bench_results.jsonl" -Value ($row | ConvertTo-Json -Compress) -Encoding UTF8
Stop-Process -Id $sid -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
$row | ConvertTo-Json -Depth 4 | Set-Content -Path "$root\smoke_last.json" -Encoding UTF8
Write-Output ($row | ConvertTo-Json -Depth 4)
