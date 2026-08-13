param(
  [string]$Report = 'freq_dsv4.json',
  [int]$N = 500
)
$ErrorActionPreference = 'Stop'
$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH

$root = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna'
$model = 'C:\Users\dai86\.lmstudio\models\puwaer\DeepSeek-V4-Flash-0731-reap-150b-gguf\DeepSeek-V4-Flash-0731-reap-150b-Q3_K_M.gguf'
Set-Location $root

$outLog = "$root\pass1_srv_out.log"; $errLog = "$root\pass1_srv_err.log"
Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue
Remove-Item "$root\$Report" -ErrorAction SilentlyContinue

$args = @('-m', $model, '--host', '127.0.0.1', '--port', '8092', '--no-webui', '-lv', '3',
          '-c', '8192', '-np', '1', '-t', '6',
          '--cpu-moe',
          '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0',
          '--moe-gpu-expert-slot-num', '30',
          '--moe-freq-report-out', $Report,
          '--sleep-idle-seconds', '10')

$server = Start-Process -FilePath "$root\build-hip\bin\llama-server.exe" -ArgumentList $args -WorkingDirectory $root -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$serverId = $server.Id

$ready = $false
for ($i = 0; $i -lt 240; $i++) {
    if (-not (Get-Process -Id $serverId -ErrorAction SilentlyContinue)) { break }
    if (Test-Path $errLog) {
        if ((Get-Content $errLog -Raw -ErrorAction SilentlyContinue) -match 'listening on http') { $ready = $true; break }
    }
    Start-Sleep -Seconds 2
}

$row = [ordered]@{ run = 'pass1-freq-server'; report = $Report; n = $N; ready = $ready }

if ($ready) {
    $body = @{ prompt = '与えられた課題に対して、論理的で実用的な回答をしてください。まず全体方針を述べ、次に具体的な手順、最後に注意点を簡潔にまとめてください。'; n_predict = $N; temperature = 0.4 } | ConvertTo-Json
    $bodyFile = "$root\pass1_body.json"
    $respFile = "$root\pass1_resp.json"
    $body | Set-Content -Path $bodyFile -Encoding UTF8
    & curl.exe -s --max-time 1500 -H "Content-Type: application/json" -d "@$bodyFile" "http://127.0.0.1:8092/completion" -o $respFile
    if ($LASTEXITCODE -eq 0 -and (Test-Path $respFile)) {
        $r = Get-Content $respFile -Raw | ConvertFrom-Json
        $t = $r.timings
        $row.pred_tps = [math]::Round($t.predicted_per_second, 3)
        $row.prompt_tps = [math]::Round($t.prompt_per_second, 3)
        $row.ok = $true
    } else {
        $row.ok = $false; $row.err = "curl exit=$LASTEXITCODE"
    }
} else {
    $row.ok = $false; $row.err = 'server not ready'
}

# wait for the frequency report (written on sleep/destroy)
$reportSeen = $false
for ($i = 0; $i -lt 60; $i++) {
    if (Test-Path "$root\$Report") { $reportSeen = $true; break }
    Start-Sleep -Seconds 2
}
$row.report_seen = $reportSeen
if ($reportSeen) {
    $row.report_bytes = (Get-Item "$root\$Report").Length
}

# stop server
Stop-Process -Id $serverId -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

$row | ConvertTo-Json -Depth 4 | Set-Content -Path "$root\pass1_last.json" -Encoding UTF8
Write-Output ($row | ConvertTo-Json -Depth 4)
Add-Content -Path "$root\bench_results.jsonl" -Value ($row | ConvertTo-Json -Compress) -Encoding UTF8