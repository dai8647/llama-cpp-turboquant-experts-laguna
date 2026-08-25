param([int]$Port = 8093, [int]$Predict = 64, [string]$ExtraArgs = "")
$ErrorActionPreference = 'Continue'
$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:LLAMA_MOE_SLOT_STATS = '1'
Remove-Item Env:LLAMA_MOE_PREFETCH_MS -ErrorAction SilentlyContinue

$root = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna'
Set-Location $root
Remove-Item "$root\diag_err.log" -ErrorAction SilentlyContinue
$baseArgs = @('-m','C:\Users\dai86\.lmstudio\models\gbuzhf\Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF\Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Mini-v2D-lite.gguf',
    '--host','127.0.0.1','--port',"$Port",'--no-webui','-c','4096','-np','1','-t','6',
    '-fa','on','-ctk','q8_0','-ctv','q8_0')
$extra = if ($ExtraArgs) { $ExtraArgs -split ' ' } else { @() }
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath "$root\build-hip\bin\llama-server.exe" `
    -ArgumentList ($baseArgs + $extra) `
    -RedirectStandardOutput "$root\diag_out.log" -RedirectStandardError "$root\diag_err.log" -PassThru

$ready = $false
while ($sw.Elapsed.TotalSeconds -lt 90) {
    Start-Sleep -Milliseconds 800
    if ($p.HasExited) { break }
    try { $tcp = New-Object Net.Sockets.TcpClient; $tcp.Connect('127.0.0.1', $Port); $tcp.Close(); $ready = $true; break } catch {}
}
Write-Output ("startup: ready={0} t={1:N1}s alive={2}" -f $ready, $sw.Elapsed.TotalSeconds, (-not $p.HasExited))

if ($ready) {
    foreach ($i in 1..3) {
        $body = @{ prompt = "test $i : answer in Japanese, two short sentences."; n_predict = $Predict; temperature = 0.4 } | ConvertTo-Json
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
            $head = $r.content.Substring(0,[Math]::Min(60,$r.content.Length)).Replace("`n",' ')
            Write-Output ("req{0}: ok pred_tps={1} head='{2}'" -f $i, [math]::Round($r.timings.predicted_per_second,2), $head)
        } catch {
            Write-Output ("req{0}: FAIL {1}" -f $i, $_.Exception.Message)
        }
        Write-Output ("  alive_after_req{0}={1}" -f $i, (-not $p.HasExited))
        if ($p.HasExited) { break }
    }
}

Start-Sleep -Seconds 2
Write-Output ("final_alive={0}" -f (-not $p.HasExited))
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
Write-Output '--- diag_err.log tail ---'
Get-Content "$root\diag_err.log" -Tail 12
