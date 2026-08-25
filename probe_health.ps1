param([int]$Port = 8097, [string]$ExtraArgs = "--cpu-moe --no-warmup", [int]$TimeoutSec = 600, [switch]$SlotStats)
$ErrorActionPreference = 'Continue'
$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
Remove-Item Env:LLAMA_MOE_SLOT_STATS -ErrorAction SilentlyContinue
if ($SlotStats) { $env:LLAMA_MOE_SLOT_STATS = '1' }

$root = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna'
Set-Location $root
Remove-Item "$root\probe_err.log","$root\probe_out.log" -ErrorAction SilentlyContinue
$baseArgs = @('-m','C:\Users\dai86\.lmstudio\models\gbuzhf\Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF\Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Mini-v2D-lite.gguf',
    '--host','127.0.0.1','--port',"$Port",'--no-webui','-c','4096','-np','1','-t','6',
    '-fa','on','-ctk','q8_0','-ctv','q8_0')
$extra = if ($ExtraArgs) { $ExtraArgs -split ' ' } else { @() }
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath "$root\build-hip\bin\llama-server.exe" `
    -ArgumentList ($baseArgs + $extra) `
    -RedirectStandardOutput "$root\probe_out.log" -RedirectStandardError "$root\probe_err.log" -PassThru
Write-Output ("launched pid={0} cfg='{1}'" -f $p.Id, $ExtraArgs)

$healthy = $false
while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    Start-Sleep -Seconds 5
    if ($p.HasExited) { Write-Output ("EXITED code={0} at {1:N0}s" -f $p.ExitCode, $sw.Elapsed.TotalSeconds); break }
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
        if ($h.status -eq 'ok') { $healthy = $true; Write-Output ("HEALTHY at {0:N0}s" -f $sw.Elapsed.TotalSeconds); break }
        else { Write-Output ("t={0:N0}s status={1}" -f $sw.Elapsed.TotalSeconds, $h.status) }
    } catch { }
}
if (-not $healthy) { Write-Output ("NOT healthy after {0:N0}s (exited={1})" -f $sw.Elapsed.TotalSeconds, $p.HasExited) }

if ($healthy) {
    $sw2 = [Diagnostics.Stopwatch]::StartNew()
    $body = @{ prompt = "answer in Japanese, two short sentences."; n_predict = 64; temperature = 0.4 } | ConvertTo-Json
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300
        Write-Output ("req ok pred_tps={0} head='{1}'" -f [math]::Round($r.timings.predicted_per_second,2), $r.content.Substring(0,[Math]::Min(60,$r.content.Length)).Replace("`n",' '))
    } catch { Write-Output ("req FAIL: {0}" -f $_.Exception.Message) }
    Write-Output ("req_wall={0:N1}s alive={1}" -f $sw2.Elapsed.TotalSeconds, (-not $p.HasExited))
}
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; Write-Output 'server stopped' }
Write-Output '--- probe_err.log tail ---'
Get-Content "$root\probe_err.log" -Tail 8