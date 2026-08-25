$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
$env:LLAMA_MOE_SLOT_STATS = '1'
$env:LLAMA_MOE_QSTAR_STATS = '1'
$outLog = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna\hq11_out.log'
$errLog = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna\hq11_err.log'
if (Test-Path $outLog) { Remove-Item $outLog }
if (Test-Path $errLog) { Remove-Item $errLog }
$args = @(
    '-m', 'C:\Users\dai86\.lmstudio\models\gbuzhf\Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF\Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Mini-v2D-lite.gguf',
    '--host', '127.0.0.1', '--port', '8095', '--no-webui',
    '-lv', '4', '-c', '4096', '-np', '1', '--cpu-moe',
    '--moe-gpu-expert-slot-num', '96', '--moe-qstar'
)
$proc = Start-Process -FilePath 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna\build-hip\bin\llama-server.exe' -ArgumentList $args -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -WindowStyle Hidden
Write-Output ('launched pid=' + $proc.Id)
