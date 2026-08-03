$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH

$model = 'C:\Users\dai86\.lmstudio\models\poolside\Laguna-XS-2.1-GGUF\Laguna-XS-2.1-Q4_K_M.gguf'
$exe = 'C:\Users\dai86\llama-cpp-turboquant\build\bin\llama-cli.exe'
$freq_report = 'C:\Users\dai86\llama-cpp-turboquant\benchmarks\laguna_s_freq_pass1.json'
$outdir = 'C:\Users\dai86\llama-cpp-turboquant'
$prompt = 'The transformer architecture has revolutionized natural language processing by'

# Test 1: basic (all-GPU, no MoE placement)
Write-Host "=== TEST 1: Basic (default all-GPU) ==="
$p1 = Start-Process -FilePath $exe -ArgumentList '-m', $model, '-ngl', '30', '-n', '128', '-p', $prompt, '--simple-io', '--no-disclaim', '--temp', '0', '--no-cnv' -NoNewWindow -RedirectStandardOutput "$outdir\bench_basic.txt" -RedirectStandardError "$outdir\bench_basic_err.txt" -Wait -PassThru
Write-Host "EXIT=$($p1.ExitCode)"
if ($p1.ExitCode -eq 0) {
    Select-String -Path "$outdir\bench_basic_err.txt" -Pattern '( tokens/sec|token/s|milliseconds|秒|ms per token)' | ForEach-Object { Write-Host $_.Line }
}

# Test 2: frequency placement
Write-Host "=== TEST 2: Frequency placement ==="
$p2 = Start-Process -FilePath $exe -ArgumentList '-m', $model, '-ngl', '30', '-n', '128', '-p', $prompt, '--simple-io', '--no-cnv', '--temp', '0', '--moe-expert-placement', 'frequency', '--moe-freq-report-in', $freq_report -NoNewWindow -RedirectStandardOutput "$outdir\bench_freq.txt" -RedirectStandardError "$outdir\bench_freq_err.txt" -Wait -PassThru
Write-Host "FREQ_EXIT=$($p2.ExitCode)"
if ($p2.ExitCode -eq 0) {
    Select-String -Path "$outdir\bench_freq_err.txt" -Pattern '( tokens/sec|token/s|milliseconds|秒|ms per token)' | ForEach-Object { Write-Host $_.Line }
}

Write-Host "=== DONE ==="