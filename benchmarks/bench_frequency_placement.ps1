# frequency配置の性能比較ベンチマーク
# Conditions: CPU baseline, HIP slot-disabled, HIP frequency placement
# Each: 3 runs, measure timing

$hip_cli = if (Test-Path "build-hip\bin\llama-cli.exe") {
    "build-hip\bin\llama-cli.exe"
} else {
    "C:\Users\dai86\llama-cpp-turboquant\build-hip\bin\llama-cli.exe"
}
$cpu_cli = if (Test-Path "build-cpu\bin\Debug\llama-cli.exe") {
    "build-cpu\bin\Debug\llama-cli.exe"
} else {
    "C:\Users\dai86\llama-cpp-turboquant\build-cpu\bin\Debug\llama-cli.exe"
}
$model = if (Test-Path "models\DeepSeek-V2-Lite.Q4_K_M.gguf") {
    "models\DeepSeek-V2-Lite.Q4_K_M.gguf"
} else {
    "C:\Users\dai86\llama-cpp-turboquant\models\DeepSeek-V2-Lite.Q4_K_M.gguf"
}
$rocmbin = "C:\Program Files\AMD\ROCm\7.1\bin"
$prompt = "Explain the concept of Mixture of Experts (MoE) in large language models. Include details about how expert routing works, the benefits of sparse activation, and how it differs from dense transformer architectures. Provide a concrete example with a model that has 8 experts and a top-2 routing strategy."
$n_predict = 128
$seed = 42
$runs = 3
$results = @()

# Add ROCm to PATH
$env:PATH = "$rocmbin;" + $env:PATH
$env:ROCM_PATH = "C:\Program Files\AMD\ROCm\7.1"

function Run-Bench {
    param($name, $cli, $extra_args, $runNum)
    Write-Host "=== $name Run $runNum ===" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $output = & $cli -m $model @extra_args -n $n_predict -p $prompt --temp 0 -s $seed 2>&1
    $sw.Stop()
    $total_sec = $sw.Elapsed.TotalSeconds

    $prompt_ts = 0.0
    $gen_ts = 0.0
    foreach ($line in $output) {
        if ($line -match "Prompt:\s+([\d.]+)\s+t/s") {
            $prompt_ts = [double]$Matches[1]
        }
        if ($line -match "Generation:\s+([\d.]+)\s+t/s") {
            $gen_ts = [double]$Matches[1]
        }
    }

    Write-Host "  Prompt: $prompt_ts t/s | Gen: $gen_ts t/s | Total: $([math]::Round($total_sec, 1))s"

    return @{
        Name = $name
        Run = $runNum
        PromptTS = $prompt_ts
        GenTS = $gen_ts
        TotalSec = $total_sec
    }
}

Write-Host "=== Frequency Placement Benchmark ===" -ForegroundColor Green
Write-Host "Model: $model"
Write-Host "Prompt: $($prompt.Substring(0,50))..."
Write-Host "n_predict: $n_predict"
Write-Host "Repeats: $runs"
Write-Host ""

# CPU baseline
for ($i = 1; $i -le $runs; $i++) {
    $r = Run-Bench "CPU" $cpu_cli @("-ngl", "0") $i
    $results += $r
    Start-Sleep -Seconds 2
}

# HIP slot-disabled
for ($i = 1; $i -le $runs; $i++) {
    $r = Run-Bench "HIP-slot-off" $hip_cli @("-ngl", "999") $i
    $results += $r
    Start-Sleep -Seconds 2
}

# HIP frequency placement (ratio=0.6)
for ($i = 1; $i -le $runs; $i++) {
    $r = Run-Bench "HIP-freq-0.6" $hip_cli @("-ngl", "999", "--moe-expert-placement", "frequency", "--moe-gpu-expert-ratio", "0.6") $i
    $results += $r
    Start-Sleep -Seconds 2
}

# HIP full-slot (ratio=1.0)
for ($i = 1; $i -le $runs; $i++) {
    $r = Run-Bench "HIP-freq-1.0" $hip_cli @("-ngl", "999", "--moe-expert-placement", "frequency", "--moe-gpu-expert-ratio", "1.0") $i
    $results += $r
    Start-Sleep -Seconds 2
}

# Summary
Write-Host "`n=== SUMMARY ===" -ForegroundColor Green
$results | Format-Table -AutoSize

# Medians
Write-Host "`n=== MEDIANS ===" -ForegroundColor Yellow
foreach ($name in @("CPU", "HIP-slot-off", "HIP-freq-0.6", "HIP-freq-1.0")) {
    $group = @($results | Where-Object { $_.Name -eq $name })
    if ($group.Count -gt 0) {
        $sorted_p = $group | Sort-Object PromptTS
        $sorted_g = $group | Sort-Object GenTS
        $idx = [math]::Floor($group.Count / 2)
        $prompt_median = $sorted_p[$idx].PromptTS
        $gen_median = $sorted_g[$idx].GenTS
        Write-Host "$name : Prompt=$prompt_median t/s | Gen=$gen_median t/s"
    }
}

# Save results
$results | ConvertTo-Json | Out-File "benchmarks\frequency_placement_results.json"
Write-Host "`nResults saved to benchmarks\frequency_placement_results.json"
