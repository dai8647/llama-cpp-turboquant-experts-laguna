# Laguna-S 118B frequency placement benchmark runner
# Usage: powershell -File laguna_runner.ps1 <label> <extra_args_csv>
param(
    [string]$Label,
    [string]$ExtraArgsCsv = ""
)

$ErrorActionPreference = "Continue"
$cli = "C:\Users\dai86\llama-cpp-turboquant\build-hip\bin\llama-cli.exe"
$model = "C:\Users\dai86\.lmstudio\models\unsloth\Laguna-S-2.1-GGUF\Laguna-S-2.1-UD-IQ3_XXS.gguf"
$promptFile = "C:\Users\dai86\llama-cpp-turboquant\benchmarks\laguna_prompt.txt"
$rocmbin = "C:\Program Files\AMD\ROCm\7.1\bin"
$env:PATH = "$rocmbin;" + $env:PATH
$env:ROCM_PATH = "C:\Program Files\AMD\ROCm\7.1"

# Convert CSV extra args to array (each arg separated by comma; quotes handled simply)
$extraArgs = @()
if ($ExtraArgsCsv -ne "") {
    $extraArgs = $ExtraArgsCsv -split ","
}

$baseArgs = @("-m", $model, "-ngl", "999", "-n", "128", "-f", $promptFile, "--temp", "0", "-s", "42")
$allArgs = $baseArgs + $extraArgs

$log = "$env:TEMP\laguna_bench_$Label.log"
Write-Host "=== $Label : $($allArgs -join ' ') ==="

$proc = Start-Process -FilePath $cli -ArgumentList $allArgs `
    -RedirectStandardError $log -RedirectStandardOutput "$log.out" `
    -PassThru -NoNewWindow

# Wait up to 12 min
$deadline = (Get-Date).AddMinutes(12)
while (-not $proc.HasExited -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 10
    $proc.Refresh()
}
if (-not $proc.HasExited) {
    Write-Host "TIMEOUT - killing"
    $proc.Kill()
    exit 2
}

Write-Host "EXIT: $($proc.ExitCode)"

# Extract timings
$prompt_ts = $null
$gen_ts = $null
Get-Content $log -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_ -match "Prompt:\s+([\d.]+)\s+t/s") { $prompt_ts = $Matches[1] }
    if ($_ -match "Generation:\s+([\d.]+)\s+t/s") { $gen_ts = $Matches[1] }
}
Write-Host "Prompt: $prompt_ts t/s | Generation: $gen_ts t/s"
Write-Host "RESULT $Label prompt=$prompt_ts gen=$gen_ts"
