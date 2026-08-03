# Generate Laguna-S 118B pass1 MoE expert frequency report.
# Runs the S bench workload (fixed prompt + N gen tokens) with access tracking on
# and writes the frequency report JSON. Uses the Release gfx1101 build only.
#
# NOTE: llama-cli only writes the report when launched via cmd /c with direct file
# redirection. Under Start-Process (pipe redirection) the process exits before the
# save step, so do NOT use Start-Process here.
#
# Usage:
#   powershell -File benchmarks\gen_s118b_freq_report.ps1 [-N 256] [-Out laguna_s_freq_pass1.json]

param(
    [int]$N = 256,
    [string]$Out = "C:\Users\dai86\llama-cpp-turboquant\benchmarks\laguna_s_freq_pass1.json"
)

$ErrorActionPreference = "Continue"

$model = "C:\Users\dai86\.lmstudio\models\unsloth\Laguna-S-2.1-GGUF\Laguna-S-2.1-UD-IQ3_XXS.gguf"
$promptFile = "C:\Users\dai86\llama-cpp-turboquant\benchmarks\laguna_prompt.txt"
$bin = "C:\Users\dai86\llama-cpp-turboquant\build-hip-rel\bin"
$cli = "$bin\llama-cli.exe"
if (-not (Test-Path $cli)) {
    Write-Host "ERROR: Release build not found at $cli" -ForegroundColor Red
    exit 2
}

$rocmbin = "C:\Program Files\AMD\ROCm\7.1\bin"
$env:PATH = "$rocmbin;" + $env:PATH
$env:ROCM_PATH = "C:\Program Files\AMD\ROCm\7.1"

$outLog = "$env:TEMP\genfreq_out.log"
$errLog = "$env:TEMP\genfreq_err.log"
Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue

$argsLine = "-m `"$model`" -fa on -ngl 999 -c 4096 -n $N -f `"$promptFile`" --temp 0 -s 42 --simple-io -ub 256 -lv 3 --moe-freq-report-out `"$Out`""

Write-Host "=== Generating S 118B freq report (N=$N) ==="
Write-Host "  cli : $cli"
Write-Host "  out : $Out"

cmd /c "cd /d `"$bin`" && llama-cli.exe $argsLine 1>`"$outLog`" 2>`"$errLog`" & echo EC=%ERRORLEVEL% > `"$env:TEMP\genfreq_ec.txt`""

$ec = ""
if (Test-Path "$env:TEMP\genfreq_ec.txt") {
    $ec = (Get-Content "$env:TEMP\genfreq_ec.txt" -Raw).Trim()
}
Write-Host "  exit=$ec"
if (Test-Path $Out) {
    Write-Host ("  report written: {0} bytes" -f (Get-Item $Out).Length)
} else {
    Write-Host "  REPORT MISSING" -ForegroundColor Red
}
