param(
    [string]$Step = "pass1",
    [string]$ExtraArgsCsv = ""
)
$ErrorActionPreference = "Continue"
$cli = "C:\Users\dai86\llama-cpp-turboquant\build-hip\bin\llama-cli.exe"
$model = "C:\Users\dai86\.lmstudio\models\unsloth\Laguna-S-2.1-GGUF\Laguna-S-2.1-UD-IQ3_XXS.gguf"
$promptFile = "C:\Users\dai86\llama-cpp-turboquant\benchmarks\laguna_prompt.txt"
$rocmbin = "C:\Program Files\AMD\ROCm\7.1\bin"
$env:PATH = "$rocmbin;" + $env:PATH
$env:ROCM_PATH = "C:\Program Files\AMD\ROCm\7.1"

$baseArgs = @("-m", $model, "-ngl", "999", "-n", "128", "-f", $promptFile, "--temp", "0", "-s", "42")
$extraArgs = @()
if ($ExtraArgsCsv -ne "") { $extraArgs = $ExtraArgsCsv -split "," }
$allArgs = $baseArgs + $extraArgs
$log = "$env:TEMP\laguna_s_$Step.log"

Write-Host "=== $Step : $($allArgs -join ' ') ==="
$proc = Start-Process -FilePath $cli -ArgumentList $allArgs -RedirectStandardError $log -RedirectStandardOutput "$log.out" -PassThru -Wait
Write-Host "EXIT: $($proc.ExitCode)"

$gen_ts = $null
$prompt_ts = $null
foreach ($line in (Get-Content $log)) {
    if ($line -match "Prompt:\s+([\d.]+)") { $prompt_ts = $Matches[1] }
    if ($line -match "Generation:\s+([\d.]+)") { $gen_ts = $Matches[1] }
}
Write-Host "Prompt: $prompt_ts t/s | Generation: $gen_ts t/s"
Write-Host "LOG: $log / $log.out"
