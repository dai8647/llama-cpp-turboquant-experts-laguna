# S 118B dedicated benchmark (Laguna-S-2.1-UD-IQ3_XXS only)
# Pass criteria: Generation >= 20 tok/s at n_predict=256, median of 3 runs.
# Prompt t/s is reported but not used for pass/fail.
#
# Usage:
#   powershell -File benchmarks\bench_s118b.ps1 -FlashAttn on
#   powershell -File benchmarks\bench_s118b.ps1 -FlashAttn off
#   powershell -File benchmarks\bench_s118b.ps1 -FlashAttn on -Runs 3 -N 256 -Ctx 4096

param(
    [ValidateSet("on", "off")]
    [string]$FlashAttn = "on",
    [int]$Runs = 3,
    [int]$N = 256,
    [int]$Ctx = 4096,
    [int]$Ub = 256,
    [int]$Seed = 42
)

$ErrorActionPreference = "Continue"

# S 118B model only. XS reports/models are not valid inputs.
$model = "C:\Users\dai86\.lmstudio\models\unsloth\Laguna-S-2.1-GGUF\Laguna-S-2.1-UD-IQ3_XXS.gguf"
$promptFile = "C:\Users\dai86\llama-cpp-turboquant\benchmarks\laguna_prompt.txt"

# Release gfx1101 build only. Debug build-hip binaries must not be used for bench.
$cli = "C:\Users\dai86\llama-cpp-turboquant\build-hip-rel\bin\llama-cli.exe"
if (-not (Test-Path $cli)) {
    Write-Host "ERROR: Release build not found at $cli" -ForegroundColor Red
    exit 2
}
if ($cli -match "build-hip\\") {
    Write-Host "ERROR: refusing to use Debug build-hip binary for benchmarking" -ForegroundColor Red
    exit 2
}

$rocmbin = "C:\Program Files\AMD\ROCm\7.1\bin"
$env:PATH = "$rocmbin;" + $env:PATH
$env:ROCM_PATH = "C:\Program Files\AMD\ROCm\7.1"

$target = 20.0
$faArg = if ($FlashAttn -eq "on") { @("-fa", "on") } else { @("-fa", "off") }
# -v raises CLI verbosity above the default ERROR so INFO logs (flash_attn, FA dispatch) are emitted to stderr.
$baseArgs = @("-m", $model, "-ngl", "999", "-c", "$Ctx", "-n", "$N", "-f", $promptFile,
              "--temp", "0", "-s", "$Seed", "--simple-io", "-ub", "$Ub", "-v") + $faArg

$fallbackMarkers = @(
    "Flash Attention was auto, set to disabled",
    "flash attention is not supported",
    "FlashAttention is not supported",
    "not supported",
    "failed to enable Flash Attention"
)

Write-Host "=== S 118B Benchmark ===" -ForegroundColor Green
Write-Host "Model : $model"
Write-Host "CLI   : $cli"
Write-Host "FlashAttn : $FlashAttn | n_predict: $N | ctx: $Ctx | ubatch: $Ub | seed: $Seed"
Write-Host "Runs  : $Runs | Pass: Generation >= $target tok/s"
Write-Host ""

function Invoke-Run {
    param($i)
    $outLog = "$env:TEMP\s118b_run${i}_out.log"
    $errLog = "$env:TEMP\s118b_run${i}_err.log"
    Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue

    # Use Start-Process with an argument array. cmd /c after a full model load
    # does not return control to the caller in this environment.
    $proc = Start-Process -FilePath $cli -ArgumentList $baseArgs `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
        -Wait -PassThru
    # Start-Process spuriously reports 130 for successful console runs here.
    # Real failures (missing DLL, load crash) show a different code.
    $exitCode = $proc.ExitCode

    $gen = $null
    $prompt = $null
    Get-Content $outLog -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_ -match "Prompt:\s+([\d.]+)\s+t/s") { $prompt = [double]$Matches[1] }
        if ($_ -match "Generation:\s+([\d.]+)\s+t/s") { $gen = [double]$Matches[1] }
    }

    $err = Get-Content $errLog -Raw -ErrorAction SilentlyContinue

    if ($exitCode -ne 0 -and $exitCode -ne 130) {
        Write-Host "  RUN $i FAILED exit=$exitCode" -ForegroundColor Red
        return @{ ok = $false; reason = "exit $exitCode" }
    }
    if ($err -match "Segmentation fault|segmentation fault|AddressSanitizer") {
        Write-Host "  RUN $i CRASH detected" -ForegroundColor Red
        return @{ ok = $false; reason = "crash" }
    }
    if ($gen -eq $null) {
        Write-Host "  RUN $i NO TIMING (exit=$exitCode)" -ForegroundColor Red
        return @{ ok = $false; reason = "no generation timing" }
    }

    if ($FlashAttn -eq "on") {
        $enabled = $err -match "flash_attn\s+=\s+enabled"
        $fellback = $false
        foreach ($m in $fallbackMarkers) {
            if ($err -match [regex]::Escape($m)) { $fellback = $true; break }
        }
        if (-not $enabled -or $fellback) {
            Write-Host "  RUN $i FA_DISPATCH_FAIL en=$enabled fallback=$fellback" -ForegroundColor Red
            return @{ ok = $false; reason = "flash attention silent fallback" }
        }
    }

    Write-Host ("  RUN {0}  Prompt: {1,6} t/s | Generation: {2,6} t/s" -f $i, $prompt, $gen)
    return @{ ok = $true; gen = $gen; prompt = $prompt }
}

$gens = @()
$allOk = $true
for ($i = 1; $i -le $Runs; $i++) {
    $r = Invoke-Run $i
    if (-not $r.ok) {
        $allOk = $false
        Write-Host "  reason: $($r.reason)"
    } else {
        $gens += $r.gen
    }
}

Write-Host ""
if ($gens.Count -eq 0) {
    Write-Host "NO VALID RUNS" -ForegroundColor Red
    exit 1
}

$sorted = @($gens | Sort-Object)
$medIdx = [math]::Floor($sorted.Count / 2)
$median = $sorted[$medIdx]
$pass = $allOk -and ($median -ge $target)

Write-Host "=== RESULT ===" -ForegroundColor Green
Write-Host ("Generation median: {0} tok/s (runs: {1})" -f $median, ($gens -join ", "))
if ($pass) {
    Write-Host "PASS (>= $target tok/s)" -ForegroundColor Green
} else {
    Write-Host "FAIL (< $target tok/s)" -ForegroundColor Red
    Write-Host "Bottleneck decomposition requires: FA dispatch / expert transfer / VRAM capacity / RAM bandwidth / context."
}

$resultFile = "C:\Users\dai86\llama-cpp-turboquant\benchmarks\s118b_last_result.json"
@{ flash_attn = $FlashAttn; runs = $gens; median = $median; target = $target; pass = $pass; ok = $allOk } |
    ConvertTo-Json | Set-Content $resultFile
exit $(if ($pass) { 0 } else { 1 })
