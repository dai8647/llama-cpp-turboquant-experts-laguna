[CmdletBinding()]
param(
    [string] $Model = "C:\Users\dai86\.lmstudio\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
    [int] $Runs = 3,
    [int] $Threads = 12,
    [int] $Seed = 42,
    [string] $Prompt = "Explain how sparse mixture-of-experts routing works. Give a concise technical example.",
    [string] $ResultsPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Runs -ne 3) {
    throw "The canonical benchmark requires exactly 3 runs."
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Cli = Join-Path $Root "build-stage1\bin\llama-cli.exe"
$Cache = Join-Path $Root "build-stage1\CMakeCache.txt"
$LogDir = Join-Path $Root "bench_qwen38_20tps_logs\$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path -LiteralPath $Cli -PathType Leaf)) {
    throw "Canonical benchmark binary is missing: $Cli (no fallback build is allowed)."
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Model file is missing: $Model"
}
if (-not (Test-Path -LiteralPath $Cache -PathType Leaf)) {
    throw "Canonical build cache is missing: $Cache"
}

$env:ROCM_PATH = "C:\Users\dai86\rocm-sdk-core\_rocm_sdk_core"
$env:HIP_PATH = $env:ROCM_PATH
$env:HIP_PLATFORM = "amd"
$env:HCC_AMDGPU_TARGET = "gfx1101"
$env:HIP_VISIBLE_DEVICES = "0"
$env:LLAMA_MOE_HOST_BANK = "0"
$env:LLAMA_MOE_SLOT_STATS = "1"
$env:PATH = "$($env:ROCM_PATH)\bin;$($env:ROCM_PATH)\lib\llvm\bin;$($env:PATH)"

$cacheText = Get-Content -LiteralPath $Cache -Raw
$requiredCache = @(
    "CMAKE_BUILD_TYPE:STRING=RelWithDebInfo",
    "GGML_HIP:BOOL=ON",
    "GGML_HIP_GRAPHS:BOOL=ON",
    "GGML_CUDA_FA:BOOL=ON"
)
$missingCache = @($requiredCache | Where-Object { $cacheText -notmatch [regex]::Escape($_) })
$gpuTargetMatch = $cacheText -match "(?m)^GPU_TARGETS:(?:STRING|UNINITIALIZED)=gfx1101\s*$"
if (-not $gpuTargetMatch) { $missingCache += "GPU_TARGETS=gfx1101" }
if ($missingCache.Count -gt 0) {
    throw "build-stage1 does not satisfy the canonical configuration: $($missingCache -join ', ')"
}

if ([string]::IsNullOrWhiteSpace($ResultsPath)) {
    $ResultsPath = Join-Path $Root "bench_qwen38_20tps_results_$((Get-Date).ToString('yyyyMMdd-HHmmss')).json"
}

function Get-LastMatch {
    param([string[]] $Lines, [string] $Pattern)
    $matches = @($Lines | Where-Object { $_ -match $Pattern })
    if ($matches.Count -eq 0) { return $null }
    return $matches[-1]
}

function Get-Number {
    param([string] $Line, [string] $Pattern)
    if ($null -eq $Line -or $Line -notmatch $Pattern) { return $null }
    return [double]$Matches[1]
}

function Get-SlotStats {
    param([string[]] $Lines)
    $line = Get-LastMatch $Lines "MoE GPU slot stats:"
    if ($null -eq $line) { return $null }
    $stats = [ordered]@{
        raw = $line
        copies = $null
        hit = $null
        miss = $null
        evict = $null
        copyMiB = $null
        avgMs = $null
        hitRate = $null
        bandwidthMiBPerSec = $null
    }
    foreach ($name in @("copies", "hit", "miss", "evict")) {
        if ($line -match "\b$name=([0-9]+)") { $stats[$name] = [int64]$Matches[1] }
    }
    $stats.copyMiB = Get-Number $line "\bcopy=([0-9]+(?:\.[0-9]+)?)\s*MiB"
    $stats.avgMs = Get-Number $line "\bavg=([0-9]+(?:\.[0-9]+)?)\s*ms"
    if ($null -ne $stats.hit -and $null -ne $stats.miss -and ($stats.hit + $stats.miss) -gt 0) {
        $stats.hitRate = [math]::Round($stats.hit / ($stats.hit + $stats.miss), 4)
    }
    if ($null -ne $stats.copyMiB -and $null -ne $stats.avgMs -and $stats.avgMs -gt 0) {
        $stats.bandwidthMiBPerSec = [math]::Round($stats.copyMiB / ($stats.avgMs / 1000.0), 2)
    }
    return [pscustomobject]$stats
}

function Invoke-BenchmarkRun {
    param([int] $RunNumber)
    $logPath = Join-Path $LogDir ("run-{0}.log" -f $RunNumber)
    $args = @(
        "-m", $Model,
        "--moe-hot-expert",
        "-ngl", "99",
        "-fa", "on",
        "-c", "8192",
        "-n", "256",
        "-t", "$Threads",
        "-s", "$Seed",
        "--reasoning", "off",
        "--no-display-prompt",
        "-p", $Prompt
    )

    Write-Host ("=== qwen4exp run {0}/{1} ===" -f $RunNumber, $Runs) -ForegroundColor Cyan
    $output = @(& $Cli @args 2>&1 | ForEach-Object { [string]$_ | Tee-Object -FilePath $logPath -Append })
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "llama-cli failed on run $RunNumber with exit code $exitCode. See $logPath"
    }

    $generationLine = Get-LastMatch $output "Generation:\s+([0-9]+(?:\.[0-9]+)?)\s+t/s"
    if ($null -eq $generationLine) {
        throw "Run $RunNumber has no Generation t/s line. See $logPath"
    }
    $genTs = Get-Number $generationLine "Generation:\s+([0-9]+(?:\.[0-9]+)?)\s+t/s"
    $promptLine = Get-LastMatch $output "Prompt:\s+([0-9]+(?:\.[0-9]+)?)\s+t/s"
    $promptTs = Get-Number $promptLine "Prompt:\s+([0-9]+(?:\.[0-9]+)?)\s+t/s"
    $slotStats = Get-SlotStats $output
    $compute = @($output | Where-Object { $_ -match "compute_buffer: device=.*buft=" })
    $faLines = @($output | Where-Object { $_ -match "(?i)(flash_attn\s*=|flash attention|fattn)" })

    [pscustomobject]@{
        run = $RunNumber
        generationTs = $genTs
        promptTs = $promptTs
        slotStats = $slotStats
        computeBuffer = $compute
        faLines = $faLines
        log = $logPath
    }
}

Write-Host "Canonical binary: $Cli"
Write-Host "Model: $Model"
Write-Host "Conditions: ctx=8192 n_predict=256 runs=3 median gen t/s fa=on HOST_BANK=0"

$runResults = @()
for ($run = 1; $run -le $Runs; $run++) {
    $runResults += Invoke-BenchmarkRun $run
    if ($run -lt $Runs) { Start-Sleep -Seconds 2 }
}

$genValues = @($runResults | ForEach-Object { [double]$_.generationTs } | Sort-Object)
$median = $genValues[1]
$allCompute = @($runResults | ForEach-Object { $_.computeBuffer } | Where-Object { $_ })
$allFaLines = @($runResults | ForEach-Object { $_.faLines } | Where-Object { $_ })
$flashEnabled = @($allFaLines | Where-Object { $_ -match "flash_attn\s*=\s*(enabled|on)" }).Count -gt 0
$gpuCompute = @($allCompute | Where-Object { $_ -match "(?i)(ROCm|HIP|CUDA|Vulkan)" }).Count -gt 0
$fattnEvidence = @($allFaLines | Where-Object { $_ -match "(?i)(fattn|flash attention)" }).Count -gt 0
$faStatus = if ($flashEnabled -and $gpuCompute -and ($cacheText -match "GGML_CUDA_FA:BOOL=ON")) { "PASS" } else { "FAIL" }
if (-not $fattnEvidence) { $faStatus = "UNKNOWN" }

$summary = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    binary = $Cli
    model = $Model
    conditions = [ordered]@{ context = 8192; nPredict = 256; runs = $Runs; threads = $Threads; flashAttention = "on"; hostBank = 0; slotStats = 1 }
    build = [ordered]@{ cache = $Cache; type = "RelWithDebInfo"; hip = $true; hipGraphs = $true; flashAttentionCompile = $true; gpuTarget = "gfx1101" }
    runs = $runResults
    generationTsMedian = $median
    faStatus = $faStatus
    faChecks = [ordered]@{ flashAttnEnabledLog = $flashEnabled; gpuComputeBuffer = $gpuCompute; fattnRuntimeMarker = $fattnEvidence; compileOption = $true }
    logs = $runResults | ForEach-Object { $_.log }
}
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ResultsPath -Encoding UTF8

Write-Host ""
Write-Host "=== MEDIAN (canonical gen t/s) ===" -ForegroundColor Green
Write-Host ("gen: {0} t/s" -f $median)
Write-Host ("FA_STATUS: {0}" -f $faStatus)
foreach ($result in $runResults) {
    $stats = if ($null -eq $result.slotStats) { "slot_stats=unknown" } else { "hit_rate=$($result.slotStats.hitRate) bandwidth=$($result.slotStats.bandwidthMiBPerSec) MiB/s" }
    Write-Host ("run {0}: gen={1} t/s {2}" -f $result.run, $result.generationTs, $stats)
}
Write-Host "Results: $ResultsPath"
if ($faStatus -ne "PASS") {
    Write-Warning "FA verification did not produce a clean PASS; do not treat this run as FA-validated evidence."
}
