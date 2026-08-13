param(
  [string]$Report = 'freq_dsv4.json',
  [int]$N = 500,
  [string]$ExtraArgs = ''
)
$ErrorActionPreference = 'Stop'
$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH

$root = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna'
$model = 'C:\Users\dai86\.lmstudio\models\puwaer\DeepSeek-V4-Flash-0731-reap-150b-gguf\DeepSeek-V4-Flash-0731-reap-150b-Q3_K_M.gguf'
Set-Location $root

$prompt = '与えられた課題に対して、論理的で実用的な回答をしてください。まず全体方針を述べ、次に具体的な手順、最後に注意点を簡潔にまとめてください。'
$base = @('-m', $model, '--moe-freq-report-path', $Report, '-p', $prompt, '-n', "$N", '-c', '8192', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '-t', '6', '--no-display-prompt')
$extra = if ($ExtraArgs) { ($ExtraArgs -split ' ') } else { @() }
$all = $base + $extra

$out = "$root\pass1_out.log"; $err = "$root\pass1_err.log"
Remove-Item $out, $err -ErrorAction SilentlyContinue

$proc = Start-Process -FilePath "$root\build-hip\bin\llama-cli.exe" -ArgumentList $all -WorkingDirectory $root -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
$proc.WaitForExit()

$gen = ''
if (Test-Path $err) {
    $gen = (Get-Content $err | Select-String -Pattern 'timings:|tokens per second|Generation' | Select-Object -Last 3 | ForEach-Object { $_.Line.Trim() }) -join ' | '
}
[ordered]@{ run = 'pass1-freq'; report = $Report; n = $N; exit = $proc.ExitCode; timings = $gen } | ConvertTo-Json | Set-Content -Path "$root\pass1_last.json" -Encoding UTF8
Write-Output "exit=$($proc.ExitCode)"
Write-Output $gen
if (Test-Path "$root\$Report") {
    $sz = (Get-Item "$root\$Report").Length
    Write-Output "report written: $root\$Report ($sz bytes)"
} else {
    Write-Output "REPORT NOT FOUND"
}