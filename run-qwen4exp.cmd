@echo off
rem Qwen3.8-Flash-Next (qwen4exp) on RX 7800 XT - FreeToken hot-expert
rem Usage: run-qwen4exp.cmd [extra llama-cli args...]

setlocal
set ROCM_PATH=C:\Users\dai86\rocm-sdk-core\_rocm_sdk_core
set HIP_PATH=%ROCM_PATH%
set HIP_PLATFORM=amd
set HCC_AMDGPU_TARGET=gfx1101
set HIP_VISIBLE_DEVICES=0
set PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\llvm\bin;%PATH%

set MODEL=C:\Users\dai86\.lmstudio\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
set CLI=C:\Users\dai86\llama-cpp-turboquant-experts-laguna\build-stage1\bin\llama-cli.exe
if not exist "%CLI%" set CLI=C:\Users\dai86\llama-cpp-turboquant-experts-laguna\build-hip\bin\llama-cli.exe

rem 8k ctx default - raise with -c if RAM allows (n-gram shard stays mmap)
"%CLI%" -m "%MODEL%" --moe-hot-expert -ngl 99 -c 8192 -t 16 %*
endlocal
