@echo off
rem Handoff C: online frequency pin ONLY (HOST_BANK forced 0)
rem Usage: run-qwen4exp-bank.cmd [extra llama-cli args...]
rem HOST_BANK=1 is FORBIDDEN: load-time full-layer pin causes CPU~95% / Disk~100% / ROCm error
rem Measurement is owned by sessions D / F. C does not launch or bench.

setlocal
set ROCM_PATH=C:\Users\dai86\rocm-sdk-core\_rocm_sdk_core
set HIP_PATH=%ROCM_PATH%
set HIP_PLATFORM=amd
set HCC_AMDGPU_TARGET=gfx1101
set HIP_VISIBLE_DEVICES=0
set PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\llvm\bin;%PATH%

set LLAMA_MOE_HOST_BANK=0
set LLAMA_MOE_SLOT_STATS=1
rem online pin default is already on with --moe-hot-expert (N=2000)
rem optional override: set LLAMA_MOE_AUTO_PIN_AFTER=2000

set MODEL=C:\Users\dai86\.lmstudio\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
set CLI=C:\Users\dai86\llama-cpp-turboquant-experts-laguna\build-bank\bin\llama-cli.exe

"%CLI%" -m "%MODEL%" --moe-hot-expert -ngl 99 -c 8192 -t 16 --reasoning off %*
endlocal
