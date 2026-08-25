@echo off
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.1
set HCC_AMDGPU_TARGET=gfx1101
set HIP_VISIBLE_DEVICES=0
set LLAMA_MOE_SLOT_STATS=1
cd /d C:\Users\dai86\llama-cpp-turboquant-experts-laguna
build-hip\bin\llama-server.exe -m "C:\Users\dai86\.lmstudio\models\gbuzhf\Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF\Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Mini-v2D-lite.gguf" --host 127.0.0.1 --port 8093 --no-webui -lv 4 -c 4096 -np 1 -t 6 --cpu-moe -fa on -ctk q8_0 -ctv q8_0 --moe-gpu-expert-slot-num 30 --no-warmup > diag_out.log 2> diag_err.log