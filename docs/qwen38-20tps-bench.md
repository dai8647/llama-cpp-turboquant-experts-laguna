# Qwen3.8-Flash-Next 20 t/s benchmark

This is the canonical Session C measurement procedure.

## Fixed rules

- Binary: `build-stage1\bin\llama-cli.exe` only.
- Build: RelWithDebInfo, HIP, `gfx1101`, `GGML_HIP_GRAPHS=ON`, `GGML_CUDA_FA=ON`.
- Model: Qwen3.8-Flash-Next GSQ-RCO Q2_0 GGUF.
- Runtime: `--moe-hot-expert`, `-fa on`, context 8192, `n_predict 256`.
- Environment: `LLAMA_MOE_HOST_BANK=0` and `LLAMA_MOE_SLOT_STATS=1`.
- Result: median of exactly three runs, judged by end-to-end `Generation: ... t/s` only.
- Do not report numbers from `build`, `build-hip`, `build-bank`, or any other build.
- Never use `LLAMA_MOE_HOST_BANK=1`; it pins the complete expert bank and is a known CPU/disk/ROCm failure mode.

## Run the benchmark

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\bench_qwen38_20tps.ps1
```

The script fails instead of falling back when `build-stage1` or the model is missing. It writes one combined log per run under `bench_qwen38_20tps_logs\<timestamp>\` and a timestamped JSON summary under the repository root.

Optional parameters are available for a local model path or prompt, but the canonical run remains three repetitions:

```powershell
powershell -ExecutionPolicy Bypass -File .\bench_qwen38_20tps.ps1 `
  -Model 'D:\models\Qwen3.8-Flash-Next-Q2.gguf' `
  -ResultsPath .\qwen38-result.json
```

## Required log checks

Every run must retain:

- `flash_attn = enabled` (the `-fa on` request was accepted).
- `compute_buffer: device=... buft=...` with a ROCm/HIP GPU device.
- MoE slot initialization and pinned-stage lines.
- `MoE GPU slot stats: copies=... hit=... miss=... evict=... copy=... MiB avg=... ms`.

The benchmark reports slot hit rate as `hit/(hit+miss)` and an approximate transfer rate as `copy MiB / avg ms`. These are diagnostic values; they do not replace generation t/s as the acceptance metric.

`FA_STATUS=PASS` requires the enabled flash-attention log, a GPU compute buffer, and the build cache's `GGML_CUDA_FA=ON`. If no runtime `fattn`/flash-attention marker is present, the status is `UNKNOWN`; neither `UNKNOWN` nor `FAIL` is valid FA-validated evidence. A silent fallback is a failure.

## FA and Qwen head shape

HIP reuses the CUDA FA sources in `ggml/src/ggml-hip/CMakeLists.txt`. The HIP list includes the tile and MMA kernels and the required vector instances for f16, q8, bf16, and TurboQuant cache combinations. Do not add a second HIP FA implementation.

The runtime graph creates `ggml_flash_attn_ext` when flash attention is enabled and there is no KQ bias. The compiled FA dispatcher has specializations for the supported Q/K/V head dimensions, including 64, 80, 96, 112, 128, 192/128, 256/256, 320/256, and the larger MLA forms. Confirm the actual Qwen head dimensions in the model-load metadata/log before claiming a shape-specific validation. The benchmark's runtime log and build cache are the evidence to attach to a result.

Before a canonical run, the build cache can be checked with:

```powershell
Select-String -Path .\build-stage1\CMakeCache.txt `
  -Pattern 'CMAKE_BUILD_TYPE:STRING=RelWithDebInfo','GGML_HIP:BOOL=ON',
           'GGML_HIP_GRAPHS:BOOL=ON','GGML_CUDA_FA:BOOL=ON','GPU_TARGETS:(STRING|UNINITIALIZED)=gfx1101'
```

## Minimal llama-server startup for Session D

This is a startup smoke procedure, not a replacement for the CLI benchmark. Use the stage1 server and preserve stderr:

```powershell
$env:ROCM_PATH = 'C:\Users\dai86\rocm-sdk-core\_rocm_sdk_core'
$env:HIP_PATH = $env:ROCM_PATH
$env:HIP_PLATFORM = 'amd'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:LLAMA_MOE_HOST_BANK = '0'
$env:LLAMA_MOE_SLOT_STATS = '1'
$env:PATH = "$env:ROCM_PATH\bin;$env:ROCM_PATH\lib\llvm\bin;$env:PATH"

$server = '.\build-stage1\bin\llama-server.exe'
$model = 'C:\Users\dai86\.lmstudio\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'
& $server -m $model --moe-hot-expert -ngl 99 -fa on -c 8192 -t 12 `
  --host 127.0.0.1 --port 8080 2> .\qwen38-server.stderr.log
```

In another shell, wait for `http://127.0.0.1:8080/health`, then inspect `qwen38-server.stderr.log` for `compute_buffer`, `flash_attn = enabled`, pinned-stage, and slot-stat lines. Keep `HOST_BANK=0` for this procedure as well.
