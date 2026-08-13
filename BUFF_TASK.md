# TASK: Fix frequency placement (hot-expert GPU slot) deadlock on HIP

## Context
Private fork `llama-cpp-turboquant-experts-laguna` (AMD HIP/ROCm). DeepSeek V4 Flash 150B.

## Problem
Pass 1 (access tracking for frequency placement) and Pass 2 (eval-time remap for frequency whitelist) both use `build_moe_gpu_slot_ids` in `src/llama-graph.cpp`, which creates:
- `ggml_cpy(ctx0, logical_experts, ggml_new_tensor_2d(...))` (contiguous_experts)
- `ggml_map_custom1(...)` with `llm_moe_gpu_slot_remap` callback (record_access + remap expert id -> slot id)

`map_custom1` is a CPU-only op (only `ggml-cpu` implements it). When the graph runs on HIP (GPU), the scheduler tries to handle the cross-backend dependency (GPU argsort -> CPU cpy -> CPU map_custom1 -> GPU mul_mat_id), but it deadlocks at device level with:

```
decode() failed: resource deadlock would occur
```

The deadlock also manifests with `--cpu-moe` (all experts on CPU): server crashes (connection reset / SIGSEGV-like) at the first decode step.

The user already confirmed the feature is needed (`hot expert` placement) but blocked by this.

## What to fix
The `contiguous_experts` tensor needs to be explicitly placed on a CPU backend (not GPU) before being fed into `map_custom1`. Or the remap mechanism needs to run fully on host before/after the graph, rather than as a device op embedded in the GPU graph pipeline.

Key references in source:
- `src/llama-graph.cpp`: `build_moe_gpu_slot_ids` (line ~1621), `llm_moe_gpu_slot_remap` (line ~266), `llm_graph_input_moe_gpu_slot_map` (line 233)
- `src/llama.cpp`: `save_moe_freq_report_if_requested`, frequency mode activation (`requested_slots` logic), `llama_moe_gpu_expert_cache` init
- `src/llama-model.h`: `ensure_resident`, `find`, `record_access`, `frequency_whitelist`

The commit `b56a5b2cb` added `recursive_mutex` guard for the cache; `6b526435f` added the `ggml_cpy` materialization. The deadlock is likely because the cpy dst is allocated on GPU by the scheduler (since its consumer `map_custom1` is CPU, but the scheduler may place dst on the GPU backend that handles the source of cpy), and `map_custom1` then tries to read GPU memory as a host pointer.

## Deliverable
1. Reproduce the deadlock locally (already done: `--cpu-moe` + `--moe-gpu-expert-slot-num 30` -> deadlock/crash)
2. Fix: either force `contiguous_experts` buffer onto CPU explicitly, or redesign the remap to avoid `map_custom1` in the GPU graph path.
3. Verify: Pass 1 (`--moe-freq-report-out stats.json` with `--moe-gpu-expert-slot-num 30` + `-t 6 -c 8192`) writes a non-empty `freq_dsv4.json` containing `n_layers: 43`, `n_experts: 132`, and sorted expert counts.
4. Verify Pass 2 (`--moe-expert-placement frequency --moe-freq-report-path stats.json --moe-gpu-expert-ratio 0.18`) does NOT deadlock and produces faster generation than baseline.
5. Build command (Windows): `cmake --build build-hip --config RelWithDebInfo -j` (rebuild time: ~10-30 min).
6. Do NOT change commit messages or PR descriptions. Only fix the code.

## Environment notes for BUFF
- Machine: RX 7800 XT (gfx1101), 16 GB VRAM
- OS: Windows (PowerShell 5.1), ROCm 7.1 at `C:\Program Files\AMD\ROCm\7.1`
- Required env vars: `$env:ROCM_PATH='C:\Program Files\AMD\ROCm\7.1'; $env:HCC_AMDGPU_TARGET='gfx1101'; $env:HIP_VISIBLE_DEVICES='0'`
- Build: `cmake --build build-hip --config RelWithDebInfo -j` (HIP target must include gfx1101)
