# Session A follow-up: measurement and event wait analysis

Date: 2026-09-19

## Measurement attempt

The canonical command was started with:

```powershell
powershell -ExecutionPolicy Bypass -File .\bench_qwen38_20tps.ps1
```

Conditions were the required Session 3 rules:

- `build-stage1\bin\llama-cli.exe`
- `LLAMA_MOE_HOST_BANK=0`
- `LLAMA_MOE_SLOT_STATS=1`
- context 8192
- `n_predict 256`
- `-fa on`
- three runs intended, median of `Generation t/s`

The stage1 binary and server were present and reported version 12 / `3f39c191f`. The disabled model files were present as `src/models/bailingmoe3.cpp.disabled` and `src/models/kimi-k3.cpp.disabled`. No MTP build or `src/models/qwen4exp.cpp` file was touched.

The run loaded the model and reached the interactive prompt, but aborted at the first generation with the generic HIP/ROCm abort from `ggml_cuda_error` (`ggml-cuda.cu:108`). It produced no `Generation`, `MoE GPU slot stats`, `compute_buffer`, or transfer-path result. Therefore this attempt is not a valid throughput measurement and must not be used as a baseline.

Artifacts:

- Wrapper log: `bench_sessionA_run_20260919.log`
- CPU sample log: `bench_sessionA_cpu_20260919.csv`
- Detailed run log: `bench_qwen38_20tps_logs\20260919-164005\run-1.log`

The CPU samples cover model loading, not decode. The process working set grew to roughly 70 GB during loading and the process CPU time was high. Since generation never began, no decode CPU percentage can be reported. The success gate (`bw >= 10 GB/s` and decode CPU <= 70%) is therefore **not evaluated**.

## Event synchronization analysis

The decode miss path in `src/llama.cpp` does this:

1. Obtain the backend extension copy stream.
2. Issue one or more `ggml_backend_cuda_ext_h2d_async()` calls on that dedicated stream.
3. Record an event on the same copy stream.
4. Call `ggml_backend_cuda_ext_event_synchronize()`.
5. Mark the slot resident and return the slot id to the remap operation.

The extension implementation in `ggml/src/ggml-cuda/ggml-cuda.cu` confirms that the extension stream is separate from the backend compute stream. `event_synchronize()` calls `cudaEventSynchronize`/the HIP equivalent on the event; it does not call `cudaDeviceSynchronize` and does not directly block the compute stream.

The important critical-path effect is on the host: expert-id remap is a CPU custom operation that runs before the graph's GPU work is launched. The remap callback holds `cache_mutex` while materializing and waiting. Thus the wait blocks the remap CPU thread and delays submission of the following compute graph. It is correct to describe this as a compute critical-path stall, but not as a direct synchronization of all work on the compute stream.

After remap returns, the compute stream has no event-wait operation for this slot. Safety currently comes from the host-side wait completing the copy before `resident=true` is returned. Removing the wait while returning the slot id would allow the compute graph to read a bank region whose copy stream is still writing it.

## Safe deferred-wait design

A safe implementation requires all of the following:

1. Keep an event handle associated with every assigned slot until its H2D completes.
2. Keep the slot non-resident or otherwise unavailable to eviction until completion.
3. Insert a wait for that event on the actual compute stream before the first kernel that reads the slot.
4. Retire/destroy the event only after the compute-stream wait has been enqueued and the slot cannot be overwritten.
5. Keep the current fallback for backends without the required stream/event API.

The current extension exposes a copy stream and event record/query/synchronize, but it does not expose a `compute_stream_wait_event` operation to `llama.cpp`. A host-side delayed wait alone is not safe. Adding that API and integrating it with slot eviction is a larger cross-layer change and overlaps Session B's early-fire/prefetch work. No event wait was removed in this follow-up.

The existing prefill path follows the same safety rule: it records an event, keeps `resident=false`, and the remap path synchronizes a matching in-flight event before using the slot.

## Theoretical throughput estimate

No valid `copy_ns`, `copy_bytes`, or generation timing was produced in this attempt, so a numeric improvement estimate would be fabricated. Once a valid run exists, calculate:

- `W = wait_time / total_generation_time`
- ideal no-wait speedup: `1 / (1 - W)`
- ideal no-wait generation rate: `measured_gen_ts / (1 - W)`

This is an upper bound only. It assumes all removed host wait overlaps useful GPU compute and ignores event bookkeeping, routing, eviction, and bandwidth contention. The transfer-path `bw` value is an effective end-to-end materialization rate, not a PCIe-only measurement.

## Next measurement prerequisite

Before retrying the canonical three-run benchmark, capture the full HIP statement/error text for the generation abort (the current wrapper only retained the generic `ggml-cuda.cu:108` line). Without a valid generation run and slot stats, Session A cannot be declared successful or unsuccessful based on bandwidth.
