# Session A: 78 MiB/s transfer split

## Handoff status

The reported `bw=78.3 MiB/s` is the existing cumulative materialization rate:

```text
copy_bytes / copy_ns
```

`copy_ns` starts after bank setup and any prior prefill-event drain. It includes the pinned-stage source preparation, H2D enqueue, copy-stream event wait, and event destruction. It is not a PCIe-only bandwidth measurement.

With `stage=100%`, `stage_nonhost=0`, and `pageable=0`, the current run used the pinned-stage branch and classified every source as host-backed. This does not mean the source is anonymous RAM: an mmap-backed GGUF tensor can be host-backed to the backend while its pages are file-backed and demand-paged.

## New split telemetry

The Session A instrumentation now emits a separate line when `LLAMA_MOE_SLOT_STATS=1`:

```text
MoE transfer split: stage_memcpy=... MiB stage_memcpy_ms=... stage_get=... MiB stage_get_ms=... h2d_wait=... MiB h2d_wait_ms=... h2d_bw=... MiB/s fallback_get_ms=... fallback_set_ms=... events=...
```

Definitions:

- `stage_memcpy`: host-backed source slice bytes copied into the shared pinned stage, and its elapsed CPU time.
- `stage_get`: non-host source bytes read with `ggml_backend_tensor_get` into the pinned stage, and its elapsed time.
- `h2d_wait`: H2D payload bytes associated with successful event waits, and the elapsed `event_synchronize` time.
- `h2d_bw`: `h2d_wait_bytes / h2d_wait_ns`; this is the closest current diagnostic to the completed H2D path, but still includes event timing and does not isolate PCIe hardware from driver behavior.
- `fallback_get_ms` / `fallback_set_ms`: synchronous fallback path timing. They are separate from successful pinned-stage H2D telemetry.

The existing `copy_bw` remains the end-to-end materialization rate and should be compared with `h2d_bw` rather than interpreted as the same quantity.

## Root-cause decision table

- Large `stage_memcpy_ms`, small `h2d_wait_ms`: host-side source preparation or mmap page faults dominate.
- Large `stage_get_ms`: non-host backend read path dominates.
- Small preparation time, large `h2d_wait_ms`, and low `h2d_bw`: HIP H2D/copy-stream behavior dominates.
- High `h2d_bw` but low `copy_bw`: staging, event setup, allocation, or other per-materialize overhead dominates.

## Lazy-mode note

`--lazy-mode` is not the model-loader residency switch. The model loader uses mmap by default; the relevant source can still be file-backed even when the backend reports it as host-backed. Therefore `--lazy-mode off` is not a valid explanation for the transfer split and should not be used as the primary Session A performance experiment.

## Measurement status

The earlier canonical attempt aborted before generation because of a transient GPU stream-creation failure. A later successful run reported the aggregate 78.3 MiB/s, 100% `stage`, 0 `stage_nonhost`, and 0 `pageable`, with a final hit rate around 71% (7708 hits / 3072 misses). Those values are useful diagnostics, but the new split fields must be collected in a fresh build-stage1 run before assigning the bottleneck to memcpy, page faults, or H2D.

Canonical rules remain unchanged: use only `build-stage1`, `LLAMA_MOE_HOST_BANK=0`, ctx 8192, `n_predict=256`, three-run median, and judge by generation t/s.
