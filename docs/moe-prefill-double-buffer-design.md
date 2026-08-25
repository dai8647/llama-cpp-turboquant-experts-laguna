# Prefill double buffering + HIP graph capture fix + elastic VRAM - design note

Status: Phase 2 implementation design. Companion to `moe-slot-cache-async-design.md`
(Phase 1a/1b). Branch: `feat/prefill-double-buffer`.

Scope (three items):

1. Blocker: argsort abort during HIP graph capture ("operation not permitted
   when stream is capturing") on prefills of roughly 3000+ tokens.
2. Prefill double buffering: hide miss H2D behind GPU compute (Phase 2 of the
   Phase-1 note).
3. Elastic VRAM sizing: `--moe-gpu-expert-slot-num auto`.

## 1. Current implementation map

All references are to this repo at commit 62f8d92c7.

- argsort/top_k CUDA ops: `ggml/src/ggml-cuda/argsort.cu`, `top-k.cu`. On HIP
  (`GGML_CUDA_USE_HIPCUB`) argsort ALWAYS takes the hipCUB
  `DeviceSegmentedRadixSort` path (llama.cpp #24177: AMD shared memory is too
  small for bitonic at large ncols); top_k falls back to the same primitive.
- Graph capture: `ggml_backend_cuda_graph_compute`
  (ggml/src/ggml-cuda/ggml-cuda.cu:4365). Warmup requires two evaluations with
  identical uid/properties; capture starts at `cudaStreamBeginCapture`
  (ggml-cuda.cu:4416).
- Pool: `ggml_cuda_pool_leg` (ggml-cuda.cu:461), per-device/per-stream,
  best-fit reuse of up to 256 cached buffers. A miss calls
  `ggml_cuda_device_malloc` (= hipMalloc), and the OOM retry path additionally
  calls `cudaDeviceSynchronize`.
- Miss path today: remap custom op (`llm_moe_gpu_slot_remap`,
  src/llama-graph.cpp:266) -> `ensure_resident` -> `materialize_cb` ->
  blocking `ggml_backend_tensor_set` (sync H2D) while the whole ubatch waits.
- Decode prefetch: `llama_moe_gpu_expert_slot_prefetch` (src/llama.cpp:841),
  hooked at src/llama-context.cpp:2030 behind `n_tokens_all == 1`; prefill is
  explicitly out of scope there.
- Slot resolution: `--moe-gpu-expert-slot-num` parsed at common/arg.cpp:2705,
  consumed at src/llama.cpp:1101 during model load (`cache.init(effective)`
  + preload). Expert tensors are forced off the GPU at load time whenever the
  param is >= 0 (src/llama-model.cpp:1724).

## 2. Findings

### F1: the capture abort is a pool-miss hipMalloc inside the captured region

The reported error surfaces in `argsort_f32_i32_cuda_hipcub`. Two candidate
causes were named in the outsourcing brief: (a) pool miss -> hipMalloc during
capture, (b) hipCUB internals doing non-capturable work.

Code-level analysis strongly indicts (a):

- Every workspace in the hipcub paths comes from `ggml_cuda_pool_alloc`
  (temp keys, temp indices, offsets, rocPRIM temp storage). All four are
  alive simultaneously inside one sort call.
- On a pool miss, `ggml_cuda_pool_leg::alloc` calls hipMalloc mid-capture;
  HIP rejects it with `hipErrorStreamCaptureUnsupported` ("operation not
  permitted when stream is capturing"). The OOM retry path would additionally
  hit `cudaDeviceSynchronize`, which is equally illegal during capture.
- Workspace sizes scale linearly with `nrows` (= tokens in the ubatch) for
  router argsort `[n_expert, n_tokens]` and with vocab-size ncols for sampler
  top_k. Decode shapes repeat forever, so their pools stay warm; long prefills
  create NEW larger shapes whose workspaces exceed every cached buffer, and
  the first captured evaluation of such a shape deterministically misses the
  pool -> abort. That matches "crashes at 3130+ token prompts, decode is fine".
- The null-storage sizing query (`SortPairs(nullptr, &bytes, ...)`) is pure
  host arithmetic in both cub and rocPRIM, so it cannot be the abort source;
  nothing else in the sort path allocates or synchronizes.

Fix strategy: guarantee the pool holds every workspace the graph will request
BEFORE `cudaStreamBeginCapture` runs (per-brief option "ワークスペースの
キャプチャ前確保"). This removes the hipMalloc class of failures entirely and
keeps argsort inside captured graphs (no perf give-up, unlike
GGML_CUDA_DISABLE_GRAPHS=1).

### F2: prefill misses are synchronous and serialized with compute

Same as F2/F4 of the Phase-1 note, now quantified: slot30 costs ~6% pp
(127.8 vs 135.4 t/s baseline) because every miss blocks the ubatch on a
host->VRAM copy inside the remap op. Decode gained ~11% tg from caching; the
goal is to keep that while making pp >= the no-slot baseline.

### F3: overlap requires a second stream, which llama.cpp cannot reach

`ggml_backend_tensor_set_async` exists but issues on the backend COMPUTE
stream (ggml-cuda.cu:2638) - same-stream copies serialize with kernels, so it
hides nothing. True overlap needs a dedicated copy stream plus events. ggml
does not expose either to llama.cpp today. We add a minimal extension API
(see Phase 2b below); ownership note: this lives in ggml-cuda (shared with
CUDA), guarded so other backends are untouched.

### F4: selection data for chunk l+1 is only known while chunk l runs

Router logits depend on earlier layers, so "prefetch layer l+1 while l
computes" cannot know exact ids ahead of time. FreeToken prefetches all of
l+1; for 256-expert Qwen that is most of the layer (~all experts hot in a
long-context chunk), i.e. near-worst-case traffic. The practical predictor
for this fork is the per-layer top-k union of the PREVIOUS chunk (router
choices are temporally stable across adjacent chunks of the same document),
recorded already by the remap op (`record_selections`) for the decode
prefetcher. Wrong predictions cost bandwidth only; they never become visible
because predicted slots stay non-resident until their copy completes.

## 3. Phase 2a design: capture-safe argsort/top_k workspaces

1. New helpers in argsort.cu (declared in argsort.cuh):
   `argsort_f32_i32_reserve_capture_pool(ggml_cuda_pool &, const ggml_tensor * src0)`
   - mirrors EXACTLY the dispatch of `ggml_cuda_op_argsort` (bitonic early-out
     under CUB when ncols<=1024 and shared mem fits; unconditional hipcub on
     HIP; chunking via `*_chunk_nrows`),
   - for each chunk, obtains `temp_storage_bytes` via the null-pointer query,
   - allocates the full simultaneous set (keys, indices, offsets, storage)
     inside one scope and frees it, leaving correctly sized buffers in the
     pool free list. Because the four allocations are made before any of them
     is freed, they land in distinct pool buffers - matching the lifetime of
     the real sort call.
2. Same pattern for top_k (`top_k_f32_i32_reserve_capture_pool` in top-k.cu)
   covering DeviceTopK (CCCL>=3.2) and the argsort-fallback variants.
3. Hook in `ggml_backend_cuda_graph_compute`: immediately BEFORE
   `cudaStreamBeginCapture`, walk `cgraph->nodes` and reserve for every
   ARGSORT / TOP_K node. Sampler graphs go through the same entry point, so
   vocab-sized top_k is covered too.
4. Always-on (bug fix, not a feature flag). No behavior change when graphs
   are disabled.

Failure containment: if reservation itself cannot get memory (OOM outside
capture), it degrades to no-op - capture-time behavior then equals today's
(including the possible abort), which is no worse than status quo.

## 4. Phase 2b design: prefill double buffering (side-stream prefetch)

Goal: miss H2D overlaps GPU compute instead of stalling it.

### Extension API (minimal, in ggml-cuda)

Declared in ggml/include/ggml-cuda.h, implemented in ggml-cuda.cu (works for
HIP and CUDA alike; returns NULL elsewhere so callers can fall back):

- `ggml_backend_cuda_ext_copy_stream(backend)` - dedicated copy stream
  (created lazily per backend context, never the compute stream).
- `ggml_backend_cuda_ext_h2d_async(backend, tensor, offset, host, size)` -
  `cudaMemcpyAsync` on the copy stream into `tensor->data + offset`.
- `ggml_backend_cuda_ext_event_create/record/query/synchronize/destroy` -
  thin wrappers over cudaEvent{CreateWithFlags(DisableTiming),
  Record, Query, Synchronize, Destroy} recorded on the copy stream.

### Runtime flow (opt-in: LLAMA_MOE_PREFILL_PF=1, budget via LLAMA_MOE_PREFILL_PF_MB)

Hooked in src/llama-context.cpp next to the existing decode prefetch hook,
but gated on `n_tokens_all > 1` (prefill ubatches). The two paths are
mutually exclusive per step.

At each prefill ubatch boundary (previous graph has been ENQUEUED and is
likely still running; next graph not yet built):

1. Poll outstanding events; completed copies flip their slot to resident
   under `cache_mutex` (lock order preserved: cache_mutex -> access_mutex).
2. Prediction: per MoE layer, take last chunk's selections
   (`take_last_selections()` - decode prefetch consumes the same ring only in
   decode steps, so no contention), union across layers' worth of ids.
3. For each predicted id not resident and not already in flight:
   `preload_or_assign_slot` (public cache API - assigns a free/LRU slot and
   maps expert->slot) with a fresh clock value, then enqueue
   `ext_h2d_async` per bank tensor slice directly into the assigned slot's
   bank region, record one event per expert.
4. Caps: total bytes per step (default 256 MB) and in-flight entries
   (default 64). Pageable sources (weights are mmap-backed) are staged by the
   driver; host-side call cost is a RAM memcpy, DMA continues asynchronously.

Residency/correctness rules (the invariant from the Phase-1 note - "synchronous
copies happen only while no graph is in flight" - is RELAXED, not broken):

- An in-flight slot is NOT resident: `ensure_resident` (Coder A code,
  untouched) treats it as a miss and may synchronously materialize the same
  expert into a different slot. Double-copy wastes bandwidth once; readers
  only ever see fully-written regions.
- The in-flight slot holds a valid expert->slot mapping with a bumped clock,
  so LRU eviction picks other victims; while in flight its bank region is
  written by nobody else. Known limitation: adversarial churn could evict an
  in-flight slot (would require touching ensure_resident to fully close -
  deferred, documented).
- Bank slices of DIFFERENT layers are disjoint; within a layer, one slot is
  written by at most one in-flight copy. No graph reads non-resident slots.

Decode path unchanged (`llama_moe_gpu_expert_slot_prefetch` untouched).

Expected effect: with ~200 unique experts x ~2 MiB per layer chunk and
~20 GB/s effective PCIe Gen4 H2D, transfer ~= 20 ms per heavy layer; overlapped
with attention+MoE compute of the neighbouring layers, the sync-stall share of
the 6% pp penalty should drop to near zero. Acceptance: slot30 pp >= 135.4 t/s
(no-slot baseline) at tg >= 11.84 t/s.

## 5. Phase 2c design: elastic VRAM sizing (--moe-gpu-expert-slot-num auto)

- Parsing (common/arg.cpp): accept literal `auto` ->
  `params.moe_gpu_expert_slot_auto = true` (numeric field stays -1).
- Load-time placement: expert tensors are routed off-GPU when
  `(num >= 0 || auto)` - same treatment as a positive manual count
  (src/llama-model.cpp:1724, common.cpp gating likewise).
- Resolution: at the END of llama_context construction (KV caches allocated),
  before the first token is served:
  1. `ggml_backend_dev_memory` on the primary GPU device -> free bytes.
  2. Budget = free * margin (default 0.80; env LLAMA_MOE_AUTO_VMARGIN_FRACTION).
  3. Unit cost = sum over MoE layers of one-expert bytes (sum of
     `nb[expert_dim]` of every expert tensor) - i.e. bytes per slot across
     ALL banks.
  4. slots = floor(budget / unit), clamped to [n_expert_used, n_expert] and
     optional env cap LLAMA_MOE_AUTO_SLOT_CAP.
  5. Init the cache with that count, wire materialize_cb, run the standard
     preload, and LOG the full rationale (free, margin, unit, cap, chosen N).
- Structure keeps dynamic re-partitioning possible: resolution is just
  "compute N then call the same init path as manual mode"; nothing at load
  time hard-codes the count beyond routing weights off-GPU.
- With auto, load-time preload is skipped (cache not yet enabled) and runs
  once after resolution instead.

## 6. File ownership (Coder A parallel work)

Owned here: ggml/src/ggml-cuda/argsort.{cu,cuh}, top-k.{cu,cuh},
ggml-cuda.cu (reservation hook + ext API), ggml/include/ggml-cuda.h (ext
API decls), src/llama-context.cpp (prefill hook), src/llama.cpp
(preload/bank/resolution area), common/arg.cpp + common/common.{h,cpp}
(auto param plumbing). NOT touched: ensure_resident/LRU/evict/remap internals
(A-owned); access strictly through existing public cache APIs
(preload_or_assign_slot, take_last_selections, bank_for_layer).

## 7. Validation ladder

1. Build: `cmake --build build-hip` clean; `llama-server --version` boots.
2. Graph fix: 3130-token and 4670-token prefills with graphs ENABLED (no
   GGML_CUDA_DISABLE_GRAPHS) - zero aborts, capture succeeds repeatedly;
   tg parity with graphs-off run.
3. Baseline re-measure: Qwen3.6-35B-A3B, slot-off vs slot30 pp/tg.
4. Double buffering ON: slot30 pp >= 135.4 t/s, tg >= 11.84 t/s.
5. `--moe-gpu-expert-slot-num auto`: rationale log line sane; performance
   >= manual slot30.
6. Results appended to bench_results.txt (tab-separated, one config per line).
