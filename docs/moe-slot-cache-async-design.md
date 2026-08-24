# MoE GPU expert slot cache - async materialization design note

Status: Phase 0 (design). Implementation target: Phase 1a/1b of the
FreeToken-style expert caching roadmap.

Goal: run 35B-118B MoE GGUFs (Ornith-35B, LongCat-Flash-Lite, Laguna-S-118B,
DeepSeek-V4-Flash Q2) on RX 7800 XT 16GB + 96GB RAM by keeping only hot
experts in VRAM banks and making cache misses cheap.

## 1. Current implementation map

All references are to this repo as of commit f84c58d3a.

- Cache state: `llama_moe_gpu_expert_cache` (src/llama-model.h:615)
  - `slots_by_layer` / `banks_by_layer` / `expert_to_slot`, LRU `clock`,
    hit/miss/evict counters
  - guarded by `cache_mutex` (recursive); lock order cache_mutex -> access_mutex
- Bank storage: one packed VRAM buffer per MoE layer holding `n_slots` copies
  of every expert tensor (gate/up/down variants), see
  `llama_moe_gpu_expert_bank_ensure` (src/llama.cpp:479)
- Eval-time remap: per MoE layer, `ggml_map_custom1` op built by
  `build_moe_gpu_slot_ids` (src/llama-graph.cpp:1621), executed by
  `llm_moe_gpu_slot_remap` (src/llama-graph.cpp:266) on the CPU thread pool
  during graph compute. For each router-selected expert it calls
  `ensure_resident` (src/llama-model.h:1007): hit bumps `last_used`, miss does
  LRU assign then calls `materialize_cb`.
- Materialization: `llama_moe_gpu_expert_bank_copy_tensor`
  (src/llama.cpp:474) allocates a heap staging vector, does
  `ggml_backend_tensor_get` (host -> host memcpy, sources live in host/mmap
  buffers), then `ggml_backend_tensor_set` (blocking H2D into the bank).
- Load-time preload: `llama_moe_gpu_expert_slot_preload` (src/llama.cpp:799)
  fills banks for whitelist/full-slot modes before serving.
- Frequency workflow: Pass 1 collection mode records accesses into
  `access_counts`; report -> whitelist -> Pass 2 static placement.

## 2. Findings

### F1: materialize_cb is never wired (correctness hazard)

`materialize_cb` (src/llama-model.h:625) is checked in `ensure_resident`
(src/llama-model.h:1056) but no code assigns it. Consequences:

- On a runtime miss the slot gets LRU-assigned and its id returned to the
  remap, but no weights are copied into the bank slice.
- Downstream `mul_mat_id` then reads whatever occupies that slot (stale or
  zero) - silent weight corruption.
- Today this stays latent because (a) full-slot mode preloads everything and
  never misses at runtime, and (b) frequency mode is typically used with
  `n_slots >= |whitelist|`, so runtime misses cannot happen. Any config with
  `slots < |whitelist|` (or future dynamic eviction) hits this path.

Phase 1a must wire the callback before adding async behavior.

### F2: miss cost is three copies deep and fully synchronous

Per missed expert inside the remap op:

1. `std::vector<uint8_t> data(nbytes)` heap allocation
2. `ggml_backend_tensor_get`: host(mmap) -> heap staging memcpy
3. `ggml_backend_tensor_set`: blocking H2D

Sources are always host memory here (model weights are CPU-buffer backed),
so step 2 is pure overhead. Step 1 can be removed entirely. The blocking H2D
stalls the whole decode step while the custom op runs.

### F3: remap is single-threaded by design

`llm_moe_gpu_slot_remap` returns immediately unless `ith == 0`, so one CPU
thread serializes all misses of a layer. Fine for bookkeeping; matters when
we batch copies.

### F4: graphs/capture

The remap is already a CPU custom op, so HIP graph capture already has to
split around it today. Inter-step prefetch work happens outside any graph and
cannot affect capture. Intra-graph async copies (side stream + event gating)
would need access to the backend compute stream, which ggml does not expose
to llama.cpp; deferred to Phase 3+.

## 3. Phase 1a design (sync-path fixup)

Scope: correct, minimal, no new threading.

1. Wire the callback at model load: set
   `cache.materialize_cb = llama_moe_gpu_expert_slot_materialize_cb`,
   `materialize_userdata = &model` next to where the cache is initialized
   (same place preload params are set).
2. Direct copy in `llama_moe_gpu_expert_bank_copy_tensor`:
   if `ggml_backend_buffer_is_host(bank_tensor.src->buffer)` then call
   `ggml_backend_tensor_set(bank_tensor.dev, (const uint8_t*)bank_tensor.src->data + src_offset, dst_offset, nbytes)`
   directly; drop the staging vector. Keep the generic get/set fallback for
   non-host sources (future-proofing).
3. Batch within a layer: extend the remap userdata with a small
   pending-materialize list; first pass classifies hit/miss/assign for all
   selected ids, second pass materializes misses back-to-back (amortizes
   per-call device lookups; keeps lock hold time bounded).
4. Telemetry: env `LLAMA_MOE_SLOT_STATS=1` enables periodic (every N steps)
   LLAMA_LOG_INFO line: hit/miss/evict totals, bytes copied, accumulated
   copy time. Counters live in the cache struct under existing locks.

Fallback: direct-copy failure keeps old staging path. Behavior with feature
disabled unchanged.

## 4. Phase 1b design (inter-step speculative prefetch)

Constraint: no intra-graph async yet (needs backend stream access), so hide
latency between decode steps instead, where the GPU is idle and plain
synchronous copies are race-free.

1. During remap, record per-layer selected expert ids of the finished step
   (small ring buffer in the cache, already under cache_mutex).
2. After a decode ubatch completes (hook in llama-context where the graph
   compute returns, before the next sched step), run an inline prefetch pass
   with a wall-clock budget (default ~2 ms, env `LLAMA_MOE_PREFETCH_US`):
   for each layer, take last-step selections (+ optionally previous-step
   union), call `preload_or_assign_slot` + materialize for ids not resident,
   skipping anything already hit.
3. Decode has strong temporal locality in router choices; predicted-wrong
   prefetches simply warm LRU order and cost bandwidth only.
4. No worker thread in v1 (avoids races with graph build/compute and keeps
   locking trivial). A background thread variant is a follow-up once
   measurements show the inline budget truncating useful prefetches.

## 5. Later phases (recorded, not designed here)

- Phase 2: prefill streaming - per-chunk top-k union bulk placement with
  double buffering across layers.
- Phase 3: q* split - decide PCIe fetch vs CPU GEMM per miss; requires
  measured effective H2D bandwidth vs DDR GEMV throughput; likely also
  requires exposing the backend stream for true overlap.
- Multi-tier note: keep mmap enabled so RAM acts as tier 2 and NVMe as tier 3;
  prefetcher doubles as RAM warmer. DeepSeek-V4-Flash Q2 (86.7 GB) fits 96 GB
  RAM but leaves little slack, making tiers 2/3 mandatory.

## 6. Validation ladder (machine idle required)

1. Qwen3.8-27B IQ4_XS: harness sanity, hit-rate telemetry sanity.
2. Ornith-35B abliterated Q4 (~20 GB): slots sweep 25/50/75%, tg128/tg512.
3. LongCat-Flash-Lite Heretic MTP Q4_K_M (40.3 GB): arch port status check
   (longcat-flash-ngram + native MTP in this fork vs erm14254 fork).
4. Laguna-S-2.1 APEX Q6_K (74.7 GB) + DFLASH BF16 draft (2.23 GB):
   large-bank regime, DFlash interplay.
5. Huihui DeepSeek-V4-Flash-0731 abliterated Q2 (86.7 GB): final target,
   mmap/NVMe tier stress.
