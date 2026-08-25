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

### F1: materialize_cb was never wired (correctness hazard, FIXED in Phase 1a)

`materialize_cb` (src/llama-model.h:625) was checked in `ensure_resident`
but no code assigned it. On a runtime miss the slot got LRU-assigned and its
id returned to the remap, but no weights were copied into the bank slice;
downstream `mul_mat_id` read stale bytes - silent weight corruption.

Reachability history: originally latent (full-slot preloads all; frequency
mode used with slots >= |whitelist|). Commit e684d233c removed the
whitelist-skip in `ensure_resident`, so non-whitelisted router selections in
frequency mode now take the miss path too, making the unfilled-slot return
reachable in a common config. Fixed by wiring the callback at cache init.

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

## 5. Later phases

- Phase 2: prefill streaming - per-chunk top-k union bulk placement with
  double buffering across layers. **Outsourced 2026-08-25** (coder B,
  branch `feat/prefill-double-buffer`, prompt:
  `docs/outsourcing/coder-b-prefill-double-buffer.md`). Bundled with the
  argsort graph-capture fix (§8.3) and `--moe-gpu-expert-slot-num auto`
  elastic sizing as prerequisites.
- Phase 3: q* split - decide PCIe fetch vs CPU GEMM per miss; requires
  measured effective H2D bandwidth vs DDR GEMV throughput; likely also
  requires exposing the backend stream for true overlap.
  **Outsourced 2026-08-25** (coder A, branch `feat/qstar-global-lru`,
  prompt: `docs/outsourcing/coder-a-qstar-global-lru.md`), bundled with
  globalizing the slot budget (per-layer fixed banks -> shared pool with
  cross-layer LRU eviction; the current per-layer partition cannot move
  budget from cold layers to hot ones).
- Semantic checkpoints (FreeToken technique 3): not on this roadmap yet;
  deferred to a later round after Phase 2/3 land.
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

## 7. Implementation status

Phase 1a (commit 2cc99fa06, build pending machine idle):

- materialize_cb wired at cache init (src/llama.cpp, cache init block)
- direct host->VRAM copy in llama_moe_gpu_expert_bank_copy_tensor when the
  source buffer is host-backed; staging fallback kept for other cases
- telemetry: LLAMA_MOE_SLOT_STATS=1 enables n_copy / copy_bytes / copy_ns
  counters with a log line every 4096 materializations

Phase 1b (inter-step speculative prefetch, opt-in):

- remap op records per-layer router selections of the finished step
  (record_selections / take_last_selections, cache_mutex guarded)
- llama_moe_gpu_expert_slot_prefetch runs after each single-token decode,
  before the next graph, bounded by LLAMA_MOE_PREFETCH_MS (default 0 = off);
  synchronous H2D is race-free because no graph is in flight
- known limitation: assumes serialized decode per model (single context or
  externally serialized contexts); multi-context concurrent decode against
  one model must not enable it until stream-level isolation lands

Phase 1c (runtime validation + lifetime fix, commits 14ebc64aa / ba24f2254 /
f43c5e4e8, validated 2026-08-25):

- remap userdata ownership moved from graph side to the cache
  (per-layer pooled unique_ptr in llama_moe_gpu_expert_cache, model
  lifetime). Fixes dangling userdata when the graph outlives the
  llm_graph_context; also fixed the gfx1101 build break.
- runtime validation on Ornith-1.5-35B-A3B and Qwen3.6-35B-A3B:
  slot_remap fires on all MoE layers, dynamic materialize works, no
  crash across repeated requests.
- diagnostic fprintf instrumentation (31 sites) removed after validation;
  only the stderr-unbuffer setvbuf kept.

## 8. Runtime measurements (2026-08-25, RX 7800 XT 16GB + 96GB RAM)

Model: Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated Q4_K (20.2 GB,
qwen35moe, 40 MoE layers, 256 experts, top-8). 32K ctx, KV q8_0, --cpu-moe,
-fa on, -t 6, GGML_CUDA_DISABLE_GRAPHS=1 (see §8.3). Long prompt = 4670
tokens, tg = 200 tokens.

| config | short tg | long pp / tg |
|---|---|---|
| slots disabled (pure CPU streaming) | 12.38 t/s | 135.4 / 10.67 t/s |
| slot30 (LRU) | 11.91 t/s | 127.8 / **11.84 t/s (+11%)** |
| slot30 + prefetch 100ms | 11.06 t/s | n/m |

Findings:

1. Cache works when routing has temporal locality: after a long prefill,
   decode is +11% vs slotless (long context keeps consecutive tokens on
   similar routing).
2. Short-prompt single-shot decode is -4% with slots: synchronous miss
   copies stall more than they save. This is exactly the regime q*
   (Phase 3) must fix.
3. Prefill pays a -6% slot tax (miss-copy churn) - the target of Phase 2
   double buffering.
4. Phase 1b prefetch at 100ms budget hurts single-shot decode (11.06):
   synchronous copies outweigh the idle-time saving. Its value must be
   re-evaluated under batched/consecutive requests, or folded into q*.
5. Low-locality adversarial model (Ornith-1.5-35B-A3B): ~12.4 t/s flat
   across slot30/slot96/slotless - per-token routing churn defeats the
   cache; q* is the only lever there.

Frequency placement (Pass 1/2) status: the DeepSeek-era BLOCKED note in
BENCH_RESULTS.md (remap deadlock) is stale - fixed by the cache_mutex
clock fix (5edf04758) and e684d233c. Pass 1 collection now works:
`ft_freq.json` holds a Qwen3.6-35B-A3B report (~6.9k tokens sampled).
Top-30/layer covers 50.9% of selections (33-61% per layer), top-96
85.1%; ~247/256 experts are touched, i.e. routing is diffuse.
Pass 2 was exercised on 2026-08-25 with negative results (hit rate
pinned at static coverage, no LRU adaptation, workload-mismatch
timeouts) - see docs/frequency-placement-findings.md, including the
finding that the collection-mode gate in ensure_resident disables
no-whitelist slot caching entirely (relevant to the global-LRU work).

### 8.3 Known bug: HIP graph capture crashes on long prefill (argsort)

Prefills of ~3130+ tokens crash with:

    ROCm error: operation not permitted when stream is capturing
    -> argsort_f32_i32_cuda_hipcub (DeviceSegmentedRadixSort)

ggml/src/ggml-cuda/argsort.cu already switches to the capture-safe
DeviceSegmentedRadixSort path, but on ROCm something still allocates
during capture (pool-miss hipMalloc of the temp workspace and/or hipcub
internals). Workaround: GGML_CUDA_DISABLE_GRAPHS=1 (used for all §8
measurements; costs graph wins). Assigned to coder B as the first
deliverable of feat/prefill-double-buffer.
