# MoE decode path: q* bandwidth-adaptive split + global LRU slot pool

Status: Phase 2 design + implementation. Extends docs/moe-slot-cache-async-design.md
(Phase 1a/1b). Coder A scope: decode path only. Branch feat/qstar-global-lru.

Goal: make cache misses cheap enough that slot caching never loses to pure CPU
streaming on short single-shot decodes (baseline 12.38 vs 11.91 t/s), and beats
it on low-locality routing (Ornith, ~12.4 t/s flat today) by splitting misses
between PCIe transfer and host-side execution instead of always paying the
synchronous H2D stall.

## 1. Current implementation map

All references are to this repo as of commit 62f8d92c7 (branch main).

- Cache state: `llama_moe_gpu_expert_cache` (src/llama-model.h:624). Per-layer
  slot vectors sized `n_slots`, per-layer banks (`banks_by_layer`) allocated
  lazily with `n_bank_slots = min(n_slots, n_experts)` capacity
  (llama_moe_gpu_expert_bank_ensure, src/llama.cpp:512). LRU victim search is
  per-layer (`find_lru_victim`, src/llama-model.h:972): a hot layer can never
  take VRAM residency budget from a cold layer because every layer owns a full
  independent pool of `n_slots` positions.
- Eval-time remap: `build_moe_gpu_slot_ids` (src/llama-graph.cpp:1592) builds a
  `ggml_map_custom1` op executed by `llm_moe_gpu_slot_remap`
  (src/llama-graph.cpp:233) with n_tasks=1 (only ith==0 runs). Misses call
  `ensure_resident` (src/llama-model.h:1077), which assigns a slot and calls
  `materialize_cb` -> synchronous H2D copy inside graph compute; the whole
  decode step stalls per missed expert, serialized behind one thread.
- Graph wiring: when the slot cache is enabled the FFN weights are redirected
  to bank tensors via `compute_tensor_by_src` and `mul_mat_id` consumes the
  remapped slot ids (src/llama-graph.cpp:2130-2145). The graph is static;
  routing is unknown until eval time; there is no per-expert fallback branch:
  an expert is either in the bank (GPU computes it) or the copy stalls first.
- Prefill streaming / graph capture around llama-context.cpp is owned by coder
  B and out of scope here; nothing below touches that file or ggml sources.

## 2. Findings

### F1: per-layer pools waste residency exactly where it matters

With `--moe-gpu-expert-slot-num 30` each layer can hold up to 30 residents no
matter what the other layers hold. Long-prefill decode has strong temporal
locality (measured +11%), so most layers sit far below their cap while one or
two hot layers saturate theirs. The total resident count across layers stays
far below the VRAM the banks could hold, yet hot layers still miss. The fix is
accounting, not storage: cap the SUM of residents globally and evict the
globally-oldest resident, while keeping per-layer bank capacity at `n_slots`
so bank buffers stay lazy, stable for graph reuse/capture, and byte-compatible
with today's worst-case VRAM.

### F2: a transfer miss and a CPU-computed miss cost different things

Both paths stream the same weight bytes out of host DRAM, but:

- transfer: DRAM -> PCIe -> VRAM, then GPU still runs mul_mat_id. Today this
  is a synchronous stall of the entire step, serialized per miss behind the
  ith==0 remap thread. Cost ~= S/B_h2d + fixed call overhead, plus mmap cold
  page faults on first touch.
- CPU exec: DRAM -> CPU cores computing the expert's gate/up/down directly,
  replacing the GPU work. Cost ~= S/B_cpu where B_cpu is the multi-threaded
  GEMV streaming rate of the dedicated worker set. No VRAM write, no slot
  assignment, no eviction pressure.

Which wins depends on measured rates, not constants: q* = decide per miss from
calibrated bandwidths, re-evaluated continuously.

### F3: map_custom ops mirror input shape; outputs need packing tricks

`ggml_map_custom1_impl` dups the input tensor as output shape
(ggml/src/ggml.c:5975). Multi-output custom ops are built by padding the input
with a `ggml_fill` zero segment and slicing the output with views. Public API
pieces used: ggml_cast, ggml_concat, ggml_fill, ggml_view_* — no ggml changes.

### F4: quant kernels are not exported; reuse them via a mini graph

`ggml_vec_dot_q4_K_q8_K` & friends live in ggml-cpu without GGML_API, so they
cannot be linked from llama.dll against the shared build. Hand-rolling scalar
GEMV would be ~10x slower. Instead the CPU-exec engine builds ONE tiny static
ggml graph per MoE layer whose weight "tensors" are metadata-only tensors whose
->data pointers alias the original host/mmap weight storage (zero copies, any
expert id selectable at run time via mul_mat_id ids). It runs through
`ggml_graph_compute` on a dedicated persistent threadpool created with the
public GGML_BACKEND_API ggml_threadpool_new (ggml/include/ggml-cpu.h:58),
isolated from the outer inference pool — nested compute without deadlock.

### F5: activations reach the remap op only through graph edges

The remap callback cannot dereference arbitrary graph tensors: only its input
is guaranteed host-resident. Everything the CPU side needs (router selections,
FFN input vector) must flow into the custom op as packed concat inputs; results
flow out through the op output and ordinary downstream nodes. This keeps every
access sched-visible and HIP-capture safe.

## 3. Design

### 3.1 Global LRU slot pool (opt-in)

Flag: `--moe-gpu-expert-global-lru` or env `LLAMA_MOE_GLOBAL_LRU=1`. Default
off; legacy per-layer semantics preserved exactly when off.

State added to the cache: `n_resident_global`, `n_evict_cross`,
`global_lru_enabled`. Counter maintenance covers assign_slot (fresh assign +
1), release_slot (-1 if was resident), clear/init (reset).

Eviction policy in ensure_resident when enabled (dynamic paging regime):

1. pos = find_free(layer); if found AND n_resident_global < budget: use it.
2. If global budget is full, evict the globally-oldest resident
   (`find_global_lru_victim`: scan all layers' slots for min last_used). This
   may free a position in another layer (cross eviction, counted separately).
3. Re-find a local free position after any cross eviction; assign.

Invariant kept simple by sizing every layer's capacity at `n_slots` (= budget):
a locally-full layer implies a globally-full-and-entirely-local cache, so the
victim search degenerates safely. Frequency-whitelist placement keeps legacy
per-layer behavior (static Pass 2 layout must not churn).

Bank storage unchanged: per-layer lazy buffers, capacity n_slots, stable
pointers for graph reuse. Typical VRAM drops versus legacy because fewer
layers get banks touched at all.

Telemetry: extended LLAMA_MOE_SLOT_STATS line adds residents=<global count>
cross_evict=<count>.

### 3.2 q* policy (Phase A)

Flags/env: `--moe-qstar`, `LLAMA_MOE_QSTAR=1`; knobs `LLAMA_MOE_QSTAR_THREADS`
(dedicated GEMM workers, default 3), `LLAMA_MOE_QSTAR_BUDGET_US` (per-layer
per-step cumulative transfer budget, default 300), `LLAMA_MOE_QSTAR_STATS=1`.

Calibration at load (after tensors, before serving):

- B_h2d_eff: materialize a real expert slice 3 times via the existing direct
  H2D path, keep median rate; also derives the fixed per-copy overhead.
- B_cpu_eff: build the mini graph for layer 0 and time 3 executions with a
  synthetic activation over one expert; bytes/time gives the GEMV streaming
  rate of the dedicated pool.
- Runtime refinement: B_h2d EMA updated from telemetry copy_ns/copy_bytes;
  B_cpu EMA updated from actual CPU-exec timings.

Decision, applied per missed expert inside the planning pass of the remap op:

    t_xfer(e) = S_e / B_h2d_ema + T_fix
    t_cpu(e)  = S_e / B_cpu_ema
    choose transfer iff t_xfer(e) <= t_cpu(e)
                       AND budget_used_layer_step + t_xfer(e) <= budget

Overflow misses go to CPU execution. Deferred experts consume no slots, so the
global-LRU pool only ever tracks genuinely-resident experts. Telemetry counts
n_qstar_xfer / n_qstar_cpu plus current EMAs, logged with SLOT_STATS and per-
step with QSTAR_STATS ("layer=L xfer=a cpu=b budget_left=us").

True overlap of the two paths inside one remap call (transfer issue on one
thread, GEMM on others via cooperative nth partitioning of the custom op) is
designed but deferred: v1 executes serially inside the op — transfers first up
to budget, then ONE batched multi-threaded CPU GEMM. This already removes the
unbounded per-miss stall; overlap is a follow-up once measurements show the
serial sum dominating.

### 3.3 q* graph structure (Phase B, decode-only)

Built only when q* is on AND the ubatch being built for is decode-shaped
(n_tokens == 1) AND the dynamic paging regime applies. Other shapes build the
legacy structure unchanged; graph caching keys by shape so both coexist.

Per MoE layer:

    ids_f32        = cast(selected_experts, F32)              # [n_used]
    in_a           = concat(ids_f32, fill(0, n_used))         # [2*n_used]
    plan           = map_custom1(in_a, slot_plan_fn)          # [slots | mask]
    slot_ids       = cast(view(plan[0:n_used]), I32)          # feeds mul_mat_id
    gpu_mask       = view(plan[n_used:2*n_used])              # f32 0/1 per selection
    weights_masked = weights * gpu_mask                       # zeroes CPU-exec'd rows
    ... existing bank mul_mat_id chain with slot_ids ...
    moe_out_gpu    = aggregate(experts * weights_masked)      # existing code path
    in_b           = concat(gpu_mask, flat(cur), pad)         # [n_used*n_embd]
    partials       = map_custom1(in_b, cpu_exec_fn)           # [i*n_embd+d], zero-filled
    moe_out_cpu    = sum_i view_i(partials) * weights_i       # router weights applied here
    result         = moe_out_gpu + moe_out_cpu

Planning op (slot_plan_fn), ith==0 only:

1. Classify all selections hit/miss (ensure_resident bookkeeping for access
   stats); misses decided per §3.2. Transferred misses: ensure_resident +
   materialize as today, batched back-to-back.
2. CPU-executed entries: emit dummy slot id + mask 0. Dummy id points at a
   guaranteed-valid resident slot (the first hit of the step; if the step had
   zero hits, force-transfer the first miss so at least one valid slot exists)
   — prevents mul_mat_id touching uninitialized VRAM whose NaNs would survive
   masking.
3. Record the deferred list into pooled userdata for the exec op; record
   selections for prefetch as today.

Exec op (cpu_exec_fn), ith==0: zero-fills its output, then for each deferred
expert runs the layer mini graph (§3.4) accumulating unweighted outputs into
the partials region. Reads its inputs strictly from the packed buffer (mask to
know which rows are live, cur vector for activations).

The extra D2H of `cur` per layer per step (~10 KB decode) rides the sync point
the remap already forces; prefill shapes never pay it.

### 3.4 CPU-exec mini graph engine

Per layer, lazily built on first CPU exec (state in the cache struct):

- Metadata-only ctx (no_alloc). Weight tensors shaped like the originals with
  ->data aliased into the source host tensors; nb[] copied. Supports gate+up
  separate or fused gate_up layouts; down likewise.
- Activations: x [n_embd] f32 scratch (copied in per call from packed input);
  h [n_ff]; y [n_embd, r_max].
- Graph: h = swiglu(mul_mat_id(W_gate,x,ids), W_up...) ; y = mul_mat_id(W_down,h,ids).
  The token vector is shared across selected experts via a zero-stride view.
  ids tensor data pointer swapped per call.
- Compute: ggml_graph_compute with cplan.threadpool = dedicated pool
  (LLAMA_MOE_QSTAR_THREADS workers), work buffer sized once via
  ggml_graph_plan and cached.
- Per-tensor scale (_s) supported only when scalar; bias (_b) and non-scalar
  scales disable the CPU path for that layer with a one-time warning (falls
  back to transfer-always there).

Calibration reuses this engine (§3.2).

### 3.5 Interaction with existing features

- Inter-step prefetch (Phase 1b): unchanged, still opt-in via
  LLAMA_MOE_PREFETCH_MS. With q* on, prefetch warms likely-next experts so the
  transfer side hits more often; the CPU side makes wrong predictions cheap.
  Measured interaction goes into bench_results.txt rather than assumptions.
- Frequency whitelist mode: q* graph variant not used (whitelist layers are
  fully placed by construction); collection mode unaffected.
- Lock order cache_mutex -> access_mutex untouched; the mini graph state lives
  under cache_mutex like all other cache fields. The dedicated threadpool
  never touches cache structures.

## 4. Validation ladder (machine idle required)

1. Build: cmake --build build-hip, error 0; llama-server --version OK.
2. smoke_moe.ps1: slot_remap fires on all layers, no crash, legacy configs
   byte-identical behavior (features off).
3. Qwen3.6-35B-A3B baselines re-measured (slot-off, slot30, slot30+prefetch)
   with new binary; regressions block enabling new flags.
4. Global LRU on: long-text tg >= 11.84 t/s held; short-text >= 12.38 t/s
   (target: close the -4% gap); telemetry shows cross evictions > 0 and
   residents concentrating in hot layers.
5. q* on (+ global LRU): Ornith low-locality tg > 12.4 t/s (beat slot-off);
   short-text no regression vs slot-off; SLOT_STATS logs show xfer/cpu splits.
6. All results appended to bench_results.txt (tab-separated), PR carries
   baseline comparison table + telemetry excerpts + exact commands.

## 5. Acceptance criteria

- Short single-shot decode: slot-enabled >= slot-disabled (12.38 t/s baseline).
- Long post-prefill decode: >= 11.84 t/s maintained.
- Ornith (low locality): >= 12.4 t/s with q* + global LRU.
- LLAMA_MOE_SLOT_STATS reports hit/miss/evict/cross-evict/xfer/cpu ratios.
- Default-off everywhere; no fprintf left behind; commits split per feature
  (moe : ..., llama : ..., present tense).
