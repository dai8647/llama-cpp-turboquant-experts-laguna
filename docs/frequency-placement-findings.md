# Frequency placement: findings from the dynamic-paging audit

Context: while wiring the q* policy and the global LRU pool (see
`moe-qstar-global-lru-design.md`) it turned out the collection-mode gate
in `ensure_resident` made several downstream features unreachable. This
note records the finding and the fix, since the original handoff
referenced a file that did not exist yet.

## Finding 1 — collection-mode gate disabled paging regimes entirely

`llama_moe_gpu_expert_cache::ensure_resident()` began with:

```cpp
if (frequency_whitelist.empty() && n_slots < n_experts) {
    return expert_id; // Pass 1 collection (track_access): count only
}
```

The comment said "collection mode", but the condition is really *any*
regime with fewer slots than experts and no whitelist — i.e. every
dynamic-paging configuration. Consequences:

- `ensure_resident` returned the logical expert id unchanged; nothing was
  ever made resident at eval time.
- The graph-level redirect in `build_moe_ffn`
  (`use_moe_gpu_slot_cache = full_layer || has_freq_whitelist`) never
  fired for that regime either, so bank tensors were not even wired into
  the graph.
- Net effect: running with e.g. 30 slots on a 256-expert model behaved
  identically to slot mode being off. Any feature built on top of dynamic
  residency (global LRU budget sharing, q* transfer-vs-CPU decisions) was
  dead code.

## Fix

Two coupled changes, both gated so legacy modes are untouched:

1. **Gate** (`src/llama-model.h`, `ensure_resident`): skip only when the
   run is actually collecting or when no dynamic-paging feature is on:

   ```cpp
   if (frequency_whitelist.empty() && n_slots < n_experts &&
           (track_access || !(global_lru_enabled || qstar_enabled))) {
       return expert_id;
   }
   ```

2. **Graph redirect** (`src/llama-graph.cpp`, `build_moe_ffn`): extend
   `use_moe_gpu_slot_cache` with `has_paging_dynamic` under exactly the
   complementary conditions (`size < n_expert`, whitelist empty,
   `!track_access`, and global-LRU or q* enabled), so bank tensors are
   redirected whenever the gate above starts assigning slots.

Both sides must stay in sync: redirect without the lifted gate yields
graphs indexing banks that never get filled; the lifted gate without
redirect makes residents that no graph ever reads.

## Finding 2 — unbanked fallback ids can escape into graphs

With paging active, a failed materialization used to leave
`expert_to_slot` pointing at an empty slot, or worse, let a logical id
larger than the bank capacity flow into `mul_mat_id` (out-of-bounds read).
Mitigations now in place:

- `release_slot()` undoes resident markings whose storage could not be
  materialized.
- `safe_unbanked_fallback()` returns a resident slot of the same layer
  (or 0) instead of an unbankable logical id whenever the graph indexes
  bank tensors (`n_slots < n_experts`).
- The q* planning op additionally verifies `find(layer, expert) == slot`
  after `ensure_resident` and routes the expert to host execution when
  the id does not match, plus a forced first-transfer so a valid dummy
  slot always exists for masked-out reads (uninitialized VRAM NaNs would
  otherwise survive zero masking).

## Status

- Gate + redirect: implemented (this branch).
- Global LRU pool: implemented behind `--moe-gpu-expert-global-lru` /
  `LLAMA_MOE_GLOBAL_LRU=1`.
- q* policy: implemented behind `--moe-qstar` / `LLAMA_MOE_QSTAR=1`.
- Frequency whitelist (static Pass 2) behavior: unchanged by design;
  whitelist layers keep per-layer semantics and never enter the dynamic
  paging path.
