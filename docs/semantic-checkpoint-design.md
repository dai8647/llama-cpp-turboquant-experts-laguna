# Semantic Checkpoint Design Study (FreeToken technique 3) — Round-2 feasibility

Status: design study / feasibility. NOT scheduled for the current round (coder A =
q* + global LRU, coder B = prefill double-buffer + capture fix). This doc scopes the
one FreeToken technique with zero coverage so a future "coder C" can start immediately.

Target hardware unchanged: RX 7800 XT 16GB + 96GB RAM, ROCm 7.1, gfx1101.

## 1. What it is, and why it is a speed feature

Agentic workloads repeatedly EDIT their own history: a tool call returns garbage and
the agent retries with a corrected call; a chain-of-thought block is revised; a wrong
answer is regenerated with a different strategy. Each edit invalidates the KV cache
from the edit point onward, so the naive implementation re-prefills the entire prefix.
On a 6.5k-token agent prompt with a 32K context, that re-prefill is the dominant cost
of every retry loop — measured prefill here is ~135–157 t/s, so re-doing several
thousand tokens costs tens of seconds per retry.

A semantic checkpoint snapshots the recurrent state (KV cache, and optionally the MoE
expert-residency state) at semantic boundaries — end of a thinking block, immediately
before/after a tool call, end of an assistant turn. When history is edited, the engine
restores the nearest checkpoint strictly before the edit point and recomputes only the
tail, instead of the whole prefix. For an agent that retries from a stable prefix, this
turns an O(prefix) re-prefill into an O(tail) one.

This is complementary to A/B's work, not overlapping: A/B make each forward pass and
each prefill faster; semantic checkpoints reduce HOW MUCH has to be re-prefilled.

## 2. The concrete problem in this fork

llama-server keeps one KV sequence per slot. When a follow-up request arrives whose
prompt shares a prefix with the previous turn, the server can reuse the cached prefix
(longest-common-prefix reuse). But when the request EDITs an earlier part of the
history (not just appends), everything after the edit is stale and must be recomputed.
There is currently no way to "jump back" to a mid-prefix state, because the KV cache
only ever grows forward and is trimmed from the tail.

## 3. Existing machinery to build on (already in this fork)

- `llama_memory_seq_rm(mem, seq, p0, p1)` — delete a KV position range. This is the
  rollback primitive: to return to a checkpoint at position P, remove (P, end).
- `llama_memory_seq_cp(mem, seq_src, seq_dst, p0, p1)` — copy a position range to
  another seq id. This is the fork primitive: a checkpoint can be materialized as a
  separate seq id that stays alive while generation continues, so restoring is a
  seq swap rather than a re-copy.
- `llama_memory_seq_keep` / `llama_memory_seq_add` — supporting range ops.
- `llama_state_save_file` / `llama_state_load_file` — full state to disk. Heavy
  (whole-context serialize); only appropriate for long-lived cross-session anchors,
  not per-tool-call checkpoints.

So the lightweight path is in-memory seq forking/trimming; the file path is a fallback
for anchors that must survive a restart.

## 4. Design sketch

### 4.1 Boundary detection
Checkpoints are only useful at positions the agent is likely to edit back to. Detect:
- tokenizer-level special tokens (tool-call open/close, thinking-block delimiters).
  Qwen-family models emit recognizable tool/thinking markers; make the marker set a
  per-model-config list, defaulting to "end of each assistant turn".
- server-level turn boundaries (the server already knows where each assistant
  response ends and each tool result begins in a chat-templated conversation).
Prefer server-level turn boundaries (model-agnostic, always available); use special
tokens as a refinement when present.

### 4.2 Checkpoint representation
Two tiers:
- Hot (in-memory): a forked KV seq id via `llama_memory_seq_cp`, kept resident. Cost =
  KV bytes for [0, P]. Cheap to restore (seq swap). Limited by VRAM/RAM, so keep only
  the K most recent checkpoints (ring) and evict oldest.
- Warm (host-RAM / disk): demote evicted hot checkpoints by copying their KV range to
  host RAM (or `llama_state_save_file` for cross-session). Restore costs a RAM→VRAM
  KV copy, still far cheaper than re-prefill.
Host RAM is the natural tier here (96GB available), consistent with the fork's
"RAM as source of truth" philosophy.

### 4.3 Restore / rollback
On an edit at position E:
1. find the newest checkpoint with P <= E.
2. if hot: swap the active seq to that checkpoint's seq id, `llama_memory_seq_rm`
   anything past P, continue from P.
3. if warm: copy the checkpoint's KV range back into the active seq, then continue.
4. recompute only tokens in [P, new_end].

### 4.4 MoE expert-residency synergy (fork-specific, high value)
The MoE GPU expert slot cache (this fork's core feature) has its own state: which
experts are resident in VRAM banks. A naive KV restore leaves the expert-residency
state wherever the forward pass left it, which may be cold for the restored tail and
cause a burst of H2D materializations right after restore. Optionally snapshot the
residency map (slot→expert per layer + LRU clocks) alongside the KV checkpoint and
restore it too, so the tail recompute starts warm. This is cheap (integers, not GB)
and directly reuses coder A's global-LRU residency structures. Flag it as a
stretch goal, not a blocker.

## 5. Hard problems (honest list)

1. KV snapshot cost. A 32K-context q8_0 KV is multiple GB. Forking a hot checkpoint
   copies that range. Budget: cap hot checkpoints by VRAM, spill to host RAM. Must
   measure KV bytes/position for the target models before choosing K.
2. Boundary correctness. A checkpoint at the wrong position (mid-tool-argument, mid-
   thinking) gives a restore that silently produces wrong continuations. Default to
   conservative turn boundaries; make it opt-in per request.
3. Multi-slot / concurrency. The server runs multiple slots; checkpoints are per-slot
   state. Keep them out of any shared/global path. (Same serialized-decode assumption
   as the existing prefetcher — see moe-slot-cache-async-design.md §Phase 1b.)
4. Graph interaction. Restore changes the KV contents but not the graph topology, so
   captured graphs remain valid; however this must be re-verified once B's graphs-ON
   work lands. Low risk, but test it.
5. Position/seq bookkeeping. llama.cpp KV uses seq ids + positions; forking many
   checkpoints can exhaust seq ids or fragment the KV buffer. Needs a small allocator
   and a defrag policy.

## 6. Phased scope

- Phase S1 (feasibility bench, no new feature): measure re-prefill cost of an edited
  history vs an appended one on the existing server, to quantify the win. Pure
  measurement; establishes the baseline number to beat.
- Phase S2 (rollback-only): implement "restore to last turn boundary" using
  `llama_memory_seq_rm` (no forking yet). Smallest useful version; already removes
  the commonest retry case.
- Phase S3 (hot checkpoints): add seq-fork checkpoints (`llama_memory_seq_cp`) with a
  VRAM-bounded ring + host-RAM spill.
- Phase S4 (MoE residency restore): snapshot/restore the expert-residency map
  alongside KV (depends on A's global-LRU structures being merged).
- Phase S5 (cross-session anchors): `llama_state_save_file`-based anchors for
  long-lived agent sessions. Optional.

## 7. File ownership (deliberately disjoint from A and B)

- `src/llama-kv-cache*.cpp/.h` — checkpoint fork/trim/restore primitives.
- `src/llama-context.cpp` — seq swap / restore orchestration.
- `examples/server/server.cpp` (+ `common/chat.cpp` if needed) — boundary detection
  and the request-level "resume from checkpoint" plumbing.
- `include/llama.h` — small public API additions if the server can't express restore
  with existing seq ops.

None of these are owned by A (llama-model/llama.cpp MoE/llama-graph) or B
(ggml-cuda capture/argsort/llama-context prefill hooks). The only touch point is
llama-context.cpp, where B owns the prefill hook and this work owns restore — keep
them in separate functions to avoid merge friction.

## 8. Target numbers (acceptance criteria for a future round)

- S1: quantify — edited-history re-prefill currently costs ~ (prefix_tokens / pp_t/s);
  report the measured seconds for the 6.5k bench prompt.
- S2: restore-to-last-turn recompute cost <= tail-only recompute + 5%; correctness
  verified by byte-identical continuation vs a from-scratch run on the same edited
  history.
- S3: hot-checkpoint restore latency < RAM→VRAM KV copy time for the checkpoint range;
  no VRAM growth beyond the bounded ring.
- S4: post-restore H2D materialization burst reduced vs S3 (fewer misses in the first
  decoded steps after restore, via LLAMA_MOE_SLOT_STATS).

## 9. Open questions

- KV bytes/position for the target 35B–118B models at q8_0 KV (drives the hot ring
  size). Measure in S1.
- Does the server's existing longest-common-prefix reuse already cover the append-only
  case well enough that only true edits need checkpoints? (Likely yes — checkpoints
  target edits specifically.)
- Should checkpoint trigger be automatic (every turn) or request-driven (agent signals
  a stable boundary)? Start request-driven, add automatic later.
