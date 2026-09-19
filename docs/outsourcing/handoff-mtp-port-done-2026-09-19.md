# Handoff — qwen4exp MTP (draft-mtp) port: code-complete, awaiting GPU test

Date: 2026-09-19. Session A (Buffy). Status: **all code landed and compiles clean; no GPU run yet** (GPU was busy during this session — user rule).

## Git

Committed as `e77458c7d` on `feat/qwen4exp-hot-expert` and pushed to origin. Files in the commit: `src/models/qwen4exp.cpp`, `src/models/models.h`, `src/llama-model.cpp`, `common/speculative.cpp`, `tools/prep-mtp-draft-qwen38.py`, this handoff. (Other sessions' uncommitted work — AGENTS.md, run-*.cmd, bailingmoe3/kimi-k3 .disabled moves, bench scripts — was deliberately left out.)

## Goal

Port the qwen-next MTP/NextN draft head (commits `140c6a5` + `f8bb068` in `C:\llama-qwen-next`, upstream PR #28243 lineage) into this fork so `--spec-type draft-mtp` works with Qwen3.8-Flash-Next. Expected win on the same-class 27B was 16 → 28 t/s; baseline here is 13.4 t/s → target 20 t/s.

## Code changes (this session)

| File | Change |
|---|---|
| `src/models/qwen4exp.cpp` | `LLM_KV_NEXTN_PREDICT_LAYERS` optional read in `load_arch_hparams`; `mtp_only` probe (`blk.0.hc_attn_norm.weight` absent) → `trunk_flags` on **all** trunk tensors; MTP block loader loop (il 48, `ml.load_mtp ? 0 : TENSOR_SKIP`, indexer tensors always SKIP, nextn enorm/hnorm/eh_proj); members-only `graph(model, params, bool)` ctor; full `graph_mtp` (eh_proj concat of enorm(e)+hnorm(h) → hc_mix/attn(hybrid mctx nullptr → dense)/hc_combine → hc_mix/FFN → hc_combine → `res->t_h_nextn = flat [hc*n_embd, T]` → out_ids select → hc_head mix → output); trunk graph now hands `res->t_h_nextn = res_hc` before the final head mix; `build_arch_graph` dispatches `LLM_GRAPH_TYPE_DECODER_MTP` → `graph_mtp`; `qsa` guard `mctx_hyb != nullptr` |
| `src/models/models.h` | qwen4exp `graph_mtp : graph` decl + members-only ctor decl (was already half-prepared; completed) |
| `src/llama-model.cpp` | `mtp_on_hybrid_qwen` now includes `LLM_ARCH_QWEN4EXP` → MTP context gets a plain `llama_kv_cache` with filter `il >= n_layer()` instead of the hybrid_idx wrapper |
| `common/speculative.cpp` | fixed draft path bug: load `model_path` (the -md file), was passing `params.model.path` (the target) |

`load_mtp` was already wired: `common.cpp:1655` sets it when `draft-mtp` is in `--spec-type`.

## Draft GGUF

Built `tools/prep-mtp-draft-qwen38.py` (repo gguf-py, CPU only). The shipped draft GGUF is "thin" (32 tensors, all `blk.48.*`, no tok_embd/output) which no loader accepts. The script writes a standalone draft:

**`C:\Users\dai86\Downloads\mtp-draft\mtp-Qwen3.8-Flash-Next-draft-q8_0.gguf` (3.06 GB)**

- `token_embd.weight` Q8_0 copied from trunk Q2_0 shard1 (776 MB) — graph_mtp reads tok_embd REQUIRED
- `blk.48.nextn.hc_head_{norm,down,up}` renamed → `output_hc_{norm,down,up}` (loader's names)
- KV copied incl. `block_count=49`, `nextn_predict_layers=1`, `compress_ratios` (49, blk.48=0 dense), flat rope_sections `[11,11,10,0]`, full tokenizer
- verified by re-read: 33 tensors, arch qwen4exp, eos 248046

## Build

`build-mtp/` (new, stage1 untouched): Ninja / RelWithDebInfo / HIP gfx1101 / HIP_GRAPHS / MMQ_MFMA / NO_VMM, `CMAKE_PREFIX_PATH=...rocm-sdk-core\_rocm_sdk_core`, RC from Windows Kits 22621, HIP_PLATFORM=amd. **llama-cli / llama-server / llama.dll all link.** Config was rebuilt from scratch — `build-stage1/CMakeCache.txt` has the canonical flags (incl. the rocm-path CXX/HIP flags and RC path) if it needs recreating.

## Run when GPU frees (NOT yet executed)

```cmd
build-mtp\bin\llama-server.exe -m <GSQ-RCO-Q2_0 or IQ3_XXS shard1> -md C:\Users\dai86\Downloads\mtp-draft\mtp-Qwen3.8-Flash-Next-draft-q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 --moe-hot-expert -ngl 99 -fa on -c 8192 -t 12
```

Fallback to GPU-light draft: add `-ngla 0`-style override on the draft via `--spec-*`/`-md` tensor-buft overrides if the 2.5 GB draft MoE + target doesn't fit at 8K ctx (H-3 keeps the draft out of the expert-slot cache already).

Expected success markers: log line `loading draft model '...draft-q8_0.gguf'`, `n_layer_nextn = 1` on the draft load, `creating MTP context`/draft KV allocate, then acceptance stats in the speculative logs. Judgement metric is **gen t/s vs the 13.4 baseline** (common rule: build-stage1 numbers only otherwise; this new build is the MTP A/B).

## Risks / notes for the tester

- Trunk GGUF has no `nextn_predict_layers` → my optional read keeps `n_layer_nextn=0`; main-model behaviour unchanged (t_h_nextn handover is only consumed when `embeddings_nextn` is set).
- If the draft load throws on a missing tensor, check which: the mtp loop covers every tensor class the trunk loop creates for a full-attention layer; PLE tensors are NOT created for blk.48 (draft GGUF has none — correct).
- `hc_attn_norm` shape in the draft is `[10240]` flat = `[n_embd, hc]` written flat; loader reshapes via TENSOR_ALLOW_RESHAPE (trunk files store the same flat form).
- Do NOT run with HOST_BANK=1 (standing rule).
