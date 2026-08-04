# DeepSeek-V4 (REAP GGUF) on ROCm - Merge Result & Verification Guide

Date: 2026-08-03

## 1. What was done: upstream merge into the fork

The fork (`dai8647/llama-cpp-turboquant-experts-laguna`) did not have
`deepseek4` architecture support. Upstream `ggml-org/llama.cpp` added it
(PR #24162, "DeepSeek V4", merged 2026-06-29) plus follow-up fixes. Per the
requested policy ("follow upstream completely"), the latest upstream master
was merged into the fork.

- Work branch: `merge-upstream-next` (local; created from the previously
  pushed `merge-upstream` branch)
- Merge commit: `c461b2786ed0c3743a58cd63930f3d4f6257b375`
  - Parent 1 (fork side): `8bd5ba0b03` (`merge-upstream`, already contained
    deepseek4 base support #24162)
  - Parent 2 (upstream side): `fe2adf0e72` (upstream master, 2026-08-03)
- Result: `git rev-list --count upstream/master ^c461b278` == **0** -> fully
  caught up with upstream.

### Branch layout (as of 2026-08-03)

| Branch | Where | HEAD | Note |
|---|---|---|---|
| `main` | origin | `09ec1633` | Fork's default branch (MoE expert freq fix) |
| `bench/frequency-placement-fix` | origin | `09ec1633` | Same commit as `main` |
| `merge-upstream` | origin | `8bd5ba0b` | Upstream merge up to `f5919bf45` (2026-08-02), incl. #24162 |
| `merge-upstream-next` | local only | `c461b278` | This merge: upstream master up to `fe2adf0e72` |

Note: `feature/turboquant-kv-cache` (mentioned in `TURBOQUANT_UPSTREAM_MERGE.md`)
does **not** exist on origin. The fork's canonical branch on GitHub is `main`.

### Conflicts resolved (1 file: src/llama-kv-cache.cpp)

The 25-commit upstream range included a refactor (PR #26338, "M3: Move MSA
into a new memory implementation") that moved the MSA indexer out of
`llama_kv_cache` into a separate `llama-kv-cache-msa` class. The fork kept an
inline `k_idx` implementation (used in ~100 places). Resolution:

- `kv_layer` struct: followed upstream (dropped `k_idx`, `k_idx_stream` fields)
- `layers.push_back(...)`: followed upstream 5-field form
- Removed `llama_kv_cache_context::get_k_idx()` (upstream deleted it)
- **Kept** all TurboQuant fork deltas: `get_turbo_rotation(_inv)`,
  `get_turbo_rot_forward/inverse`, `get_turbo_innerq_scale_inv` wrappers,
  rotation-matrix init, auto-asymmetric K upgrade, layer-adaptive policy,
  attn-rotation default-off env knobs (`LLAMA_ATTN_ROT_K/V_OVERRIDE`,
  `LLAMA_ATTN_ROT_DISABLE`)

### DeepSeek-related commits now included (range f5919bf..fe2adf0e)

- dbadb68e ggml: use dynamic allocation for split graph inputs (#22789)
- 2b63e061 llama: MTP support for DeepSeek V3.2 (#26457)
- fffbcbdb metal: implement DeepSeek V4 hyper-connections (#26459)
- 596a5795 DeepseekV4 MTP + DSpark (#25784)
- plus the original #24162 (deepseek4 arch) from the earlier `merge-upstream`

### Verification done in this workspace

- `g++ -std=c++17 -fsyntax-only` passes for `src/llama-kv-cache.cpp`,
  `src/llama-graph.cpp`, `src/llama-memory-hybrid.cpp`
- Object compile (`-c`) passes for `src/llama-kv-cache.cpp` and
  `src/llama-memory-hybrid.cpp`
- `LLM_ARCH_DEEPSEEK4` registered in `src/llama-arch.h` / `llama-arch.cpp`
  (arch name `deepseek4`)
- Fork features intact: `src/llama-moe-placement.cpp`,
  `src/turbo-rotation-data.h`, `src/models/laguna.cpp`,
  `src/llama-kv-cache-dsa.cpp`, plus the new upstream
  `src/llama-kv-cache-msa.cpp` (wired into `src/CMakeLists.txt`)

Not yet verified here: full CMake build, GPU (HIP) build, actual model load.
Do that on the target machine using section 2.

## 2. Verification / run procedure on your machine

### 2a. Build (CPU / generic, quick smoke test)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build -j$(nproc) --target llama-cli llama-server
```

### 2b. Build for AMD ROCm (HIP)

Prerequisites: ROCm SDK installed, `hipconfig` on PATH.

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build-rocm -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release \
        -DAMDGPU_TARGETS="gfx942"   # adjust to your GPU (e.g. gfx1100 for RX 7900 XTX)
cmake --build build-rocm -j$(nproc) --target llama-server
```

### 2c. Run DeepSeek-V4-Flash-0731-REAP

Model: `heath0xFF/DeepSeek-V4-Flash-0731-REAP-GGUF`, architecture `deepseek4`,
~193B params, 160 routed experts (pruned from 256). Files (verified via HF API):

| File | Size | Note |
|---|---|---|
| `DeepSeek-V4-Flash-0731-REAP-K160-MXFP4.gguf` | 101.2 GB | Best quality in limited tests, but MXFP4 has no HIP kernels yet -> CPU fallback |
| `DeepSeek-V4-Flash-0731-REAP-K160-Q4_K_M.gguf` | 109.0 GB | Model card reports repetition issues; **use with `--cpu-moe`** |
| `DeepSeek-V4-Flash-0731-REAP-K160-Q2_K.gguf` | 65.4 GB | Same caveat |

Recommended (Q4_K_M + clean sampling, per the model card):

```bash
./build-rocm/bin/llama-server \
  -hf heath0xFF/DeepSeek-V4-Flash-0731-REAP-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 -c 32768 -ngl 99 \
  --cpu-moe \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --repeat-penalty 1.0
```

Notes:
- `--cpu-moe` keeps MoE weights on CPU. Proven workaround when GPU kernels
  crash or output is broken; also required on cards with < 109 GB VRAM.
- MXFP4 file: only if your card + build supports it; expect slow CPU paths.
- On MI300X (192 GB HBM) you can try dropping `--cpu-moe` after a smoke test.
- Known upstream issues to watch: #25837 / #26521 (Apple Silicon CPU repack),
  #26423 (deepseek4 + quantized KV broken; fixed by disabling attn-rotation -
  this fork defaults rotation OFF, which is the correct setting).

## 3. Safe merge point analysis

- The fork's `merge-upstream` branch already merged upstream up to
  `f5919bf45` (2026-08-02), which includes deepseek4 base support (#24162).
- The remaining 25 commits contained mostly non-deepseek fixes plus 4
  deepseek4 refinements (#22789, #26457, #26459, #25784). None of them
  conflict with TurboQuant except the MSA refactor (#26338) resolved above.
- No further upstream commits exist beyond `fe2adf0e72` in this workspace
  (`unmerged == 0`), so this merge point is as safe as it gets today.
- Recommendation for the future: keep merging on `merge-upstream` / create a
  fresh `merge-upstream-next` per merge round; do not merge `master` (the fork
  has no `master`; the default branch is `main`).

## 4. Handoff / next steps

- Review the merge commit locally: `git show --stat c461b278`
- When satisfied, push `merge-upstream-next` to the fork and open a PR into
  `main` via the Freebuff Changes panel / your normal GitHub flow.
- Run section 2a/2b builds on the machine that has the AMD GPU.
