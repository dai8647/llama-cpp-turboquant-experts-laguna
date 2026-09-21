# 統合ハンドオフ — Qwen3.8-Flash-Next 20 t/s: MTP 修正 + 転送ボトルネック (2026-09-21)

**あなた1人に全委託します。まず計画書 §3「全セッション共通ルール」を読んでください。**
`C:\Users\dai86\llama-cpp-turboquant-experts-laguna\docs\outsourcing\PLAN-qwen38-20tps-delegation-2026-09-19.md`

最大の地雷: **HOST_BANK=1 絶対禁止** (CPU95%/Disk100%/ROCm error)、**測定は build-stage1 / build-mtp のみ**、判定は gen t/s のみ。

---

## 現状サマリ (2つのブロッカー)

### ブロッカー A: MTP draft-mtp が draft ロードで落ちる (本命・最優先)

**症状**: `build-mtp` で draft GGUF の hparam 読み込みは成功 (context_length=262144, vocab, EOS 全部 OK) するが、**`load_tensors` で `invalid vector subscript`** で落ちる。

```
llama_model_load: error loading model: invalid vector subscript
common_speculative_init_result: failed to load draft model
```

**あなたの仮説検証ポイント** (コードを読んで確定させること):
- draft GGUF は `block_count=49` を宣言するが、テンソルは **MTP ブロック (blk.48) + token_embd + output_hc_* しか持たない** (33 tensor、trunk 層 0-47 の attn/ffn は無い)
- ローダーが `block_count=49` ぶんの全層テンソルを探しに行き、存在しない層 0-47 で vector 範囲外アクセスしている可能性が高い
- `handoff-mtp-port-done-2026-09-19.md` の `mtp_only` 判定ロジック (qwen4exp.cpp) が、draft 単体ロード時に trunk 層をスキップする設計だったはず — それが draft GGUF で正しく発火していない
- 参照: `C:\Users\dai86\llama-cpp-turboquant-experts-laguna\docs\outsourcing\handoff-mtp-port-done-2026-09-19.md`

**確認に使える事実** (私が GGUF を直接読んで検証済み):
- 新 draft: `context_length=262144`, `embedding_length=2560`, `embedding_length_per_layer_input=160`, `block_count=49`, `nextn_predict_layers=1`, `nextn_shared_target_tensors=True`
- テンソル 33 個 = blk.48.* (attn/ffn/hc/indexer/nextn) + `token_embd.weight` + `output_hc_{norm,down,up}`
- **trunk 層 0-47 のテンソルは意図的に無い** (MTP は最終層の1ブロックだけで推測する)

**やること**:
1. `src/models/qwen4exp.cpp` の MTP ロードパスで、draft ロード時に trunk 層テンソルを要求しないことを確認。`mtp_only` / `load_mtp` のフラグが draft ロードで正しく立つか。
2. `invalid vector subscript` がどのテンソルループで出るか、`-v` ログの直前の tensor 名で特定。
3. 修正して build-mtp を再ビルドし、ロードが通ることを確認。
4. 通ったら実機テスト (下記「MTP 実機テスト」)。

**MTP 実機テスト (ロードが通った後)**:
```cmd
set ROCM_PATH=C:\Users\dai86\rocm-sdk-core\_rocm_sdk_core
set HIP_PLATFORM=amd
set PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\llvm\bin;%PATH%
set LLAMA_MOE_HOST_BANK=0
set LLAMA_MOE_SLOT_STATS=1

build-mtp\bin\llama-cli.exe ^
  -m "C:\Users\dai86\.lmstudio\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" ^
  -md "C:\Users\dai86\Downloads\mtp-draft\mtp-Qwen3.8-Flash-Next-draft-q8_0.gguf" ^
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 ^
  --moe-hot-expert -ngl 99 -fa on -c 8192 -t 12 --reasoning off ^
  -st -v --log-file mtp_test3.log -p "Explain MoE routing concisely." -n 256
```
- 成功マーカー: `n_layer_nextn = 1`, `creating MTP context`, draft KV allocate, acceptance 統計
- 判定: gen t/s (ベースライン 13.4、目標 20)。acceptance < 0.5 なら報告のみ (無理にいじらない)
- ログ: `mtp_test3.log`

### ブロッカー B: 転送 78→39 MiB/s、主ボトルネックは H2D event wait (基盤・MTP 後)

**実測で確定したこと** (split telemetry、コミット 2798a878f):
```
stage_memcpy_ms = 7284.99   (ホスト memcpy)
h2d_wait_ms     = 26161.91  (H2D event 待ち)  ← 3.6x 大きい、主ボトルネック
stage_get_ms    = 0         (non-host backend read は未使用)
h2d_bw          = 245.1 MiB/s
copy_bw         = 39.4 MiB/s
```

**結論**: mmap page fault でも host memcpy でもなく、**HIP H2D / copy-stream の event wait が支配的**。Session A の pinned stage 経路は正しく機能 (stage=100%, pageable=0) しているが、H2D 完了待ちが律速。

**参照**: `docs\outsourcing\session-a-followup-sync-stall-2026-09-19.md` (同期ストール設計)、`docs\outsourcing\session-a-78mib-split-2026-09-19.md` (split 分析)

**やること** (MTP が片付いてから。同時に進めない — GPU が1つ):
1. `h2d_wait` が大きい理由をコードから特定: `event_synchronize` が copy stream の完了を host スレッドで待ち、それが GPU submit を遅らせている (session-a-followup-sync-stall の解析通り)。
2. 安全に wait を遅延/除去する設計が既に資料にある (compute stream への streamWaitEvent)。現 API に無いので、追加が必要かどうか、HIP での実装可否を調査して報告。
3. **実装はまだ**。設計案と見積もり (wait 除去で理論上どれだけ速くなるか) を docs にまとめて止める。

---

## 作業の優先順位と分離

1. **最優先: ブロッカー A (MTP ロード修正 → 実機テスト)** — 20 t/s 到達の本命
2. **次: ブロッカー B (H2D wait 設計)** — MTP が 20 t/s に届いても届かなくても、限界探りの基盤

**両方とも build は build-mtp / build-stage1 を使い分ける。同時ビルド・同時 GPU 実行はしない (競合で ROCm abort する)。**

## 共通ルール (再掲・厳守)

- 測定は build-stage1 (canonical) / build-mtp (MTP A/B) のみ。他 build は使わない
- HOST_BANK=1 絶対禁止。常に `LLAMA_MOE_HOST_BANK=0`
- 判定は gen t/s のみ。ctx 8192・n_predict 256・可能なら3回中央値
- ログに `LLAMA_MOE_SLOT_STATS=1` と compute_buffer 行を残す
- 他の llama プロセスがいないことを実行前に確認: `Get-Process llama-cli,llama-server,python -EA SilentlyContinue`

## 触ってよいファイル

- MTP 修正: `src/models/qwen4exp.cpp`, `src/models/models.h`, `src/llama-model.cpp`, `common/speculative.cpp`, `tools/prep-mtp-draft-qwen38.py`
- H2D 調査: `src/llama.cpp`, `src/llama-model.h`, `ggml/src/ggml-cuda/ggml-cuda.cu` (調査のみ、実装は指示まで)
- docs: `docs/qwen4exp-practical.md`, `docs/outsourcing/*` (新規作成可)

## 完了条件

- MTP: draft がロードされ、acceptance 統計が出て、gen t/s が記録される (目標 20、最低でも 13.4 からの向上を示す)
- H2D: wait 除去の設計案 + 理論改善幅の見積もりが docs にまとまっている
- 両方の結果を docs/qwen4exp-practical.md に追記

### MTP shape-repair update (2026-09-21)

The draft writer was corrected to preserve quantized raw byte-row shapes for 1D/2D/3D tensors. The regenerated draft now reports loader-visible shapes matching the qwen4exp loader, including `token_embd.weight [2560,248320]`, `output_hc_down [10240,320]`, and the 3D MoE expert tensors. `block_count=49` and `nextn_predict_layers=1` are preserved. `build-mtp` rebuilt successfully.

The next MTP load no longer reports tensor shape errors, but still fails in `load_tensors` with `invalid vector subscript` after all 33 draft tensors are enumerated. No MTP context, draft KV, acceptance, or gen t/s was reached. No ROCm abort occurred. The remaining blocker is an internal vector access in the qwen4exp draft loader path; do not change the loader speculatively until a stack trace or narrow source logging identifies the exact access.

### Root cause found and fixed (2026-09-21, second pass)

Diagnostic logging in `src/models/qwen4exp.cpp` and `src/llama-model.cpp` proved that the draft model loads correctly:

- `n_layer_all=49`, `n_layer_nextn=1`, `n_layer=48`
- `layers.size()=49`, `load_mtp=1`, `mtp_probe_missing=1`, `mtp_only=1`
- `loading MTP layer il=48 layers=49` succeeded
- `adding speculative implementation 'draft-mtp'` was reached

The `invalid vector subscript` exception was not a layer-vector problem. It came from `src/llama-model.cpp`:

```text
std::upper_bound(...) - splits.begin()  // returned devices.size() at the split endpoint
devices.at(layer_gpu)                    // std::out_of_range -> "invalid vector subscript"
```

The fix clamps the upper_bound result to `devices.size()-1` and replaces the `.at()` with a checked `find` on `gpu_buft_list` (committed). This is a single-GPU split-rounding issue that the draft load hit on the first offloaded layer.

With that fix the draft loads and the speculative draft-mtp implementation is installedifting, but generation stops with a new failure:

```text
ggml_backend_tensor_get_async OOB: tensor=l_last-47 offset=0 size=368640 nbytes=0
ggml-backend.cpp:272: tensor read out of bounds
```

`l_last-47` is the last trunk layer's hidden-state tensor; it is empty (`nbytes=0`) when the MTP path tries to read it. This is a graph-construction/execution issue after the loader, not a loader or shape issue. No acceptance stats or gen t/s were produced; those remain unmeasured.

### Graph handover fix (2026-09-21)

The qwen4exp trunk graph was gathering the final residual with `inp_out_ids` regardless of `embeddings_nextn_masked`. The target path uses `masked=false` and expects full-row nextn embeddings, so the gather made `l_last-47` shorter than the copy request (and zero bytes during an empty-output prefill). The minimal fix gates the trunk gather on `cparams.embeddings_nextn_masked` and performs a separate final-LM-output gather for the unmasked target path, matching qwen3next.

The build-mtp smoke (`ctx=2048`, `-n 1`, `HOST_BANK=0`, `--no-warmup`) no longer hits the OOB/null-buffer path and reaches draft-mtp initialization and token generation. A canonical `ctx=8192`, `-n 256` run generated 8 tokens and then hit a separate transient ROCm copy-stream creation failure (`hipStreamCreateWithFlags`, current device -1). No valid MTP gen t/s or acceptance result exists yet.

## 報告

各ブロッカーについて「何が原因だったか / 何を直したか (または設計案) / 実測値 / 残課題」を報告。数値は推測せず、取れなければ「未計測」と明記。
