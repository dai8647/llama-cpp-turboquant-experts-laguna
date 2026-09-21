# ハンドオフ v3 — MTP クラッシュ: 根因特定と修正方針 (2026-09-22)

**まず計画書 §3 (共通ルール) を読んでください。**
`docs/outsourcing/PLAN-qwen38-20tps-delegation-2026-09-19.md`

前版: HANDOFF-mtp-launchfailure-v2-2026-09-21.md (LB 切り分けマトリクス依頼)

---

## 結論 (最重要)

**根因: CUDA graphs (capture/replay) × MTP の交互作用 = 非同期クラッシュ**

| 条件 | 結果 |
|---|---|
| full (hot-expert + MTP) + graphs ON (通常) | abort at 3:55 (`unspecified launch failure`) |
| full + graphs ON + LB | survived >9 min (timeout で kill, crash なし) |
| nohot (MTP only) + graphs ON | abort at 1:03 (prefill 中 D2H sync) |
| nohot + graphs ON + LB | silent death at 1:04 (graph warmup 中) |
| nohot + **graphs OFF** + LB | **CLEAN COMPLETE at 1:13** ✅ |
| full + **graphs OFF** (no LB, real speed) | survived >420 sec, crash なし ✅, but **極度に遅い** (~0.1 t/s) |
| nomtp (hot-expert only) + graphs ON | stable (daily driver 13.4 t/s) |

**graphs 無効化はクラッシュを止めるが、パフォーマンスが壊れる。** 正しい修正が必要。

---

## 詳細な切り分け結果

### Run 1: full + LB (HIP_LAUNCH_BLOCKING=1)
- log: `mtp_diag_full-lb.stdout.log` / `mtp_diag_full-lb.stderr.log`
- 9:09 で timeout kill (-crash なし, ROCm error なし)
- stdout 1704 bytes (生成テキスト確認済み)
- **結論: LB (全部列化) ではクラッシュしない → 本物の非同期レース**

### Run 2: nohot (MTP のみ) + no LB
- log: `mtp_diag_nohot.stdout.log` / `mtp_diag_nohot.stderr.log`
- 1:03 で abort (prefill 中)
- 検出点: `ggml_backend_cuda_buffer_get_tensor` (D2H sync)
- abort 直前: expert materialize (layer 45) が連続 → copy stream 取得中に死亡
- **結論: hot-expert 関係なく MTP 単体でクラッシュ**

### Run 3: nohot + LB
- log: `mtp_diag_nohot-lb.stdout.log` / `mtp_diag_nohot-lb.stderr.log`
- 1:04 で無言死 (ROCm error ログなし)
- 死亡直前: CUDA graph warmup の連発中
- **結論: graph capture 中に異常**

### Run 4: nohot + LB + GGML_CUDA_DISABLE_GRAPHS=1
- log: `mtp_diag_nohot-lbng.stdout.log` / `mtp_diag_nohot-lbng.stderr.log`
- 1:13 でクリーン完走 (exit code 0)
- **結論: graphs 無効化でクラッシュ回避**

### Run 5: full + GGML_CUDA_DISABLE_GRAPHS=1 (no LB, real speed)
- log: `mtp_diag_full-ng.stdout.log` / `mtp_diag_full-ng.stderr.log`
- 420+ 秒生存, ROCm error なし
- Stats: copies=4864, hit=9676, miss=4864, evict=15 (~50 decode steps)
- stdout: `**MoE (Mixture of Experts)` のみ (partial generation)
- llama_perf line なし (kill されたため)
- **結論: crash は回避されたが、decode が極度に遅い (~0.1 t/s vs 13.4 t/s baseline)**

---

## 根因の技術的解説

### なぜ CUDA graphs × MTP がクラッシュするか

1. **MTP ドラフトモデルが 2 番目の graph を毎ステップ生成**: target decode → draft decode → accept check のサイクルで、draft の graph が毎回 capture される
2. **ggml graph capture は op の実行を stream 上に記録**: capture 中に non-capturing stream (ext_copy_stream) 上の操作が発生すると capture 状態が破壊される可能性
3. **capture/replay で ptr が固定される**: draft graph の input (staging buffer) が毎ステップ realloc されると、replay が freed memory を参照 → `unspecified launch failure`
4. **detect point は `ext_copy_stream_get`**: 実際のクラッシュは earlier async kernel; stream creation は just the first HIP call that reports the sticky error

### 既知のメカニズムとの整合性
- `ext_graphs_disabled` フラグ (common.cuh:1529): MoE paging が graphs を無効化するのは "captured-and-replayed graph would freeze slot decisions at warmup values" (stage2 design doc)
- hot-expert のみ (nomtp): stable because decode は graph を capture しない (prefill graph が1回だけ capture される程度)
- MTP: draft decode graph が毎ステップ capture → crash trigger

---

## 残りの課題

### A. graphs-off での極度なスローダウン (最優先)
full-ng run で ~0.1 t/s (baseline 13.4 t/s の 1/130)。原因候補:
- draft graph が毎ステップ新規 capture/replay (graphs off = no capture = 毎回 JIT? → huge overhead)
- memory pressure (draft model 3 GB + target model ~14 GB on 16 GB VRAM)
- hot-expert paging が graphs-off で退化 (slot decision が毎ステップ change)
- nextn staging buffer の copy overhead

**調査方法**:
- GGML_CUDA_DISABLE_GRAPHS=1 での nomtp (hot-expert のみ) を実行 → baseline 測定
- メモリ使用量を VRAM サンプリングで確認
- decode 時の CPU/GPU 使用率を確認

### B. 正しい修正 (本線)
graphs を完全に無効にするのではなく、**draft model の context のみ** graphs を無効化:
1. `common/speculative.cpp` の draft context 作成後: `ggml_backend_cuda_ext_set_graphs_enabled(ctx_dft_backend, false)`
2. target context は graphs を維持 (hot-expert paging が graphs を切る場合のみ)
3. これにより: target decode は graphs ON (fast), draft decode は graphs OFF (safe)

**修正箇所**:
- `common/speculative.cpp`: common_speculative_init() 内、ctx_dft 作成後
- または `src/llama-context.cpp`: spec_decode パラメータで drafts_only graph disable

**確認**: target の graphs が draft の影響を受けないか (backend は共有される可能性) → 要検証

### C. 別案: MTP draft の graph capture を修正
- draft graph の input staging buffer を stable にする (毎ステップ realloc しない)
- ggml の graph capture が非互換 op を除外するようにする
- ただし、复杂的なので B が先

---

## 実行済みコマンド (再現用)

```powershell
# 必須 env
$env:ROCM_PATH = 'C:\Users\dai86\rocm-sdk-core\_rocm_sdk_core'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:HIP_VISIBLE_DEVICES = '0'
$env:LLAMA_MOE_HOST_BANK = '0'
$env:LLAMA_MOE_SLOT_STATS = '1'

# full + graphs-off (workaround)
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
& build-mtp\bin\llama-cli.exe -m <本体GSQ-RCO> -md <draft-q8_0> --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 --moe-hot-expert -fa on -c 8192 -t 12 --reasoning off --no-warmup -st -v -p "Explain MoE routing concisely." -n 256
```

---

## ドラフト GGUF (前回作成済み)
- `C:\Users\dai86\Downloads\mtp-draft\mtp-Qwen3.8-Flash-Next-draft-q8_0.gguf` (3.06 GB)
- thin 型: tok_embd 追加 + `nextn.hc_head_*` → `output_hc_*` リネーム済み
- 検証済み (metadata OK)
- 再生成スクリプト: `tools/prep-mtp-draft-qwen38.py`

---

## Git 状態
- ブランチ: `feat/qwen4exp-hot-expert`
- コミット: `3820c7e38` — feat(mtp): port qwen-next draft-mtp (NextN head) for qwen4exp, code-complete
- ファイル変更: qwen4exp.cpp, models.h, llama-model.cpp, speculative.cpp, prep-mtp-draft-qwen38.py, handoff doc
- build-mtp ディレクトリ: HIP ビルド成功済み (llama-cli / llama-server / llama.dll)

---

## 次のアクション (優先順)

1. **B を実装**: draft context のみ graphs 無効化 → real speed でクラッシュ回避
2. **A を調査**: graphs-off でのスローダウン原因 → B が解決するはず
3. **canonical を実行**: B 実装後、full (hot-expert + MTP + target graphs ON + draft graphs OFF) で gen t/s 計測
4. **ハンドオフ**: 成功すれば D セッション (LlamaDock 統合) へ引き継ぎ

---

## 禁止
- build-stage1 の変更/再ビルドを canonical 測定に使うこと
- HOST_BANK=1、force push
- H2D 78MiB/s 最適化 (MTP 完了後)
- 推測による大規模リファクタ

---

## 報告
- B 実装後の canonical 完走結果 + gen t/s
- graphs-off スローダウンの原因特定結果
- VRAM 推移
- MTP acceptance rate (生成テキストの品質確認)
