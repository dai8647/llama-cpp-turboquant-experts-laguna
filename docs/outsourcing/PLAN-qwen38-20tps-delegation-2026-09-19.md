# Qwen3.8-Flash-Next 20 t/s 計画書 — 全体レビュー + 他AI作業分担 (2026-09-19)

- 作成: audit/plan セッション (ユーザ決裁: XS 系は対象外、**Qwen3.8-Flash-Next (qwen4exp) 一択**、目標 decode **20 t/s**、LlamaDock 経由で常用、FreeToken 式 hot-expert LRU を継続)
- リポジトリ: `C:\Users\dai86\llama-cpp-turboquant-experts-laguna` @ `3f39c191f`
- モデル: `C:\Users\dai86\.lmstudio\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf` (61.9 GB, 512 experts, MTP シャード有り・未配線)

---

## 0. 現状サマリ (測定済み事実)

| 構成 | t/s | 出典 |
|---|---|---|
| 本家 + MoE CPU 固定 | ~4 | docs/qwen4exp-practical.md |
| `--moe-hot-expert` ctx 2048 | 13.5 | 同上 |
| 同 ctx 8192 | 12.1 | 同上 |
| 同 + ngram-spec ctx 8192 | 13.4 | 同上 |
| **目標** | **20** | ユーザ決裁 |

- slot = 158/512/層 (auto)。ヒット率天井 ~31% = 158/512。**容量の限界であって LRU の不良ではない**。
- ~366 MiB/tok の miss 転送が律速 (docs/research/freetoken-gap-adoption-plan §7)。

## 1. 根本原因監査 (今回のレビューで確定)

### 1-A. ★最重要★ ページャブル H2D が同期コピーに落ちている疑い → CPU 100% の主犯
- `LLAMA_MOE_PINNED_STAGE` は既定 ON (src/llama.cpp:553) だが、**miss 経路で pinned stage が実際に使われているか未検証**。HOST_BANK=0 で src が GGUF mmap ページャブルのまま H2D されていると、HIP(Windows) の hipMemcpyAsync がドライバ内ステージングでホスト CPU を食い尽くす。観測 (CPU 100%) と一致。
- `-t 16` は Ryzen 5500 (12 スレッド) をオーバーサブスクライブしてさらに悪化。
- 検証ゲート: `LLAMA_MOE_SLOT_STATS=1` の `avg=X ms` と copy MiB から帯域を逆算 (2-3 GB/s なら pageable)。起動ログに `slot bank pinned stage` が出ているか。

### 1-B. auto_pin の飛び地
- `llama_moe_gpu_expert_slot_auto_pin` は per-eval 呼び出し (src/llama-context.cpp:2088) で 2000 アクセス = 数トークンで発火。pin で `frequency_whitelist` が入ると `has_paging_dynamic` 条件 (src/llama-graph.cpp:2126-2131, `frequency_whitelist.empty()`) が false になり **global-LRU ページングが事実上死ぬ**可能性。実測でも「frequency pin はほぼ変化なし」。
- 対策: auto_pin を既定 OFF にするか、whitelist ではなく slot 保護フラグで実装し paging を殺さない形に。まず pin ON/OFF × slot stats で計測してから決める。

### 1-C. 物理容量の壁 (158/512)
- 158 slot では hit 率 31% が理論天井。20 t/s には hit 率 ~55-60% か miss 転送の完全隠蔽が必要。
- レバー: (a) より小さい expert 量子化, (b) KV cache 量子化で VRAM 確保, (c) mmproj (0.85 GB) 不要時は外す, (d) 層ごとの slot 非均等化。

### 1-D. miss 転送がクリティカルパス (同期)
- decode materialize は miss 時に同一トークン内で H2D 完了を `event_synchronize` で待つ (src/llama.cpp:900-1017)。inter-step prefetch (`LLAMA_MOE_PREFETCH_MS`) は既定 0 (無効) で、512 expert/top-10 の last-step 予測は thrash する (llama.cpp:1591-1593 コメント: full predict 10.6 vs sync pinned 13.4 t/s)。
- 本命は「層 i の計算中に層 i+k の選択 expert を早期 fire」。router 出力は層入口で確定するので、router 直後に非同期 H2D を投げ、FFN 到達までに完了させる (docs/research §3「40 t/s の本体ゲート」)。

### 1-E. Flash Attention / graphs の状態不明
- 前回計画で FA 必須。`9c54a4f7c` で fattn vec instance 追加済み、build-stage1 は `GGML_HIP_GRAPHS=ON` (ただし glru 有効時は graphs 自動無効化経路あり src/llama.cpp:1581)。
- **qwen4exp で `-fa on` が実際に HIP FA kernel に落ちているか、silent fallback していないか未検証**。fallback すると CPU 100% の一因。fallback は失敗扱い (ユーザ決裁ルール)。

### 1-F. build ディレクトリ乱立 (再発)
- `build/ build-hip/ build-bank/ build-cleanup/ build-cpu/ build-stage1/` の 6 系統。どのバイナリで何を測ったか追跡不能。ベンチは `build-stage1` に一本化。

### 1-G. LlamaDock 統合が未完了
- hot-expert 経路は llama-cli でしか実測されていない。server で `--moe-hot-expert` を通し、c76a8470f の telemetry (hit/miss) を LlamaDock に出す必要。compute_buffer device/buft 行 (2c605453c) が LlamaDock の GPU-vs-CPU 判定用。

### 1-H. HOST_BANK=1 事故の恒常化防止
- 現状 warn のみ (src/llama.cpp:1716)。**起動拒否 (abort) への格上げ**を検討。


## 2. 作業分担 (他 AI 3-4 セッション)

ファイル衝突しないよう領域を分ける。共通ルールは §3。

### Session A: 転送経路の実測と pinned stage の確実化 (最優先・単独で先行)
- **領域**: `src/llama.cpp` (slot materialize / pinned stage / stats), `docs/perf/*`
- **やること**:
  1. `LLAMA_MOE_SLOT_STATS=1` + stage ログで miss 1 件あたり帯域 (MiB/ms) を計測。pinned stage が効いているか pageable 直 H2D に落ちていないか数値で確定。
  2. pinned stage が無効/未作成なら、HOST_BANK=1 に頼らず per-layer pinned stage (small) を確実に作成する経路を修正 (bank_ensure 失敗時にスキップされるケースの洗い出し)。
  3. miss H2D の `event_synchronize` を外せるか検証: remap が slot_id を返す時点で resident=true にするなら同一ストリーム上の後続カーネルは順序保証されるはず (要設計。event wait が compute ストリームを塞いでいるならここが同期ストールの本体)。
  4. 成功ゲート: decode miss 帯域 ≥ 10 GB/s、`avg ms` が理論値 (44.8 MiB/帯域) と一致、-t 12 で CPU 70% 以下。
- **禁止**: LlamaDock 側、bench スクリプト整備 (Session C の領域)。

### Session B: decode 早期 fire / prefetch 再設計 (A の帯域確保後)
- **領域**: `src/llama-graph.cpp` (build_moe_gpu_slot_ids / remap), `src/llama.cpp` (prefetch), `include/llama.h` (flag)
- **やること**:
  1. 層 i の router 出力確定点で層 i+1..i+k の選択 expert を予測 (直前トークン last-step ではなく **同一トークン内の層先行**) し、copy stream に非同期 H2D。`prefill_pf_inflight` 機構を decode に拡張する形が既存コードと整合。
  2. リード距離 k と inflight 上限を env 調整可能に (`LLAMA_MOE_DECODE_PF_LAYERS`, 既存 `LLAMA_MOE_PF_INFLIGHT`)。
  3. auto_pin の飛び地 (1-B) を解消: whitelist 差し替えではなく slot 保護ビットで実装し global-LRU paging を維持。または auto_pin 既定 OFF + 手動トリガ化。計測で決定。
  4. 成功ゲート: ctx 8192 で 13.4 → 17+ t/s、slot stats で miss の avg wait が ~0 化 (compute に隠蔽)。
- **依存**: Session A の帯域確保。A 完了前に手を出さない。

### Session C: ベンチ / ビルド統一 / FA 検証 (A と並行可・コード非衝突)
- **領域**: `bench_*.ps1`, `run-qwen4exp*.cmd`, `docs/`, `AGENTS.md`, `ggml/src/ggml-hip/CMakeLists.txt` (FA のみ)
- **やること**:
  1. 専用ベンチ `bench_qwen38_20tps.ps1` 新規: build-stage1/bin/llama-cli.exe 固定、`-fa on` 必須、ctx 8192、n_predict 256、3 回中央値、gen t/s を合格基準。slot stats を自動収集し hit 率・帯域をレポート。
  2. **FA 検証**: `-fa on` で HIP FA kernel が選ばれているかログ/verbose で確認。silent fallback は失敗として報告。fattn vec instance (9c54a4f7c) が qwen4exp の head 形状をカバーするか確認。
  3. build 統一: ベンチは build-stage1 のみ。AGENTS.md/README に「測定は build-stage1 のみ」明記。
  4. LlamaDock 結合テスト用の最小 server 起動手順を docs に (Session D へ渡す)。
- **禁止**: src/llama.cpp, src/llama-graph.cpp の機能変更。

### Session D: LlamaDock 統合 + 容量拡大 (C の手順書待ち。LlamaDock 側のみなら並行可)
- **領域**: `C:\Users\dai86\LLAMADOCK\*` (config, web-ui, tools), `tools/server/*` (telemetry 配線のみ), 量子化比較メモ
- **やること**:
  1. LlamaDock が起動する llama-server に `--moe-hot-expert -fa on` を通し、c76a8470f の telemetry (hit/miss/slots) を LlamaDock ステータスに表示。
  2. compute_buffer device/buft 行 (2c605453c) で GPU 経路であることを LlamaDock 側で検査し CPU fallback 時は警告。
  3. 容量拡大の評価メモ: IQ2 系量子化での slot 数見積、KV q8_0 化で空く VRAM 見積、mmproj 切り離し。実装は A/B の結果を見てから。
  4. HOST_BANK=1 を LlamaDock から絶対に起動できないよう config 側でガード。
- **依存**: Session C の server 起動手順。A/B のコード変更とは非衝突。

## 3. 全セッション共通ルール (厳守)

1. **測定は `build-stage1\bin\llama-cli.exe` のみ** (RelWithDebInfo / gfx1101 / GGML_HIP_GRAPHS=ON)。他 build の数値は報告しない。
2. **HOST_BANK=1 は禁止** (load 時全層 pin → CPU 95% / Disk 100% / ROCm error の既知事故)。常に `LLAMA_MOE_HOST_BANK=0`。
3. 判定は **end-to-end decode t/s (gen t/s) のみ**。マイクロベンチ・見積は参考。ctx 8192・n_predict 256・3 回中央値。
4. 変更は小さく、コミットメッセージに測定前後の t/s を書く。
5. 衝突回避: 上記領域外のファイルを触らない。必要になったら計画書に追記してから。
6. ログには必ず `LLAMA_MOE_SLOT_STATS=1` と compute_buffer device/buft 行を残す。

## 4. 20 t/s への見通し (現時点の仮説)

- 13.4 → 20 t/s は +49%。内訳: miss 転送の隠蔽 (1-D, 最大レバー) + hit 率改善 (1-B/1-C) + FA の確実な有効化 (1-E)。
- 物理容量 (158 slot) のままなら **転送隠蔽が主戦場**。slot を増やす量子化変更 (1-C) は効果大だが別モデル生成コストあり。
- FreeToken の知見 (docs/research): 転送が compute の裏に隠れれば 40 t/s 級も視野、隠せなければ転送時間が天井。20 t/s は現実的な中間目標。

## 5. 次アクション

- ユーザ承認後、Session A と Session C を並行起動 (非衝突)。B は A の帯域結果待ち。D は C の手順書待ち。
- このファイルを別 AI への handoff として共有する。
