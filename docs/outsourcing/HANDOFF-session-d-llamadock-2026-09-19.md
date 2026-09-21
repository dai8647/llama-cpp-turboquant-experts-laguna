# Session D ハンドオフ — LlamaDock 統合 (Qwen3.8-Flash-Next 20 t/s)

**まず計画書 §3「全セッション共通ルール」を読んでから作業開始してください。**
`C:\Users\dai86\llama-cpp-turboquant-experts-laguna\docs\outsourcing\PLAN-qwen38-20tps-delegation-2026-09-19.md`

最大の地雷は **HOST_BANK=1 事故** (load 時全層 pin → CPU 95% / Disk 100% / ROCm error) と **build 統一ルール** (測定は build-stage1 のみ) です。

---

## 目的

llama-cli で動いている FreeToken 式 hot-expert 経路 (`--moe-hot-expert`) を、LlamaDock (llama-server 起動フロント) 経由で常用できるようにする。ユーザの日常利用が LlamaDock なので、ここがゴール。

## 現状

- hot-expert 経路は llama-cli で実測済み (13.4 t/s @ ctx 8192)。llama-server では未検証。
- Session C が canonical ベンチ (`bench_qwen38_20tps.ps1`) と server 最小起動手順 (`docs/qwen38-20tps-bench.md`) を提供済み。
- server 側には c76a8470f で hot-expert telemetry、2c605453c で compute_buffer device/buft 行が実装済み。LlamaDock 側から拾うだけ。

## 作業項目

### 1. 引数配線 (params-schema.json)
- ファイル: `C:\Users\dai86\LLAMADOCK\config\params-schema.json`
- 現状: `moe_expert_placement` の allowed は `["all-gpu", "frequency", "cpu-moe", "map"]` (行 197-205 付近)
- やること: `"hot-expert"` を追加し `flags: ["--moe-hot-expert"]` を配線。起動時に `-fa on` も必須化し、FA が有効にならなかった場合は警告・失敗扱いにする (silent fallback を許さない)。

### 2. Telemetry 表示 (c76a8470f / 2c605453c の拾い上げ)
- ファイル: `C:\Users\dai86\LLAMADOCK\web-ui\launch-manager.js` (health poll / log ring)
- 表示するログ行:
  - `MoE GPU slot stats: copies=... hit=... miss=... evict=... copy=... MiB avg=... ms` (cache 状態)
  - `compute_buffer: device=... buft=...` (GPU 経路確認。ROCm/HIP device なら GPU、CPU なら警告)
- これらが出ていない = hot-expert 経路に乗っていない、と判定して UI に警告。

### 3. ベンチ統合
- LlamaDock の `benchmark()` を `bench_qwen38_20tps.ps1` と同条件で実行できるようにする:
  - ctx=8192, n_predict=256, runs=3, median gen t/s
  - `LLAMA_MOE_HOST_BANK=0` `LLAMA_MOE_SLOT_STATS=1` を環境に固定
  - FA 検証ログ (`flash_attn=enabled` 等) も収集
- 判定は gen t/s のみ。20 t/s が合格ライン。

### 4. HOST_BANK=1 ガード (最重要)
- `C:\Users\dai86\LLAMADOCK\llamadock.bat`, `C:\Users\dai86\LLAMADOCK\select-model.ps1`, `launch-manager.js` で **`LLAMA_MOE_HOST_BANK=0` を強制**。
- どこかに 1 が入っていたら **起動拒否** (警告ではなく abort)。

### 5. 容量拡大評価メモ (実装は Session A/B 後)
- `docs/` にメモを残す:
  - IQ2 系量子化での slot 数見積 (現状 158/512, Q2_0)
  - KV cache q8_0 化で空く VRAM 見積
  - mmproj (0.85 GB) の切り離し可否
- 実装は A/B の結果を見てから。ここでは見積もりだけ。

## 触ってよいファイル

- `C:\Users\dai86\LLAMADOCK\config\params-schema.json`
- `C:\Users\dai86\LLAMADOCK\config\profiles.json` (hot-expert プロファイル追加なら)
- `C:\Users\dai86\LLAMADOCK\web-ui\launch-manager.js`
- `C:\Users\dai86\LLAMADOCK\llamadock.bat`
- `C:\Users\dai86\LLAMADOCK\select-model.ps1`
- `C:\Users\dai86\LLAMADOCK\docs\` または llama-cpp-turboquant-experts-laguna 側 `docs\` (容量拡大メモ)

**触らない**: `C:\Users\dai86\llama-cpp-turboquant-experts-laguna\src\*` (Session A/B の領域)、`ggml\*`。server 側で変更が必要に見えても、まず相談。

## 参照 (実在確認済み)

- 計画書: `C:\Users\dai86\llama-cpp-turboquant-experts-laguna\docs\outsourcing\PLAN-qwen38-20tps-delegation-2026-09-19.md`
- canonical ベンチ: `C:\Users\dai86\llama-cpp-turboquant-experts-laguna\bench_qwen38_20tps.ps1`
- server 起動手順: `C:\Users\dai86\llama-cpp-turboquant-experts-laguna\docs\qwen38-20tps-bench.md`
- hot-expert 起動・ログ読み方: `C:\Users\dai86\llama-cpp-turboquant-experts-laguna\docs\freetoken-hot-expert.md`

## 完了条件

- LlamaDock から Qwen3.8-Flash-Next を hot-expert モードで起動できる
- launch-manager の UI/log ring に slot stats と compute_buffer 行が表示される
- HOST_BANK=1 が設定された状態では起動を拒否する
- LlamaDock の benchmark() が canonical 条件で gen t/s を返す

## 依存

- Session C: 完了済み (bench スクリプトと server 手順書あり)
- Session A/B: 非衝突。並行作業可 (LlamaDock 側のみなのでコード衝突なし)
