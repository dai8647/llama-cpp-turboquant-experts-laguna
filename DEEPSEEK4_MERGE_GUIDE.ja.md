# DeepSeek-V4 (REAP GGUF) を ROCm で動かす - マージ結果と検証ガイド

日付: 2026-08-03

## 1. 実施したこと: フォークへの上流マージ

フォーク (`dai8647/llama-cpp-turboquant-experts-laguna`) には `deepseek4`
アーキテクチャ対応がありませんでした。上流 `ggml-org/llama.cpp` がこれを追加
(PR #24162 "DeepSeek V4"、2026-06-29 マージ) し、さらに追従修正が続いていました。
依頼方針「上流に完全追従」に従い、最新の上流 master をフォークへマージしました。

- 作業ブランチ: `merge-upstream-next` (ローカルのみ。既に push 済みの
  `merge-upstream` ブランチから作成)
- マージコミット: `c461b2786ed0c3743a58cd63930f3d4f6257b375`
  - 親1 (フォーク側): `8bd5ba0b03` (`merge-upstream`。deepseek4 本体 #24162 を含む)
  - 親2 (上流側): `fe2adf0e72` (上流 master、2026-08-03)
- 結果: `git rev-list --count upstream/master ^c461b278` == **0** -> 上流に完全追従。

### ブランチ構成 (2026-08-03 時点)

| ブランチ | 場所 | HEAD | 備考 |
|---|---|---|---|
| `main` | origin | `09ec1633` | フォークのデフォルトブランチ (MoE エキスパート周波数修正) |
| `bench/frequency-placement-fix` | origin | `09ec1633` | `main` と同じコミット |
| `merge-upstream` | origin | `8bd5ba0b` | 上流 `f5919bf45` (2026-08-02) までのマージ、#24162 を含む |
| `merge-upstream-next` | ローカルのみ | `c461b278` | 今回のマージ: 上流 master `fe2adf0e72` まで |

注意: `TURBOQUANT_UPSTREAM_MERGE.md` に記載のある `feature/turboquant-kv-cache`
は origin には**存在しません**。GitHub 上のフォークの正規ブランチは `main` です。

### 解決したコンフリクト (1 ファイル: src/llama-kv-cache.cpp)

上流 25 コミットの範囲に、MSA インデクサーを `llama_kv_cache` から独立した
`llama-kv-cache-msa` クラスへ移すリファクタ (PR #26338 "M3: Move MSA into a
new memory implementation") が含まれていました。フォーク側は `k_idx` をインライン
実装 (約 100 箇所で使用) していたため衝突。解決内容:

- `kv_layer` 構造体: 上流に追従 (`k_idx`, `k_idx_stream` フィールドを削除)
- `layers.push_back(...)`: 上流の 5 フィールド形式に追従
- `llama_kv_cache_context::get_k_idx()` を削除 (上流も削除済み)
- **TurboQuant 独自差分は全て維持**: `get_turbo_rotation(_inv)` /
  `get_turbo_rot_forward/inverse` / `get_turbo_innerq_scale_inv` ラッパー、
  回転行列の初期化、auto-asymmetric K 昇格、layer-adaptive ポリシー、
  attn-rotation デフォルト OFF の環境変数
  (`LLAMA_ATTN_ROT_K/V_OVERRIDE`, `LLAMA_ATTN_ROT_DISABLE`)

### 今回含まれた DeepSeek 関連コミット (f5919bf..fe2adf0e)

- dbadb68e ggml: split graph 入力の動的割り当て (#22789)
- 2b63e061 llama: DeepSeek V3.2 の MTP サポート (#26457)
- fffbcbdb metal: DeepSeek V4 の hyper-connections 実装 (#26459)
- 596a5795 DeepseekV4 MTP + DSpark (#25784)
- 加えて、以前の `merge-upstream` に含まれる #24162 (deepseek4 本体)

### このワークスペースで実施した検証

- `g++ -std=c++17 -fsyntax-only` が `src/llama-kv-cache.cpp`,
  `src/llama-graph.cpp`, `src/llama-memory-hybrid.cpp` で成功
- オブジェクトコンパイル (`-c`) が `src/llama-kv-cache.cpp` と
  `src/llama-memory-hybrid.cpp` で成功
- `LLM_ARCH_DEEPSEEK4` が `src/llama-arch.h` / `llama-arch.cpp` に登録済み
  (アーキ名 `deepseek4`)
- フォーク独自機能は健在: `src/llama-moe-placement.cpp`,
  `src/turbo-rotation-data.h`, `src/models/laguna.cpp`,
  `src/llama-kv-cache-dsa.cpp`。上流の新クラス `src/llama-kv-cache-msa.cpp`
  も `src/CMakeLists.txt` に組み込み済み

未検証: フル CMake ビルド、GPU (HIP) ビルド、実モデルのロード。
これらはセクション 2 を使って対象マシンで実施してください。

## 2. お使いのマシンでの検証・実行手順

### 2a. ビルド (CPU / 汎用、手軽なスモークテスト)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build -j$(nproc) --target llama-cli llama-server
```

### 2b. AMD ROCm (HIP) 向けビルド

前提: ROCm SDK インストール済み、`hipconfig` が PATH に存在すること。

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build-rocm -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release \
        -DAMDGPU_TARGETS="gfx942"   # お使いの GPU に合わせて変更 (例: RX 7900 XTX は gfx1100)
cmake --build build-rocm -j$(nproc) --target llama-server
```

### 2c. DeepSeek-V4-Flash-0731-REAP を実行

モデル: `heath0xFF/DeepSeek-V4-Flash-0731-REAP-GGUF`、アーキテクチャ `deepseek4`、
約 193B パラメータ、ルーテッドエキスパート 160 個 (256 個から枝刈り)。
ファイル (HF API で確認済み):

| ファイル | サイズ | 備考 |
|---|---|---|
| `DeepSeek-V4-Flash-0731-REAP-K160-MXFP4.gguf` | 101.2 GB | 限定的テストでは最高品質。ただし MXFP4 は HIP カーネル未実装 -> CPU フォールバック |
| `DeepSeek-V4-Flash-0731-REAP-K160-Q4_K_M.gguf` | 109.0 GB | モデルカードで繰り返し問題が報告。**`--cpu-moe` 必須推奨** |
| `DeepSeek-V4-Flash-0731-REAP-K160-Q2_K.gguf` | 65.4 GB | 同上の注意 |

推奨コマンド (モデルカード準拠の Q4_K_M + クリーンなサンプリング):

```bash
./build-rocm/bin/llama-server \
  -hf heath0xFF/DeepSeek-V4-Flash-0731-REAP-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 -c 32768 -ngl 99 \
  --cpu-moe \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --repeat-penalty 1.0
```

注意点:
- `--cpu-moe` は MoE 重みを CPU に置きます。GPU カーネルが落ちる・出力が壊れる
  場合の実証済み回避策で、VRAM 109 GB 未満のカードでも必須です。
- MXFP4 ファイルは、カードとビルドが対応している場合のみ。CPU パスは遅い想定。
- MI300X (192 GB HBM) ならスモークテスト後に `--cpu-moe` を外して試せます。
- 上流の既知問題: #25837 / #26521 (Apple Silicon CPU の repack)、
  #26423 (deepseek4 + 量子化 KV で出力破綻。attn-rotation 無効化で修正 - 本
  フォークはデフォルトで rotation OFF なので正しい設定)。

## 3. 安全なマージポイントの分析

- フォークの `merge-upstream` ブランチは上流 `f5919bf45` (2026-08-02) まで
  マージ済みで、deepseek4 本体 (#24162) を含みます。
- 残り 25 コミットは主に非 DeepSeek 修正と、DeepSeek 関連 4 件 (#22789,
  #26457, #26459, #25784)。TurboQuant と衝突するのは #26338 (MSA 分離) だけで、
  上記の通り解決済みです。
- このワークスペースでは `fe2adf0e72` 以降の上流コミットは存在しない
  (`unmerged == 0`) ため、今日時点でこれ以上安全なマージポイントはありません。
- 今後の推奨: マージは `merge-upstream` で行い、ラウンドごとに新しい
  `merge-upstream-next` を作る。`master` はフォークに存在しない (デフォルトは
  `main`) ので、master へのマージはしない。

## 4. 引き継ぎ・次のステップ

- マージコミットの確認: `git show --stat c461b278`
- 問題なければ `merge-upstream-next` をフォークへ push し、`main` への PR を
  Freebuff の Changes パネル / 通常の GitHub フローで作成。
- AMD GPU があるマシンでセクション 2a/2b のビルドを実行。
