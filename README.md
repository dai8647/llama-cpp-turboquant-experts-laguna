# llama-cpp-turboquant: Experts-First + Laguna MoE

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A fork of [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
(an AMD HIP/ROCm-enabled llama.cpp fork), integrating the following features:

- **TurboQuant KV cache** — turbo2/3/4 KV quantization (CUDA / HIP / Metal / Vulkan)
- **Experts-first MoE GPU expert slot** — ported from leoustc/llama.cpp-moe
- **Laguna architecture support** — port of upstream PR #25165 (merged 2026-07-22)
- **Frequency-based expert placement** — GPU placement based on expert selection frequency
- **DeepSeek V4 (`deepseek4`) support** — merged from upstream master (2026-08-03)
- **Generic MTP speculative decoding** — architecture-agnostic `draft-mtp` engine with `graph_mtp` in 12+ model families (DeepSeek V2/3/4, Qwen3.5/3.6, Step3.5, GLM-DSA, MiMo2, Cohere2, HY-V3, ...)
- **Ling-3.0-flash (`bailing-hybrid`) support** — hybrid KDA + gated-MLA MoE (127.5B-A5.1B), including its MTP head (`--mtp`)

Upstream sync status: **fully caught up with `ggml-org/llama.cpp` master as of
2026-08-03** (merge commit `c461b278`). See `TURBOQUANT_UPSTREAM_MERGE.md` and
`DEEPSEEK4_MERGE_GUIDE.md` (JA: `DEEPSEEK4_MERGE_GUIDE.ja.md`) for details.

---

# 概要

[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
(AMD HIP/ROCm対応 llama.cpp fork) に、以下の機能を統合した fork:

- **TurboQuant KV cache** — turbo2/3/4 KV量子化 (CUDA / HIP / Metal / Vulkan)
- **Experts-first MoE GPU expert slot** — leoustc/llama.cpp-moe からの移植
- **Laguna architecture support** — upstream PR #25165 (merged 2026-07-22) の移植
- **Frequency-based expert placement** — Expert選択頻度に基づくGPU配置
- **DeepSeek V4 (`deepseek4`) 対応** — 上流 master からマージ (2026-08-03)
- **汎用 MTP 投機デコード** — アーキ非依存の `draft-mtp` エンジン。`graph_mtp` を持つ 12+ モデルファミリー (DeepSeek V2/3/4, Qwen3.5/3.6, Step3.5, GLM-DSA, MiMo2, Cohere2, HY-V3 など) で使用可
- **Ling-3.0-flash (`bailing-hybrid`) 対応** — KDA + gated MLA ハイブリッド MoE (127.5B-A5.1B)。MTP ヘッドも `--mtp` で使用可

上流追従状況: **2026-08-03 時点で `ggml-org/llama.cpp` master に完全追従**
(マージコミット `c461b278`)。詳細は `TURBOQUANT_UPSTREAM_MERGE.md` および
`DEEPSEEK4_MERGE_GUIDE.md` (EN) / `DEEPSEEK4_MERGE_GUIDE.ja.md` (JA) を参照。

## 対象ユーザー / Target users

- AMD Radeon GPU (ROCm HIP) を持つローカルLLMユーザー / Users with AMD GPUs (ROCm HIP)
- Laguna S 2.1 (118B) や Laguna XS 2.1 (33B) をローカルで動かしたい人
  / Users running Laguna S 2.1 (118B) / XS 2.1 (33B) locally
- DeepSeek V4 (REAP) GGUF を ROCm で動かしたい人
  / Users running DeepSeek V4 (REAP) GGUF on ROCm
- MoEモデルでGPU expert配置を制御したい人 / Users tuning MoE expert placement
- KV cache を量子化して VRAM を削減したい人 / Users shrinking KV cache via quantization

## 主要機能 / Features

### 0. TurboQuant KV Cache

KV cache を量子化して VRAM 使用量を削減。`f16` に対して品質を保ちながら
turbo2 / turbo3 / turbo4 の 3 段階で圧縮可能 (backends: CUDA / HIP / Metal / Vulkan)。

```bash
# K/V cache を turbo4 に指定
llama-cli -m model.gguf -ctk turbo4 -ctv turbo4 -ngl 99

# デフォルトは auto-asymmetric: GQA 比率が高いモデルは K を自動で q8_0 に昇格
# (無効化: TURBO_AUTO_ASYMMETRIC=0)
```

- 高度な設定: `-ctk turbo3 -ctv turbo2` など組み合わせ自由
- attn-rotation はデフォルト OFF。モデル別に
  `LLAMA_ATTN_ROT_K_OVERRIDE=1` / `LLAMA_ATTN_ROT_V_OVERRIDE=1` で個別有効化
- layer 別精度調整: `TURBO_LAYER_ADAPTIVE=7` (Boundary V, 推奨) ほか

### 1. Experts-first MoE GPU Expert Slot

Expert単位でGPU VRAMへの配置を制御。全ExpertをGPUに事前配置（full-slot）或いは
頻度ベースの部分配置が可能。

```bash
# 全ExpertをGPUに配置 (full-slot mode)
llama-cli -ngl 999 --moe-gpu-expert-slot-num 999 -m model.gguf

# 頻度ベースの部分配置 (例: 上位50%だけGPU)
# frequencyモードは2-pass方式 (詳細は「4. Frequency-Based Expert Placement」参照):
#   Pass 1: --moe-freq-report-path stats.json で統計収集 (track_access有効化)
#   Pass 2: --moe-expert-placement frequency --moe-freq-report-path stats.json で配置適用 (frequencyモードで自動読み込み)
# または --moe-gpu-expert-slot-num を省略 = 自動全slot有効化
```

### 2. Laguna Architecture

118B MoEモデル (Laguna S 2.1) と 33B MoEモデル (Laguna XS 2.1) をサポート。

- 256 routed experts (top-10) + 1 shared expert
- Sliding-window attention (hybrid full/SWA)
- Softplus attention gate, QK-norm
- Sigmoid-routed MoE with score-correction bias

#### DFlash 投機デコード (DFlash speculative decoding)

**DFlash** は汎用ブロック拡散 draft エンジンで、Laguna 専用ではなく任意の対象
モデルと組み合わせ可能です (`src/models/dflash.cpp`)。対象モデル向けに学習
された DFlash 形式の draft モデル (hidden size・抽出レイヤ・語彙が対象と一致
するもの) を `--spec-draft-model` で指定すれば使えます。組み合わせ不一致時は
起動時に明確なエラーを出します (クラッシュしません)。公式レシピは K=15
(`num_speculative_tokens=15`)。

**DSpark (DeepSeek V4 の MTP 用 Markov-head draft) も同じエンジンを共有**します
(`common/speculative.cpp` の同一クラス、`is_dspark` フラグで分岐)。したがって
同じ検証 - hidden size 一致 / 抽出レイヤ範囲チェック / 最終層スロット
(`layer_id == n_layer`) の nextn 抽出へのフォールバック - が DSpark にも適用され、
不一致の組み合わせは DFlash と同じく起動時に明確なエラーになります。

```bash
# DFlash draft モデル + K 指定 (--spec-draft-n-max)
llama-server -m model.gguf --spec-type draft-dflash \
  --spec-draft-model dflash.gguf --spec-draft-n-max 3
# -hf で取得する場合: --dflash でサイドカー自動ダウンロード
```

**K の実測 (同一条件で K = 3 / 5 / 7 / 11 / 15 を比較)**:

- 発表当日の NVIDIA Developer Forums 報告: K=15 では 10-15 tok/s に低下し
  実用不可、K=7 に減らすと 40-50 tok/s まで回復
- コミュニティベンチマーク: **K=3 が最速**
- 指標: 総受理率 (draft が提案したトークンのうち本体が採用した割合) と
  平均受理長 (1 回の検証で採用されたトークン数) を確認する

なお本フォークのデフォルト `--spec-draft-n-max` はすでに **3** です
(`common/common.h` の `spec_draft.n_max = 3`)。

#### DFlash speculative decoding

Speculative decoding with **DFlash**, a generic block-diffusion draft engine, is
supported (`src/models/dflash.cpp`). It is not Laguna-only: any target model can
be paired with a DFlash-format draft trained for it (matching hidden size,
extract layers, and vocab), supplied via `--spec-draft-model`. A mismatched
pairing now fails at startup with a clear error instead of crashing. The
official recipe uses K=15 (`num_speculative_tokens=15`).

**DSpark (the Markov-head draft used by DeepSeek V4 MTP) shares the same engine**
(`common/speculative.cpp`, same class toggled by the `is_dspark` flag). The same
validation therefore applies to both: hidden-size match, extract-layer bounds,
and the final-layer slot (`layer_id == n_layer`) falling back to nextn
extraction. A mismatched pairing fails at startup with a clear error, exactly
like DFlash.

```bash
# DFlash draft model + K tuning (--spec-draft-n-max)
llama-server -m model.gguf --spec-type draft-dflash \
  --spec-draft-model dflash.gguf --spec-draft-n-max 3
# with -hf: --dflash auto-downloads the sidecar
```

Measured K sweep (K = 3 / 5 / 7 / 11 / 15 under identical conditions):

- NVIDIA Developer Forums on release day: K=15 drops to 10-15 tok/s
  (unusable); reducing to K=7 recovers 40-50 tok/s
- Community benchmark: **K=3 was the fastest**
- Metrics to watch: total acceptance rate (fraction of draft-proposed tokens
  accepted by the target model) and average acceptance length (accepted tokens
  per verification step)

Note: this fork's default `--spec-draft-n-max` is already **3**
(`spec_draft.n_max = 3` in `common/common.h`).

### 3. DeepSeek V4 (`deepseek4`)

`deepseek4` アーキテクチャ対応 (上流 PR #24162 ほか、2026-08-03 マージ済み)。
DeepSeek-V4-Flash-0731-REAP GGUF (e.g. `heath0xFF/DeepSeek-V4-Flash-0731-REAP-GGUF`)
を ROCm で実行可能。詳細は `DEEPSEEK4_MERGE_GUIDE.ja.md` を参照。

```bash
# Q4_K_M 推奨 (+ --cpu-moe は MoE 重みを CPU に置く安定化オプション)
llama-server -hf heath0xFF/DeepSeek-V4-Flash-0731-REAP-GGUF:Q4_K_M \
  -c 32768 -ngl 99 --cpu-moe
```

#### MTP / DSpark 対応 (multi-token prediction)

DeepSeek-V4-Flash-0731-REAP のチェックポイントには backbone 層 40-42 に
対応する **DSpark MTP ブロック (3 層)** が含まれています。本フォークは上流の
`DeepseekV4 MTP + DSpark (#25784)` をマージ済みのため、GGUF 側に
`nextn.*` / `mtp.*` テンソルが含まれていれば MTP 推論を利用できます
(`src/models/deepseek4.cpp` に `graph_mtp` 実装あり)。

```bash
# --mtp を付けると HF リポジトリから MTP head を自動ダウンロード
llama-server -hf heath0xFF/DeepSeek-V4-Flash-0731-REAP-GGUF:Q4_K_M \
  --mtp -c 32768 -ngl 99 --cpu-moe
# DFlash/DSpark サイドカー (Markov head を含む) を使う場合: --dflash
```

投機デコードの draft として使う場合は `--spec-type draft-dspark` (または
Markov head を持つサイドカーを `--dflash` で取得) で有効化され、DFlash と
同じ検証 (hidden size / レイヤ範囲 / 最終層スロット) を受けます。

> MTP は DeepSeek 専用ではなく、**汎用エンジン**です。`graph_mtp` を実装した
> 12+ モデルファミリー (deepseek4 / deepseek2 / deepseek32 / step35 / qwen35 /
> qwen35moe / qwen3next / cohere2moe / glm-dsa / mimo2 / hy-v3 / bailing-hybrid)
> すべてで `--mtp` が有効で、ドライバにアーキ依存の分岐はありません
> (レイヤ数・hidden size・チェーンヘッド位置はモデルのメタデータから自動取得)。

#### REAP について

**REAP** (Router-weighted Expert Activation Pruning、Cerebras) はモデル**作成時**の
MoE 枝刈り手法であり、推論時に必要な機能ではありません。K160 GGUF は
ルーテッドエキスパート 160 個 (256 個から削減) を含むだけで、`deepseek4`
アーキテクチャとして特別なフラグなしでネイティブに実行できます。

#### MTP / DSpark support

The DeepSeek-V4-Flash-0731-REAP checkpoint ships **3 DSpark MTP blocks** mapped to
backbone layers 40-42. This fork already includes upstream
`DeepseekV4 MTP + DSpark (#25784)`, so MTP inference works whenever the GGUF
contains `nextn.*` / `mtp.*` tensors (`graph_mtp` is implemented in
`src/models/deepseek4.cpp`).

```bash
# --mtp auto-downloads the MTP head from the HF repo, if available
llama-server -hf heath0xFF/DeepSeek-V4-Flash-0731-REAP-GGUF:Q4_K_M \
  --mtp -c 32768 -ngl 99 --cpu-moe
# DFlash/DSpark sidecar (carries the extra Markov head): --dflash
```

To use it as a speculative-draft, enable `--spec-type draft-dspark` (or fetch
the Markov-head sidecar with `--dflash`); it receives the same validation as
DFlash (hidden size, layer bounds, final-layer slot).

> MTP is **not** DeepSeek-specific: it is a generic engine. Any model family
> with `graph_mtp` (deepseek4 / deepseek2 / deepseek32 / step35 / qwen35 /
> qwen35moe / qwen3next / cohere2moe / glm-dsa / mimo2 / hy-v3 /
> bailing-hybrid) enables `--mtp`, and the driver has zero architecture-
> specific branches (layer count, hidden size and chain-head positions are read
> from the model metadata).

#### About REAP

**REAP** (Router-weighted Expert Activation Pruning, Cerebras) is a MoE pruning
method applied at model **creation** time; it is not a runtime feature. The K160
GGUF files simply contain 160 routed experts (pruned from 256) and run natively
as the `deepseek4` architecture with no special flags.

#### MXFP4 対応 / MXFP4 support

MXFP4 (OCP マイクロスケーリング FP4) は本フォークでエンドツーエンドに動作します。
`GGML_TYPE_MXFP4` は CPU (AVX2 ベクトル化 `ggml_vec_dot_mxfp4_q8_0` + 全アーキ
quants) と CUDA/HIP (MMQ)、Metal、Vulkan に実装済みで、ビルドフラグ追加は不要です。
ネイティブ FP4 高速パスは Blackwell (sm_100+) 専用で、7800 XT (gfx1101) では
汎用 MMQ パス (dequant -> Q8_1) で動作します。

```bash
# 同じコマンドで MXFP4 GGUF を指定するだけ。KV は TurboQuant のまま
llama-server -m DeepSeek-V4-Flash-0731-reap-150b-MXFP4_MOE.gguf \
  -c 32768 -ngl 99 -t 6 --moe-expert-placement frequency \
  --moe-freq-report-in stats.json --moe-gpu-expert-ratio 0.12 \
  -ctk turbo4 -ctv turbo4
```

注意: 150B の MXFP4_MOE は約 85GB あり、Q2_K (62GB) より 37% 多く読むため、
帯域律速の CPU MoE では Q2_K より遅くなります (品質は V4 ネイティブ形式で最良)。
96GB RAM 構成では RAM 残量に注意し、`--moe-gpu-expert-ratio` は 0.10-0.12 を目安に。

MXFP4 (OCP microscaling FP4) works end-to-end in this fork. `GGML_TYPE_MXFP4`
is implemented for CPU (AVX2-vectorized `ggml_vec_dot_mxfp4_q8_0` plus the
other arch quants), CUDA/HIP (MMQ), Metal and Vulkan; no extra build flags are
needed. The native FP4 fast path is Blackwell-only (sm_100+); on RX 7800 XT
(gfx1101) the generic MMQ path (dequant -> Q8_1) is used.

Note: the 150B MXFP4_MOE file is about 85 GB, 37% larger than Q2_K (62 GB), so
bandwidth-bound CPU MoE is slower than Q2_K (quality is best, being the native
V4 format). On a 96 GB RAM box keep `--moe-gpu-expert-ratio` around 0.10-0.12
to avoid running out of memory.

#### K216 GGUF チューニング (16GB VRAM 構成 / K216 GGUF tuning)

`heath0xFF/DeepSeek-V4-Flash-0731-REAP-K216-GGUF` は 256 エキスパート中
**216 個を保持**した REAP 版です (top-6 ルーティング、43 層、BF16 換算 ~108GB)。
公開 GGUF は `UD-IQ3_XXS` の 1 種のみで、3 shard 合計 **82.94 GiB**:

```
DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00001-of-00003.gguf  (entry point)
DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00002-of-00003.gguf
DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00003-of-00003.gguf
```

> 検証済みの注意: K216 GGUF には `nextn.*` / `mtp.*` テンソルが**含まれません**
> (K160 版は含むが、K216 は unsloth 量子化ベースのため)。したがって `--mtp`
> は使えません。投機デコードを使う場合は DSpark サイドカーを指定してください。

##### 診断: なぜ遅いのか (16GB VRAM の場合)

83GB のモデルは 16GB VRAM に載らないため、実質 CPU 推論 (DDR4 帯域律速)
になります。Ryzen 5500 級 (DDR4 デュアルチャネル ~45GB/s) で 5.5 tps は
この構成のほぼ理論上限です。対策は「GPU を効かせる」ことです。

##### 手順 1: 高頻度エキスパートを GPU に配置 (2-pass / 最大効果)

```bash
# Pass 1: 普段の使い方に近いプロンプトで使用頻度を計測 (1回だけ)
./llama-cli -m DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00001-of-00003.gguf \
  --moe-freq-report-path stats.json -p "<代表的なプロンプト>" -n 256

# Pass 2: 高頻度エキスパートだけ GPU 配置して起動
./llama-server -m DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00001-of-00003.gguf \
  -c 32768 -ngl 99 \
  --moe-expert-placement frequency \
  --moe-freq-report-path stats.json \
  --moe-gpu-expert-ratio 0.18 \
  -t 6
```

- `--moe-gpu-expert-ratio`: 16GB VRAM なら **0.15-0.25 から調整** (~18% 目安)
- `--moe-freq-report-path` は Pass 1 (書き出し) / Pass 2 (読み込み) を兼ねる単一フラグです
- 7800 XT の帯域 (624 GB/s) は DDR4 の約 14 倍。トークン生成の大半を占める
  エキスパート演算が GPU に載れば **2-4 倍の向上**が見込めます

##### 手順 2: TurboQuant KV cache で VRAM の空きを増やす

```bash
-ctk turbo4 -ctv turbo4
```

##### 手順 3: DSpark サイドカー投機デコード

K216 GGUF 自体に nextn がないため、`ggml-org/DeepSeek-V4-Flash-0731-GGUF`
の DSpark サイドカー (`dspark-DeepSeek-V4-Flash-0731-BF16.gguf`) を指定します。
DSpark ヘッドは REAP で枝刈りされない層 (MTP + Markov head) のみなので
互換です。起動時に hidden size / レイヤ範囲 / 語彙を検証し、不一致なら
明確なエラーが出ます。

```bash
./llama-server -m DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00001-of-00003.gguf \
  --spec-type draft-dspark \
  --spec-draft-hf ggml-org/DeepSeek-V4-Flash-0731-GGUF \
  --spec-draft-model dspark-DeepSeek-V4-Flash-0731-BF16.gguf \
  --spec-draft-n-max 3
```

- K=3 が最速という実測結果に合わせ `--spec-draft-n-max 3` (デフォルト)
- うまく嵌れば **1.3-1.8 倍**

##### 手順 4: スレッド数とビルド

- 帯域律速では SMT は効きません。`-t 6` (物理コア数) を明示
- 7800 XT は `gfx1101` (Navi 32)。ビルド時 `-DAMDGPU_TARGETS=gfx1101` で HIP ビルドが
  必要 (デフォルトターゲットだとカーネルが無効化され全層 CPU に落ちる。
  gfx1100 は 7900 XTX 用なので間違えない)

##### 期待値 (Ryzen 5500 + 7800 XT 16GB の場合)

| 設定 / Setup | 予想 Generation t/s |
|---|---:|
| 現状 / current | 5.5 |
| + 高頻度 expert GPU 配置 / frequency placement | 10-18 |
| + DSpark 投機 / speculative decoding | 15-25 |
| + TurboQuant KV | +数% (長文ほど効く) |

#### K216 GGUF tuning (16 GB VRAM setup) / EN

`heath0xFF/DeepSeek-V4-Flash-0731-REAP-K216-GGUF` retains **216 of 256 routed
experts** (top-6 routing, 43 layers, ~108 GB BF16 equivalent). The only public
GGUF is `UD-IQ3_XXS`, **82.94 GiB across 3 shards** (first shard is the entry
point).

> Verified caveat: the K216 GGUF does **not** contain `nextn.*` / `mtp.*`
> tensors (unlike the K160 builds, it derives from the unsloth quantization).
> So `--mtp` cannot be used; use the DSpark sidecar for speculative decoding.

##### Diagnosis: why it is slow (16 GB VRAM)

An 83 GB model cannot fit in 16 GB VRAM, so inference runs effectively on CPU
(DDR4 bandwidth-bound). ~5.5 tps on a Ryzen 5500-class DDR4 box is close to the
theoretical ceiling. The fix is to make the GPU do real work.

##### Step 1: place high-frequency experts on GPU (2-pass, biggest win)

```bash
# Pass 1: measure usage with a prompt close to your real workload (once)
./llama-cli -m DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00001-of-00003.gguf \
  --moe-freq-report-path stats.json -p "<representative prompt>" -n 256

# Pass 2: start with only high-frequency experts on GPU
./llama-server -m DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00001-of-00003.gguf \
  -c 32768 -ngl 99 \
  --moe-expert-placement frequency \
  --moe-freq-report-path stats.json \
  --moe-gpu-expert-ratio 0.18 \
  -t 6
```

- `--moe-gpu-expert-ratio`: tune **0.15-0.25** for 16 GB VRAM (~18% works well)
- `--moe-freq-report-path` is a single flag used for both Pass 1 (write) and Pass 2 (read)
- RX 7800 XT bandwidth (624 GB/s) is ~14x DDR4. Offloading the hot experts can
  give **2-4x** on generation

##### Step 2: free VRAM with TurboQuant KV

```bash
-ctk turbo4 -ctv turbo4
```

##### Step 3: DSpark sidecar speculative decoding

Since the K216 GGUF has no nextn tensors, point at the DSpark sidecar from
`ggml-org/DeepSeek-V4-Flash-0731-GGUF` (`dspark-DeepSeek-V4-Flash-0731-BF16.gguf`).
The DSpark head covers only non-pruned layers (MTP + Markov head), so it is
compatible. The fork validates hidden size / layer bounds / vocab at startup
and errors out clearly on mismatch.

```bash
./llama-server -m DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00001-of-00003.gguf \
  --spec-type draft-dspark \
  --spec-draft-hf ggml-org/DeepSeek-V4-Flash-0731-GGUF \
  --spec-draft-model dspark-DeepSeek-V4-Flash-0731-BF16.gguf \
  --spec-draft-n-max 3
```

- K=3 measured fastest, so `--spec-draft-n-max 3` (the fork default)
- Expect **1.3-1.8x** when it lands

##### Step 4: threads and build

- SMT does not help a bandwidth-bound loop: pass `-t 6` (physical cores)
- RX 7800 XT (Navi 32) needs a HIP build with `-DAMDGPU_TARGETS=gfx1101`; a generic
  build falls back all layers to CPU (gfx1100 is for the 7900 XTX)

##### Expected results (Ryzen 5500 + 7800 XT 16 GB)

| Setup | Expected gen t/s |
|---|---:|
| current | 5.5 |
| + frequency placement | 10-18 |
| + DSpark speculative | 15-25 |
| + TurboQuant KV | a few % (more at long context) |

### 4. Frequency-Based Expert Placement

Expert選択頻度を記録し、高頻度ExpertをGPUに優先配置。

#### 2パスワークフロー / 2-pass workflow

1. **Pass 1 (計測 / measure)**: 全expertをロードして推論し、各expertの使用頻度を記録
2. **Pass 2 (配置最適化 / placement)**: 頻度JSONから高頻度expertだけをGPUに配置

| フラグ / Flag | 用途 / Purpose |
|---|---|
| `--moe-freq-report-path PATH` | 単一フラグで両方を兼ねる: Pass 1 は統計をJSONに書き出し (track_access有効化)、Pass 2 は `--moe-expert-placement frequency` と併用で自動読み込み |
| `--moe-expert-placement frequency` | 頻度ベース配置モード |
| `--moe-gpu-expert-ratio 0.6` | GPU配置割合 (0.0-1.0) |

使用例 / Example:
```bash
# Pass 1: 統計収集 (--moe-freq-report-path で書き出し)
./llama-cli -m model.gguf --moe-freq-report-path stats.json -p "..." -n 256

# Pass 2: 高頻度expertだけGPU配置 (frequencyモードが同じPATHを自動読み込み)
./llama-cli -m model.gguf --moe-expert-placement frequency \
    --moe-freq-report-path stats.json --moe-gpu-expert-ratio 0.6 -p "..." -n 256
```

### 5. Ling-3.0-flash (`bailing-hybrid`)

inclusionAI [Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash)
(MIT, 127.5B total / 5.1B active) をネイティブサポート。`model_type`
`bailing_hybrid` は **KDA (Kimi Delta Attention / 線形アテンション) 35 層 +
gated MLA 7 層 + 512 エキスパート MoE (top-8)** のハイブリッド構成で、KDA 層は
KV cache を持たないためコンテキストの VRAM コストが非常に軽い
(約 8.2 KB/token、全-MLA モデルの約 1/9)。

- 実装: `src/models/bailing-hybrid.cpp`。KDA ブロックは kimi-linear、MLA は
  既存 ggml カーネルを再利用 (ggml コアは無変更)。MoE router は bailingmoe2 と
  同一の noaux_tc grouped top-k + sigmoid
- **MTP 対応**: layer 42 の `BailingMoeV3MTPLayer` (enorm/hnorm + eh_proj +
  gated MLA + MoE + shared head) を `graph_mtp` として実装済み。`--mtp` で
  投機デコードに使用可。MTP ヘッドの重み (blk.42 の全 21 テンソル) は GGUF に
  含まれ、`--mtp` 指定時のみロードされます (通常推論ではスキップ)
- GGUF: `prometheusAIR/Ling-3.0-flash-GGUF` (Q4_K_M / Q5_K_M / Q5_K_XL /
  Q5_K_XXL / Q8_0) が公開済み。256K コンテキスト対応

```bash
# MTP 投機デコード付き
llama-server -hf prometheusAIR/Ling-3.0-flash-GGUF:Q4_K_M \
  --mtp -c 32768 -ngl 99
```

注意: `sakamakismile/Ling-3.0-flash-W4A4-NVFP4` は safetensors + vLLM フォーク
専用 (NVFP4 W4A4 / compressed-tensors) のため、llama.cpp では直接ロードできません。
llama.cpp で使う場合は上記の GGUF、または bf16 ベースを本フォークの
`convert_hf_to_gguf.py` (conversion/bailing_hybrid.py) で変換したものを使います。

#### Ling-3.0-flash (`bailing-hybrid`) / EN

Native support for inclusionAI [Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash)
(MIT, 127.5B total / 5.1B active). The `bailing_hybrid` model_type is a hybrid of
**35 KDA (Kimi Delta Attention / linear attention) layers + 7 gated MLA layers +
512-expert MoE (top-8)**. KDA layers carry no KV cache, so context memory is very
cheap (~8.2 KB/token, roughly 1/9th of an all-MLA model).

- Implementation: `src/models/bailing-hybrid.cpp`. The KDA block reuses
  kimi-linear, MLA reuses existing ggml kernels (no ggml core changes); the MoE
  router is the same noaux_tc grouped top-k + sigmoid as bailingmoe2
- **MTP**: layer 42's `BailingMoeV3MTPLayer` (enorm/hnorm + eh_proj + gated MLA
  + MoE + shared head) is implemented as `graph_mtp` and usable with `--mtp`.
  The MTP head weights (all 21 blk.42 tensors) are in the GGUF and loaded only
  when `--mtp` is requested (skipped otherwise)
- GGUF: published at `prometheusAIR/Ling-3.0-flash-GGUF` (Q4_K_M / Q5_K_M /
  Q5_K_XL / Q5_K_XXL / Q8_0), verified at 256K context

```bash
# with MTP speculative decoding
llama-server -hf prometheusAIR/Ling-3.0-flash-GGUF:Q4_K_M \
  --mtp -c 32768 -ngl 99
```

Note: `sakamakismile/Ling-3.0-flash-W4A4-NVFP4` is a safetensors + vLLM-fork-only
repo (NVFP4 W4A4 / compressed-tensors) and cannot be loaded by llama.cpp directly.
Use the GGUFs above, or convert the bf16 base with this fork's
`convert_hf_to_gguf.py` (conversion/bailing_hybrid.py).

### 6. Muse Glimmer (`muse-glimmer`)

Meta [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)
(Apache 2.0、マルチモーダル・ローカルエージェント特化) をネイティブサポート。
llama.cpp 上流の #26841 (本体) / #26879 (tool-call 検出修正) / #26900
(dflash sliding_window_pattern 配列対応) を移植しました。

- 実装: `src/models/muse-glimmer.cpp` + ビジョン用 `tools/mtmd/models/muse-glimmer.cpp`
- アーキテクチャ: 52 層ハイブリッド注意 (39 層が 2048 トークンの SWA、残りは
  フル注意) + GQA (KV ヘッド 2) + QK-norm + 注意出力ゲート + dense SwiGLU FFN +
  logit tanh softcap。フォークの iswa KV cache を利用
- **DFlash ドラフター対応**: `--spec-type draft-dflash` で投機デコード可
  (公式 `dflash-kquant.gguf`、1.6 GiB)。公式 GGUF の
  `attention.sliding_window_pattern` 配列も #26900 により正しく読めます
- GGUF: `meta-models/Muse-Glimmer-30B-GGUF` の `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
  (16.7GB) は 16GB VRAM (7800 XT) にほぼ収まります

```bash
# 通常推論 (Q4_K_M は 16GB VRAM にほぼフルオフロード)
llama-server -m Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf -c 32768 -ngl 99 -t 6

# DFlash ドラフター投機デコード + ビジョン (mmproj)
llama-server -m Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  -md dflash-kquant.gguf --spec-type draft-dflash --spec-draft-n-max 5 \
  --mmproj mmproj-Muse-Glimmer-30B-Q4_K_M.gguf -c 32768 -ngl 99 -t 6
```

#### Muse Glimmer (`muse-glimmer`) / EN

Native support for Meta [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)
(Apache 2.0, multimodal, agentic local model). Ported from upstream llama.cpp
#26841 (core) / #26879 (tool-call detection after EOM) / #26900 (dflash
sliding_window_pattern array support).

- Implementation: `src/models/muse-glimmer.cpp` + vision via
  `tools/mtmd/models/muse-glimmer.cpp`
- Architecture: 52-layer hybrid attention (39 SWA layers with 2048-token window,
  rest full attention) + GQA (2 KV heads) + QK-norm + attention output gate +
  dense SwiGLU FFN + logit tanh softcap. Reuses this fork's iswa KV cache
- **DFlash drafter**: `--spec-type draft-dflash` enables speculative decoding
  with the official `dflash-kquant.gguf` (1.6 GiB); the official GGUF's
  `attention.sliding_window_pattern` array is handled correctly (#26900)
- GGUF: `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` (16.7GB) from
  `meta-models/Muse-Glimmer-30B-GGUF` fits ~entirely in 16GB VRAM (7800 XT)

```bash
# plain inference
llama-server -m Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf -c 32768 -ngl 99 -t 6

# DFlash drafter speculative decoding + vision
llama-server -m Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
  -md dflash-kquant.gguf --spec-type draft-dflash --spec-draft-n-max 5 \
  --mmproj mmproj-Muse-Glimmer-30B-Q4_K_M.gguf -c 32768 -ngl 99 -t 6
```

## ビルド / Build

### 前提条件 / Prerequisites

- Windows + AMD ROCm 7.1 (or Linux + ROCm)
- Ninja build system
- CMake

### Windows (ROCm 7.1)

```bash
set PATH=C:\Program Files\AMD\ROCm\7.1\bin;%PATH%
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.1

cmake -S . -B build-hip -G Ninja -DGPU_TARGETS=gfx1101 -DGGML_HIP=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --config RelWithDebInfo -j
```

### Linux (ROCm)

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build-rocm -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release \
        -DAMDGPU_TARGETS="gfx942"   # お使いの GPU に合わせる (7800 XT は gfx1101 / 7900 XTX は gfx1100)
cmake --build build-rocm -j$(nproc) --target llama-server
```

## テスト結果 / Test results

### Laguna XS 2.1 (33B-A3B, Q4_K_M)

| モード / Mode | Prompt t/s | Generation t/s |
|--------|------------|----------------|
| 通常 / default | 113.3 | 24.5 |
| slot (999) | 41.5 | **36.2** |
| frequency (0.5) ^1 | 124.4 | 24.5 |

### Laguna S 2.1 (118B, IQ3_XXS)

| モード / Mode | Prompt t/s | Generation t/s |
|--------|------------|----------------|
| 通常 / default | 24.1 | 4.6 |
| slot (999) | 18.2 | **9.7** |
| frequency (0.5) ^1 | 24.3 | 4.6 |

^1 frequency numbers are pre-fix (v0.1). The bug where frequency mode silently
fell through to slot-off has been fixed; post-fix numbers will differ - re-run
`bench_frequency_placement.ps1` to get current values.

- 全モードで出力一致 (temp=0, seed=42) / outputs match across modes
- HIP/ROCm 7.1, AMD GPU
- flash-attn off (ROCm SWA層クラッシュ回避 / workaround for ROCm SWA crash)

## クレジット / Credits

- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — ベースfork
- [leoustc/llama.cpp-moe](https://github.com/leoustc/llama.cpp-moe) — experts-first実装
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — upstream (Laguna PR #25165, DeepSeek V4 PR #24162)
- [poolside/Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) — Lagunaモデル
- Gabe Ortiz / signalnine — TurboQuant CUDA kernels (27 commits)

## ライセンス / License

TheTom/llama-cpp-turboquant と同一 (MIT)
