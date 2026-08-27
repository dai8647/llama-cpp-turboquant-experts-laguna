# FreeToken 16GB コミュニティ報告 + 突破機構詳解 (2026-08-27, review side)

出典: GitHub (FlashML-org/FreeToken の issues/PR/roadmap/FAQ)・arXiv 2608.16157 論文本体・Reddit (r/LocalLLaMA, r/LocalLLM, r/accelerate)・X 検索 の一次ソース実地収集。 本 doc は `freetoken-vs-laguna-mapping.md` (2026-08-26, ユーザ提供 deep-research 解説ベース) の続編で、(1) 具体的な 16GB VRAM コミュニティ報告、(2) 我々の decode 同期コピー壁に直結する機構詳細、(3) 突破パスへの対応付け を追加する。 既存 doc の採用マップ/q★ HW 依存性結論はそのまま有効。

## 1. 16GB VRAM の具体的報告 (探していた数値)

### RTX 5080 16GB + 128GB RAM (Pedro Moser @outrotuiter, X, 2026-08-26)
- **poolside Laguna S**: llama.cpp ~12 tok/s → FreeToken ~20 tok/s (1.67x)。 異種 expert geometry pool を追加する移植込み。
- **DeepSeek-V4-Flash**: llama.cpp ~13 → FreeToken ~30 tok/s single-stream (2.3x)、batch4 aggregate ~59 tok/s。
- **Qwen3.8-Flash-Next UD-IQ4_XS**: decode 26.3 tok/s / prefill ~900 tok/s。

### RTX 5080 16GB + DDR6 64GB + 9950X3D (Reddit r/LocalLLaMA "Freetokens project is impressive" OP)
- **Qwen3.6-35B-A3B NVFP4 (20GB = VRAM 非収容)**: 100 tok/s (1028 tok プロンプトで ~110)。

### 8GB RTX 4060 laptop + 64GB RAM (Reddit r/LocalLLM, vcruz305, Ornith-1.5-35B-A3B IQ3_S = 16GB GGUF)
- FreeToken decode **46.7–50.1 tok/s** (full-prompt streaming 44.5)、VRAM 6879/8188 MiB、**host RAM ~20GB pinned expert banks**、GPU 使用率 84–98%、load ~65s。
- llama.cpp CPU 同ファイル 11.08 tok/s → **4.5x**。
- 本人の要点: 「FreeToken は hot expert の GPU キャッシュを保ち、**マシンの帯域に基づいて streaming + CPU compute を自動で混ぜる**。flag hunting 不要」。
- 同氏は GGUF 対応拡張を PR #131 として上流提出 (Qwen MoE/dense, K/I quants, sharded)。

### RTX 4090 (X @0x0SojalSec)
- Qwen 3.6 35B NVFP4: **167 tok/s** (speculative decoding 0 / draft model なし)。

### RTX 5090 (X ぱぷりか炒め @WMjjRpISUEt2QZZ, Qwen3.8-Flash-Next-NVFP4 / RadixArk)
- 1x: prefill 277.5 / decode 59.13 tok/s。 2x: prefill 619.5 / decode 83.14 tok/s。

### 論文公式値 (arXiv 2608.16157)
- RTX 5090: Qwen3.6-35B-A3B **77–83 tok/s**、DSV4-Flash 22–25 tok/s (1.5–2.3x vs SOTA edge serving)。
- 8GB RTX 4060 laptop: 35B を **39.3 tok/s** (Codex の本番 decode 中央値 33 tok/s を超過; NVFP4 build は RTX 4090 rate の 92%)。
- 5 consumer systems 全体で decode 1.3–2.1x 改善。 RTX PRO 6000: GLM-5.2 14.9 vs llama.cpp 7.3 (2.0x)。

### 反例 (速くないケース)
- **GPT-OSS-20B-MXFP4** (X クインタス): Auto Mode 5.8 / 調整 15.2 tok/s → 「低 VRAM で RAM も巨大じゃない人は freetoken より **--cpu-moe** を使うべき」。
- r/accelerate OP は著者 Shuo Yang のマーケティング (Ollama 比 decode 3–4x / prefill 6–30x 主張)。

## 2. 突破機構の詳細 (論文本体 + issues)

### 2.1 q★ = 帯域適応 miss 分割 (decode 壁の本命レバー)
- miss した m 個の expert を、**2 つの実測帯域** — pinned expert 転送帯域 B_P と host 側 expert 処理帯域 B_H — のバランスで **PCIe 転送班と CPU 実行班に分割** (論文は q⋆ を B_P/B_H と m の式として記載)。
- 意味: **全 miss が同期 PCIe ストールにはならない**。ストールになる分の miss は CPU が並列で in-place 計算する。これが同期コピー壁を崩す機構。
- **Graph-resident CPU 実行**: CPU 分岐は同一 CUDA graph にキャプチャ。decode batch size ごとに stable pinned I/O buffer + persistent task descriptor を用意し、replay は per-token の host スケジューリング無しで full heterogeneous step を再実行。 worker は物理コアに pin された persistent C++ pool、in-kernel dequant。

### 2.2 Pinned expert banks + direct I/O (pinned memory 仮説の確認)
- 論文: 「aligned chunk を parallel direct I/O で exactly-size の host bank に読み込み、**埋まってから pin する**」。 expert bank は pinned memory → B_P を押し上げる。
- 我々の実測 h2d_gbps=2.63 は pageable。 FreeToken は pinning が設計の中心であることを実証 (先の pinned memory レバー仮説と一致)。 full pool を pin/DMA register できない OS/driver 構成ではフォールバック。

### 2.3 全層粒度のダブルバッファ prefill
- layer l の計算中に layer l+1 を PCIe ストリーム。 PCIe 5.0 x16 実効天井 52.7 GB/s に達し expert 計算は完全に転送の裏に隠れ 16k tok で 6.7k tok/s。 第 2 buffer 無効化は throughput の **19% 減**。

### 2.4 CUDA-graph-compatible LRU cache (論文 §4.1)
- single-pass kernel が LRU eviction 候補 K 個を特定 (slot ごとの全キャッシュ走査を回避)。 論理 routed ID を物理 slot ID か CPU 割り当てフラグに書き換える。
- **要点: FreeToken はキャッシュのために CUDA graphs を無効化しない。キャッシュを graph 互換にしている。** 我々の「glru/q* 有効時に graphs 強制無効化」(H-1 死にフラグ問題) とは逆方向の解。

### 2.5 LRU 命中率の実態 (issue #174; RTX 5090 32GB, DSV4-Flash 61 層 x 256 experts, 700 slots)
- decode 中の live expert: 240–248 / 256 (routing が 97% に拡散、working set ≈ モデル全体)。
- **LRU 実現 decode 命中率は 8–18%** (miss 82–92%)。
- **frequency-pinning oracle は同サイズで 46%**。 capacity 曲線はほぼ線形 (500 slots 40% / 700 46% / 900 52%) → **キャッシュ拡大はほぼ効かず、eviction policy がギャップ**。
- LRU が負ける理由: fine-grained MoE は routing が高速回転し recency が悪い予測器。
- **最重要 caveat**: **hybrid decode モードでは命中率はクリティカルパスではない** — miss のうち PCIe fetch される割合 (この環境 29.4%) だけがキャッシュに入り、残りは CPU lane が並列吸収する。 完全な ~30pp の機会が適用されるのは **pure GPU offload モード** (全 miss が同期 PCIe ストール) のみ。
- 論文本体の global LRU (RTX 5090 serving capacity = Qwen3.6 pool の 37% / DSV4 の 11%): decode expert read の miss 16% / 39% (vs KTransformers prefill-updated placement 41% / 59%、他 62% / 89%) → **十分なキャッシュ比率なら LRU は強い**、fine-grained + 少 slot で劣化。

### 2.6 警告: マイクロベンチ ≠ サービング (issues #151, #41)
- **#151** (2x RTX 3090, DSV4-Flash): `ft bench bw` は CPU 56.2 vs PCIe 12.3 GB/s を測り hybrid を選択。 だが実サービングは hybrid **0.67** vs offload **5.58** tok/s = **自動選択側が 8.3x 遅い**。 キャリブレートされた CPU 値はサービングに移行しない (実効 ~2 GB/s、マイクロベンチの 28 分の 1)。
- **#41** (`ft bench decode`): マイクロベンチと end-to-end tok/s は完全に食い違いうる (NUMA 固定がマイクロ +42% / サービング -6.7%)。 手法 = prefill を 2 点長で減算 `(n-1)/(t_n - t_1)` + variant を交互 cycle 化して drift 分散。

### 2.7 Predictive Expert Offloading (issue #176, feature request = 未実装)
- MTP ベースの次 expert 予測で先読み、3 階層 (primary GPU=hot / secondary GPU=MTP 予測 / RAM フォールバック)。 **我々の decode-PF (O-C) 提案と同一方向で、FreeToken でもまだ未実装。**

### 2.8 llama.cpp 上流も同機構を作業中
- **PR #25294** "llama : stream MoE routed experts from disk" (2026-07-04, open, 未マージ): 層ごと device-side n_slots expert slab キャッシュ、router top-k 後の CPU id-remap custom op、async I/O worker pool による GGUF 需要ロード、**O_DIRECT** (page cache バイパス)、**Wave-Partitional Prefill** (ubatch がキャッシュ超の expert に触れるとき wave 分割 + mask ゼロ合算)。 bit-exact。
- **RFC #24528**: thread 0 がキャッシュ row を GPU に dispatch する間、他 thread は計算する分割。

## 3. 我々の decode 同期コピー壁への対応付け (突破パス、優先順)

我々の壁: decode 同期 H2D ~6 t/s 天井 (glru コピー経路)、スラッシング 1.44 t/s、素 13 t/s、h2d_gbps=2.63 (pageable)、q* 不発 (qstar_cpu=0 / ~1.5 t/s)。

1. **[本命] q★ を「miss を CPU 並列へ分割 + graph-resident」に真に機能させる** — FreeToken の中核。 我々の q* は存在するが、残りの壁は remap op 内同期コピー。 目標: CPU 分岐を真に並列 + pinned I/O + graph キャプチャ。 ただし **#174/#151 と既存 mapping doc の HW 依存性に注意**: desktop 級 (PCIe 相対強く CPU 相対弱い) では q★≈0 が正当。 我々の RX 7800 XT (PCIe 4.0, 96GB RAM) は中間構成で、r2 受入バー `qstar_cpu>0` は「本当に割り振りが起きないか」の正しい反証テスト。
2. **[確認済み] expert bank の pin** — B_P を 2.63 (pageable) から PCIe 天井へ。 FreeToken が pinned bank + direct I/O で中心性と実証。 先の pinned memory レバー仮説を裏付け。
3. **[逆方向] graphs を無効化するのではなくキャッシュを graph 互換に** — FreeToken §4.1。 我々は現在 glru/q* 有効時に graphs 強制 OFF (H-1)。 graph-resident 実行はこのペナルティを除き、q★ の CPU 分岐 graph キャプチャの前提でもある。
4. **[eviction] frequency-informed eviction (decayed frequency で top-K pin + 尾部 LRU)** — #174 より、fine-grained MoE 向け。 ただし hybrid モードでは命中率はクリティカルパスでなく、効果は pure GPU offload 経路に限定される点に注意。
5. **[方法論] 常に end-to-end tok/s で判定しマイクロベンチを信じない** — #41/#151。 B の P2 3 軸受入 (絶対帯域/相対改善/長文完走) と接続する。

## 4. 注意
- FreeToken は **NVIDIA (Ampere 以降) / CUDA 13 / x86_64 専用**。 AMD/ROCm は未対応 (roadmap 上、Issue #82 コミュニティフォーク + Draft PR 途上 — 既存 mapping doc 参照)。 RX 7800 XT で直接は動かせないが、フォーク (Laguna) がまさにこれらのアイデアの llama.cpp ネイティブ移植。
- 数値の直接比較は無意味 (PCIe 5.0 vs 我々 4.0、NVFP4/MXFP4 vs 我々 Q4_K GGUF、DDR6 vs 我々環境)。 比較軸は常に「同一マシン上の before/after」(既存 mapping doc の注記と同一方針)。
- コミュニティ数値は単発・設定未検証の自己報告。 論文値が公式ベースライン。
