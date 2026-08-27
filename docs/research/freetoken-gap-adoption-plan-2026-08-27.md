# FreeToken ギャップ一覧 + 導入計画 (review 見解)

- 日付: 2026-08-27
- 作成: review セッション (user リクエスト「他に freetoken にあってこちらにない技術を一覧にして入れる計画作って」)
- 目標: **tg 40 t/s 級** (stretch ≥30) on glru slot96 + 6575tok 完走 (user 決裁・旧受入ライン 11 t/s は廃止)
- 根拠ドキュメント: freetoken-16gb-community-and-mechanism-2026-08-27.md (@61f1d9c8e) / freetoken-vs-laguna-mapping.md (@165590310) / FreeToken 論文 §2.1・§2.2・§4.1 / issues #174・#151・#41・#176 / 上流 PR #25294
- 前提数字: glru slot96 tg=1.01-1.44 (ラン間分散) / 素ベースライン ~13 t/s (graphs ON 状態) / E1 decode 相 h=0.413 (14 window) / B_h2d=2.12-2.69 GB/s (pageable) / calibrate cpu=1.1 GB/s / モデル = Qwen3.6-35B-**A3B** (アクティブ 3B) Q4_K 36 層・MTP モジュール入り

---

## §1. 保有済み (導入不要)

| FreeToken 技術 | うちの対応 | 状態 |
|---|---|---|
| 共有 LRU expert キャッシュ (§2.1.3) | glru = プロジェクトの出発点そのもの | ✅ |
| Prefill 全層ダブルバッファ (§2.2) | LLAMA_MOE_PREFILL_PF=1 (3b49ca50a→2d442cdb0, main マージ済 76adf21e4) | ✅ |
| Elastic VRAM 管理 (§2.4) | --moe-gpu-expert-slot-num auto (8ca1025a6) | ✅ |

## §2. ギャップ一覧 (FreeToken にあってうちにないもの)

| # | FreeToken 技術 | うちの現状 | ギャップの内容 | tg への効き目 | 優先度 |
|---|---|---|---|---|---|
| **G1** | **Pinned expert banks + direct I/O** (deep-dive §2.2) | pageable メモリ経由 H2D = 2.12-2.69 GB/s | host 側 expert bank を pinned (hipHostMalloc) で確保・aligned chunk を direct I/O で読み込み fill 後に pin。 PCIe 4.0 x16 実効 ~15-20 GB/s へ | **全レバーの土台**。 転送天井 6 t/s → 30-60 t/s 帯へ。 これなしでは他全部が無意味 | **Stage 1 (最優先)** |
| **G2** | **CUDA-graph 互換 LRU キャッシュ** (論文 §4.1) | glru/q* 有効時は graphs を init 時に自動無効化 (H-1 適用後は INFO 出力のみ・依然 OFF) | single-pass kernel が K 個の eviction 候補を選出 → 論理 routed ID を物理 slot ID か CPU 割当フラグへ書き換え。 固定 slot アドレスで 1 回キャプチャし、リプレイ間に slot 中身だけ更新。 **FreeToken はキャッシュのために graphs を切らない** | **4x の本体**。 素 13 t/s は graphs ON の数字。 glru 経路の 1.0-1.4 t/s には同期コピー＋毎トークン host 起動オーバーヘッドが両方乗っている | **Stage 2** |
| **G3** | q★ bandwidth-adaptive miss 分割 (§2.1.1-2) | コードは存在するが全ラウンド qstar_cpu=0 (r2 rebuild 系譜にのみ残存) | 実測 B_P/B_H に基づき miss expert を PCIe 転送班と CPU 並列計算班に分割 | **この機械では低い** — calibrate cpu=1.1 GB/s vs pinned PCIe ~15-20 GB/s = CPU レーンが 15-18 倍遅い。 論文 §2.1.2 自身の HW 依存性主張 (デスクトップ級はほぼ全 miss を PCIe へ) と qstar_cpu=0 は整合 | Stage 4 (条件付き・低) |
| **G4** | Graph-resident CPU 実行 (§2.1) | 無し (CPU 分岐は graph 外・毎トークン host スケジューリング) | CPU 分岐を同一 graph にキャプチャ。 decode バッチサイズ毎の安定 pinned I/O バッファ + persistent task descriptor・物理コアピン留め C++ ワーカープール・カーネル内 dequant | G3 が効くための前提。 G3 と同理由でこの機械では価値低い | Stage 4 (G3 と同時・低) |
| **G5** | Frequency-informed eviction (#174) | 純 LRU のみ (decode 相 h=0.413) | decayed-frequency top-K pin + 末尾 LRU。 #174 実測: LRU decode ヒット率 8-18% vs freq-pin oracle 46%・容量曲線はほぼ線形 (キャッシュ増設はほぼ効かない・eviction ポリシーが差) | **うちは pure GPU offload モード** (miss は全部 PCIe 同期ストール・CPU レーンが吸収しない) なので h はクリティカルパス上。 h 0.41→0.5+ = 転送量 -15% 以上。 ただし #174 caveat: hybrid モードでは h は効かない | Stage 3 |
| **G6** | 自動 bandwidth ベースモード混合 ("no flag hunting") | 手動フラグ (--moe-gpu-expert-slot-num / --moe-gpu-expert-global-lru / env) | 実測機械 bandwidth から streaming+CPU 混合を自動選択 | UX・頑健性であってピーク速度ではない | 最低 (Stage 4 以降) |
| **G7** | MTP 予測 expert オフロード (#176) | 無し (我々の decode-PF 提案と同方向) | 3 層構成: primary GPU hot / secondary GPU MTP 予測分 / RAM フォールバック。 **FreeToken 自身も未実装** (feature request 段階・issue のみ) | うちのモデルは MTP 入り (Huihui-...-MTP-GGUF) = 好機。 ただし研究グレード・Stage 1-2 の壁を越えてからの話 | Stage 4 (研究) |

### 対象外

| 技術 | 対象外理由 |
|---|---|
| NVFP4/MXFP4 量子化 | NVIDIA 専用・うちは ROCm/RDNA3 + Q4_K GGUF (別量子化パス) |
| Semantic CKPT (§2.3) | 対象モデルに recurrent 層が無い |

### 技術ではなく必須ルール (方法論)

- **判定は end-to-end tok/s のみ** (#151/#41: マイクロベンチ 8.3x 乖離の実例・NUMA ピン留めが マイクロ +42% / サービング -6.7% という逆転例)。 計算見積だけで最終結論を出さない = E1 ゲートの教訓 (169ms 見積は正しかったが、見積自体は壁を壊さない)。
- 全ゲートは `verify_a.ps1 -PromptKind short` (6575tok) 完走で測る。 マイクロベンチ・単発コピー計測は参考値どまり。

## §3. 導入計画 (ステージ制・集中作業モード)

実装は A セッション (user 直接駆動)。 以下は review の推奨順序とゲート。

### Stage 1 = G1: pinned + async + 早期 fire (= A 方針 C・graph 互換設計制約付き)

- **内容**:
  1. host 側 expert bank を pinned 化 (hipHostMalloc・pageable からの切替)
  2. 非同期 H2D (stream/event・prefill-PF 機構 = event/inflight/resident 延期/drain, llama.cpp:965-1083 の decode 移植)
  3. 早期 fire = decode の層ホッピング中に次 layer expert の H2D を先行開始
- **設計制約 (最重要)**: Stage 1 の実装は **graph-capturable な設計**で作ること = VRAM slot アドレスは固定・selection は indirection/gather 経由・動的 remap を避ける。 これを守らないと Stage 2 が全面リライトになる。
- **回避**: クラッシュ中の旧 prefetch 経路 (LLAMA_MOE_PREFETCH_MS>0 の inter-step prefetch) は使わず、生きてる既存 glru コピー経路 (H-1/H-2 適用済) に実装。
- **規模見積**: +100-200 行 (A 方針 C 見積)。
- **ゲート**: h2d_gbps ≥ 10 かつ tg ≥ 10 t/s (6575tok 完走・数字必須)。

### Stage 2 = G2: graphs 互換 paging (4x の本体)

- **内容**: 論文 §4.1 設計 — eviction 候補選出 kernel + 論理→物理 slot ID 書き換え。 固定 slot アドレスで 1 回キャプチャ・リプレイ間に slot 中身を更新。 glru 有効時の graphs 自動無効化 (H-1 で導入したパス) を解除し graphs を再有効化。
- **参照実装**: 上流 llama.cpp PR #25294 (device 側 n_slots slab キャッシュ + CPU id-remap custom op + async I/O ワーカープール・bit-exact 設計) が同設計空間。
- **ゲート**: **tg 40 t/s 級** (stretch ≥30) + 6575tok 完走。

### Stage 3 = G5: frequency pinning (ヒット率レバー)

- **内容**: decayed-frequency top-K pin + 末尾 LRU (#174 の freq-pin oracle 46% を狙う)。
- **ゲート**: h ≥ 0.5 かつ end-to-end tg 改善が数字で示せること。
- **注**: Stage 2 で 40 t/s に到達していればスキップ可 (転送が律速でなくなった後では h 改善の tg 寄与が小さい)。

### Stage 4 (optional・研究)

- **G7 MTP 予測オフロード**: モデルが MTP 入りなので好機だが FreeToken も未実装の研究グレード。 Stage 1-3 完了後の検討。
- **G3/G4 q★ CPU 分割 + graph-resident CPU 実行**: Stage 1-3 後も転送予算が足りない場合のみ。 この機械では不要の公算大 (cpu 1.1 GB/s vs pinned PCIe ~15-20 GB/s・論文 §2.1.2 の HW 依存性主張と qstar_cpu=0 実測が整合)。
- **G6 自動モード混合**: UX 改善・ピーク速度とは無関係。

## §4. なぜこの順序か

1. **G1 が全前提**: pageable 2.12 GB/s のままでは何を乗せても転送天井 ~6 t/s。 まず bandwidth を PCIe 上限まで上げる。
2. **G2 が乗数**: 素ベースライン 13 t/s は graphs ON の数字。 glru 経路が graphs OFF + 同期コピーを背負っている限り 40 t/s は構造的に不可能。 G1 で bandwidth を直し G2 で毎トークン host オーバーヘッドを消す = FreeToken の 4x の再現パス。
3. **G5 は壁突破後の微調整**: h 0.41→0.5 は転送量削減だが、B_P が 15 GB/s+ になれば転送は律速ではなくなる可能性が高い。
4. **G3/G4 はこの機械では低優先**: 論文自身の HW 依存性主張 + 我々の calibrate 実測 (cpu=1.1 GB/s) が根拠。

## §5. NVIDIA→RDNA3 移植差分

- FreeToken = CUDA 13 / Ampere+ / x86_64。 うちは HIP / gfx1101 (RDNA3・wave32 なので warp サイズ問題は無し)。
- CUDA graphs → HIP graphs: llama.cpp の graphs は ggml_backend 抽象越し・capture/update セマンティクスは等価。 H-1 で init パスは既に触ってある。
- pinned: hipHostMalloc ≈ cudaHostAlloc。
- direct I/O: Windows では FILE_FLAG_NO_BUFFERING (要アライメント)。 ただしうちは GGUF から host bank へ読み込み済みなので direct I/O 部分は低優先 — **pinned 化だけで効果の大部分が取れる**。
- q★ の CPU dequant (Stage 4 時): GGML CPU バックエンドの Q4_K カーネルが再利用可能。

## §6. -community 数字との対比 (目標の妥当性)

- 4060 laptop 8GB: Ornith-1.5-35B-A3B IQ3_S で 46.7-50.1 t/s (llama.cpp 11.08 の 4.5x) — **A3B クラス 35B が 16GB 未満の GPU で 40 t/s+ の実例**。
- 5080 16GB: Qwen3.6-35B-A3B NVFP4 で 100 t/s (VRAM 非居住)。 論文: 4060 laptop 39.3 t/s・5090 で 77-83 t/s。
- うち = RX 7800 XT 16GB (VRAM 644 GB/s) + A3B アクティブ 3B → 計算床は 40 t/s を大きく下回る。 **40 t/s は物理的に現実的**。 足りないのは HW ではなく上記 G1+G2 の機構。
- 反例注意: GPT-OSS-20B-MXFP4 は FreeToken でも低速 → 「低 VRAM なら --cpu-moe が良い」ケースが存在。 ただし A3B 35B の pure offload は成功例が厚い。
