# コーダーAへの外注プロンプト — デコード側: q* 帯域適応協調実行 + グローバルLRUスロットプール

> そのままAIコーダー(Cline / OpenCode / ZCode等)に貼り付けて使うこと。
> 作業リポジトリ: https://github.com/dai8647/llama-cpp-turboquant-experts-laguna (branch: `main` から `feat/qstar-global-lru` を切ること)
> コーダーB(`feat/prefill-double-buffer`)と並行作業するため、**自分が所有するファイル以外を触らない**(下の「ファイル所有権」参照)。

---

## 1. ミッション

このリポジトリは llama.cpp のフォークで、FreeToken (https://github.com/FlashML-org/FreeToken) の思想
「MoEエキスパートをVRAMに全部載せず、ホットな分だけGPUスロットキャッシュに置いてミスは安く捌く」
を ROCm 環境(RX 7800 XT 16GB + 96GB RAM)向けに実装している。

あなたの担当は **デコードパスの超高速化**。具体的に残り2技術を実装する:

1. **q* 帯域適応ポリシー**(FreeTokenの目玉): キャッシュミスしたエキスパートを「PCIeでVRAMに転送してGPU計算」するか「ホストRAM帯域を使ってその場でCPU計算」するかを、実測帯域の比に基づいてステップごとに動的に決める。
2. **グローバルLRUスロットプール**: 現状はレイヤーごとに固定スロット数を割り当てている(`banks_by_layer`、各層 `n_slots` 固定)ため、ホットな層が冷たい層のVRAMを奪えない。全層でVRAM予算を共有し、層の垣根を越えてLRUで競合させる。

### 現状の実測ベースライン(超えるべき数字)

モデル: Huihui-Qwen3.6-35B-A3B (Q4_K, 20.2GB, qwen35moe, 256 experts/top-8)
`C:\Users\dai86\.lmstudio\models\huihui-ai\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf`

| 構成 | 短文tg | 長文(4670tok) pp / tg |
|---|---|---|
| スロット無効(純CPUストリーミング) | 12.38 t/s | 135.4 / 10.67 t/s |
| slot30(現行キャッシュ) | 11.91 t/s | 127.8 / **11.84 t/s (+11%)** |
| slot30 + prefetch 100ms | 11.06 t/s(逆効果) | — |

- 長文プリフィル後のデコードではキャッシュが+11%効く(ルーティングの時間的局所性)。
- 短文単発デコードではキャッシュが**逆効果**(12.38→11.91)。ミス時の同期H2Dコピーがストールするから。
- 対抗モデル(低局所性): Ornith-1.5-35B-A3B `C:\Users\dai86\.lmstudio\models\gbuzhf\Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF\Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Mini-v2D-lite.gguf` — これはトークンごとにルーティングが変わり、slot30でも96でも ~12.4 t/s で横ばい。**q* はこのケースでスロット無効超えを証明しなければならない**(ミスしてもCPU側で捌けばストールゼロ、が理想)。

### 目標数値(受け入れ基準)

- 短文単発デコード: スロット有効時がスロット無効(12.38)と**同等以上**。現状の-4%劣化をゼロにする。
- 長文後デコード: 11.84 t/s を維持または向上。
- Ornith(低局所性): 12.4 t/s 以上(スロット無効超え)。
- いずれも `LLAMA_MOE_SLOT_STATS=1` のテレメトリで hit/miss/evict/CPU実行比率を報告できること。

## 2. 環境(ROCm特化)

- Windows 11, RX 7800 XT 16GB (gfx1101), 96GB RAM, PCIe Gen4 x16
- ROCm 7.1 (`C:\Program Files\AMD\ROCm\7.1`, env `HCC_AMDGPU_TARGET=gfx1101`)
- ビルド: `cmake --build build-hip`(Ninja)。既存ビルドディレクトリ `build-hip/` がある。インクリメンタルビルドが通ること。
- 実行時は ROCm の bin を PATH に前置すること(`bench_ft.ps1` / `smoke_moe.ps1` が雛形)。
- **ROCm最優先**。ただし ggml-cuda ソースは CUDA と共有なので、他バックエンドを壊すな(`#ifdef GGML_USE_HIP` や既存の分岐を活用)。

## 3. 現行アーキテクチャ地図(まずこれを読め)

**最初に `docs/moe-slot-cache-async-design.md` を全文読むこと。** 設計思想・ロック規則・既知の罠が全部書いてある。

| 要素 | 場所 |
|---|---|
| キャッシュ構造体 `llama_moe_gpu_expert_cache`(slots_by_layer / banks_by_layer / expert_to_slot / LRU clock / hit-miss-evictカウンタ / `cache_mutex` recursive) | `src/llama-model.h:615` 付近 |
| ミスパス本体 `ensure_resident`(hit→last_used更新、miss→`find_free`/`find_lru_victim`→`preload_or_assign_slot`→`materialize_cb`) | `src/llama-model.h:1090-1128` |
| バンクVRAMバッファ確保(層ごとに n_slots 分のパック済みバッファ) `llama_moe_gpu_expert_bank_ensure` | `src/llama.cpp:479` |
| materialize(ホスト→VRAM直接コピー、Phase 1aで高速化済み) `llama_moe_gpu_expert_bank_copy_tensor` | `src/llama.cpp:474` |
| eval時スロットremapカスタムop `llm_moe_gpu_slot_remap`(CPUスレッドプールで実行、`ith==0` が全ミスを直列処理) | `src/llama-graph.cpp:266` |
| グラフ構築側 `build_moe_gpu_slot_ids`(`ggml_map_custom1`) | `src/llama-graph.cpp:1621` |
| ステップ間投機的prefetch(デコード専用、`n_tokens_all==1` ガード) | `src/llama-context.cpp:2030-2038`、本体 `src/llama.cpp:841` |
| 環境変数パース(`LLAMA_MOE_SLOT_STATS` / `LLAMA_MOE_PREFETCH_MS`) | `src/llama.cpp:670`, `src/llama.cpp:1128` |
| CLI引数 `--moe-gpu-expert-slot-num` | `common/arg.cpp:2705` |
| テレメトリ(n_copy / copy_bytes / copy_ns、4096回ごとにログ) | `src/llama-model.h:642` 付近 |

### ロック規則(壊すとデッドロックする)

- `cache_mutex`(recursive)→ `access_mutex` の順。逆は禁止。
- remap op はグラフ計算スレッドから呼ばれる。prefetch はグラフ非実行中に同期実行。この不変条件を保つこと。

## 4. 実装方針(提案。最終設計はあなたが行い、**まず設計ドキュメントを `docs/` に書いてから実装に入ること**)

### 4.1 グローバルLRUスロットプール

- 現状: `n_slots` が全層一律で、層ごとに独立バンク。総VRAM = 層数 × n_slots × エキスパートサイズ。
- 目標: 総スロット予算(例: 30)を全層で共有。ある層がミスしたら、**他層のLRU最古をevictしてでも**確保する。ホットな層が自然に多く持つ。
- バンクストレージの再構成(フラットアリーナ化 or 層ごとの動的増減)は設計で正当化すること。既存の `--moe-gpu-expert-slot-num N` の意味は「総予算N」として後方互換を保つ。
- 周波数配置(Pass 1/2 ワークフロー、`frequency_whitelist`)と衝突しないこと。

### 4.2 q* 帯域適応ポリシー

本質的な難所: **ルーティングはeval時まで分からない**(グラフは静的、remap op内で初めて選択が確定する)。つまりq*の判定はeval時のremap opの中で行う必要がある。段階的に:

- **Phase A(必須): 帯域キャリブレーション + ミスコストの隠蔽。** 起動時にH2D実効帯域(既存のmaterializeで計測可能)とCPU GEMMスループット(expertサイズの行列で計測)を校正し、`copy_ns` と比較して「転送が得かCPUが得か」の閾値を導出。まずミス時の同期ストールを削減する手段を設計する(専用HIPストリーム+イベントでresident expertのGPU計算とH2Dをオーバーラップするのが本命。ggmlはバックエンドのcompute streamをllama.cppに露出していないので、必要なら `ggml/src/ggml-hip/` に最小の露出APIを追加してよい)。
- **Phase B(本命): ミスしたエキスパートのCPU実行。** 転送待ちよりCPU計算が速いケース(低局所性・PCIe混雑時)で、ミスエキスパートをCPU側で計算して結果をマージする。ggmlのグラフモデル(mul_mat_idは単一バックエンドで実行)との整合が設計の核心。remap op内での自前GEMM+出力テンソルへの加算、あるいはubatch分割など、手段は提案すること。
- q* の判定式は論文の考え方を踏襲: 両経路がホストDRAM帯域を奪い合うことをモデル化し、実測帯域比からステップごとの分割数を決める。判定ロジックはテレメトリで検証可能にすること(「今ステップはx個転送/y個CPU実行」がログに出る)。

### 4.3 既存prefetchとの関係

`llama_moe_gpu_expert_slot_prefetch`(Phase 1b)は単発デコードで逆効果(11.06)と測定済み。q*実装後は役割が被るので、**廃止ではなく相互作用を測定し、設計ドキュメントに結論を書くこと**(prefetchはq*の「予測済みresident」として統合するのが自然)。

## 5. ファイル所有権(コーダーBとの衝突回避)

- **あなたが所有**(自由に変更可): `src/llama-model.h`, `src/llama-model.cpp`, `src/llama.cpp`(MoE部), `src/llama-graph.h/.cpp`(remap op), `ggml/src/ggml-hip/`(ストリーム露出が必要な場合のみ最小変更)
- **触ってはいけない**: `src/llama-context.cpp` のprefillまわり・グラフキャプチャ関連、`ggml/src/ggml-cuda/argsort.cu`(コーダーB所有)
- `common/arg.cpp` に引数を足す場合はBと被らないよう **MoE専用プレフィックス**(`--moe-*`)を使うこと。

## 6. 検証ラダー(機械が空いている時にこの順で)

1. ビルド: `cmake --build build-hip` がエラー0。`llama-server --version` 起動確認。
2. `smoke_moe.ps1` でスモーク(slot_remap全レイヤー発火 + クラッシュなし)。
3. Qwen3.6-35B-A3B で上記ベースライン3構成を再測し、**劣化がないこと**を確認してから新機能ON。
4. 新機能ONで目標数値(§1)を計測。`LLAMA_MOE_SLOT_STATS=1` のログをエビデンスとして添付。
5. Ornith で低局所性ケースの計測(q* の真価)。
6. 計測スクリプトは `bench_ft.ps1` / `smoke_moe.ps1` を拡張して使い、結果は `bench_results.txt` の形式(タブ区切り1行1構成)で追記。

## 7. ルール

- コミットメッセージは既存スタイル(`moe : ...` / `llama : ...`、動詞は現在形)。機能単位で細かく分割。
- 新機能はすべてオプトイン(envまたはCLI引数)。既定の挙動を変えない。
- 診断用のfprintfを残さない(前セッションで31箇所剥がしたばかり)。
- 設計ドキュメントは `docs/moe-slot-cache-async-design.md` の文体・構成(現状地図→発見→フェーズ設計→検証ラダー)に倣う。
- PR/成果物には「ベースライン比較表 + テレメトリログ + 実測コマンド」を必ず添付。数字のない完了報告は受け付けない。
