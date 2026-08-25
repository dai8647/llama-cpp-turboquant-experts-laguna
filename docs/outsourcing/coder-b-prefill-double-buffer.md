# コーダーBへの外注プロンプト — プレフィル側: ダブルバッファリング + HIPグラフキャプチャ修正 + エラスティックVRAM

> そのままAIコーダー(Cline / OpenCode / ZCode等)に貼り付けて使うこと。
> 作業リポジトリ: https://github.com/dai8647/llama-cpp-turboquant-experts-laguna (branch: `main` から `feat/prefill-double-buffer` を切ること)
> コーダーA(`feat/qstar-global-lru`)と並行作業するため、**自分が所有するファイル以外を触らない**(下の「ファイル所有権」参照)。

---

## 1. ミッション

このリポジトリは llama.cpp のフォークで、FreeToken (https://github.com/FlashML-org/FreeToken) の思想
「MoEエキスパートをVRAMに全部載せず、ホットな分だけGPUスロットキャッシュに置いてミスは安く捌く」
を ROCm 環境(RX 7800 XT 16GB + 96GB RAM)向けに実装している。

あなたの担当は **プレフィルパスの超高速化(TTFT短縮)** と、その前提になるバグ修正。3本立て:

1. **【ブロッカー修正】HIPグラフキャプチャ中のargsortクラッシュ**: 3130トークン以上のプレフィルで必ず墜ちる。現状の回避策は `GGML_CUDA_DISABLE_GRAPHS=1`(=グラフの恩恵を全部捨てる)で、これは速度面の損失なので恒久修正が必須。
2. **プレフィルのダブルバッファリング**(FreeTokenの目玉の1つ): プレフィルは数千トークンを同時に処理するため各層のほぼ全エキスパートが必要になり、実質Dense化する。GPUが層 l を計算している間に、バックグラウンドで層 l+1 のエキスパートをホストRAMからPCIe経由で先読み転送し、転送待ちを隠蔽する。
3. **エラスティックVRAMサイジング**: `--moe-gpu-expert-slot-num` は起動時固定で、空きVRAMを無視した手動指定。モデル+KVロード後の空きVRAMからスロット数を自動算出する `auto` モードを追加する(FreeTokenのelastic VRAM管理の最小版)。

### 現状の実測ベースライン(超えるべき数字)

モデル: Huihui-Qwen3.6-35B-A3B (Q4_K, 20.2GB, qwen35moe, 256 experts/top-8)
`C:\Users\dai86\.lmstudio\models\huihui-ai\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf`

| 構成 | 短文tg | 長文(4670tok) pp / tg |
|---|---|---|
| スロット無効(純CPUストリーミング) | 12.38 t/s | **135.4 t/s** / 10.67 t/s |
| slot30(現行キャッシュ) | 11.91 t/s | 127.8 t/s(**-6%の罰金**) / 11.84 t/s |

- **プレフィルの罰金(-6%)の原因はミス時の同期H2Dコピーchurn**。ダブルバッファリングでこれを隠蔽し、スロット有効のまま pp ≥ 135 t/s(スロット無効超え)が目標。
- デコード側の+11%(10.67→11.84)はキャッシュの実効果なので、**プレフィル高速化でこの+11%を壊さないこと**。

### 目標数値(受け入れ基準)

1. **グラフキャプチャ修正**: 4670トークンプレフィルが `GGML_CUDA_DISABLE_GRAPHS` **なし**(グラフON)でクラッシュゼロ。かつグラフON時のtgがグラフOFF時と同等以上。
2. **ダブルバッファ**: slot30構成の pp が 135.4 t/s 以上(スロット無効超え)、tg は 11.84 t/s 維持。
3. **autoサイジング**: `--moe-gpu-expert-slot-num auto` で起動し、空きVRAMから妥当なスロット数(ログに算出根拠を出力)が決まり、手動指定と同等以上の性能。

## 2. 環境(ROCm特化)

- Windows 11, RX 7800 XT 16GB (gfx1101), 96GB RAM, PCIe Gen4 x16
- ROCm 7.1 (`C:\Program Files\AMD\ROCm\7.1`, env `HCC_AMDGPU_TARGET=gfx1101`)
- ビルド: `cmake --build build-hip`(Ninja)。既存ビルドディレクトリ `build-hip/` がある。インクリメンタルビルドが通ること。
- 実行時は ROCm の bin を PATH に前置すること(`bench_ft.ps1` / `smoke_moe.ps1` / `probe_health.ps1` が雛形)。
- **ROCm最優先**。ただし ggml-cuda ソースは CUDA と共有なので、他バックエンドを壊すな(`#ifdef GGML_USE_HIP` や既存の分岐を活用)。

## 3. 現行アーキテクチャ地図(まずこれを読め)

**最初に `docs/moe-slot-cache-async-design.md` を全文読むこと。** §4に「プレフィルのダブルバッファは未実装(Phase 2)」、§5に「真のオーバーラップにはバックエンドストリームへのアクセスが必要で、ggmlはそれをllama.cppに露出していない」という既知の制約が明記してある。この制約の攻略があなたの仕事の中核。

| 要素 | 場所 |
|---|---|
| **argsortクラッシュ箇所**(hipcub `DeviceSegmentedRadixSort`、キャプチャ判定ロジックあり) | `ggml/src/ggml-cuda/argsort.cu:83-165` |
| キャッシュ構造体 `llama_moe_gpu_expert_cache` / ミスパス `ensure_resident` | `src/llama-model.h:615`, `1090-1128` |
| バンクVRAMバッファ確保 `llama_moe_gpu_expert_bank_ensure`(層ごとに n_slots 分) | `src/llama.cpp:479` |
| materialize(ホスト→VRAM直接コピー) `llama_moe_gpu_expert_bank_copy_tensor` | `src/llama.cpp:474` |
| ロード時プリロード `llama_moe_gpu_expert_slot_preload` | `src/llama.cpp:799` |
| eval時remapカスタムop `llm_moe_gpu_slot_remap` | `src/llama-graph.cpp:266` |
| デコード専用prefetchフック(`n_tokens_all==1` ガード — **プレフィルは対象外**) | `src/llama-context.cpp:2030-2038` |
| CLI引数 `--moe-gpu-expert-slot-num`(現状は数値のみ、-1で無効) | `common/arg.cpp:2705` |
| グラフキャプチャ切替 env `GGML_CUDA_DISABLE_GRAPHS` | `ggml/src/ggml-cuda/common.cuh:1272` 付近 |

### ロック規則(壊すとデッドロックする)

- `cache_mutex`(recursive)→ `access_mutex` の順。逆は禁止。
- グラフ計算中にキャッシュを変更する経路はremap opのみ。それ以外のタイミングでの同期コピーは「グラフ非実行中」が不変条件。バックグラウンド転送スレッドを導入するなら、この不変条件をどう保つかを設計ドキュメントで明示すること。

## 4. 実装方針(提案。最終設計はあなたが行い、**まず設計ドキュメントを `docs/` に書いてから実装に入ること**)

### 4.1 argsortキャプチャクラッシュ(最優先・最初にやること)

観測されたエラー:
```
ROCm error: operation not permitted when stream is capturing
→ argsort_f32_i32_cuda_hipcub (DeviceSegmentedRadixSort)
```
- コードは既にキャプチャを検出して `DeviceSegmentedRadixSort` に切り替えている(`argsort.cu:83` のコメント参照)。しかしROCmではキャプチャ中に落ちる。
- 疑わしい箇所は2つ: ① `ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes)` がプールミス時にキャプチャ中で hipMalloc を呼ぶ、② hipcub の `DeviceSegmentedRadixSort` 内部がキャプチャ非対応の割り当て/同期を行う。
- まずどちらが原因か切り分け(キャプチャ前に同じサイズで1回ウォームアップしてプールを温めれば①は切り分けられる)、その上で恒久修正: ワークスペースのキャプチャ前確保・固定サイズ化・キャプチャ対象グラフからのargsort除外、など。
- 修正後、3130/4670トークンのプレフィルでグラフONクラッシュゼロを確認すること。top_kサンプラーを使うパスでargsortが走る点に注意(バッチサイズでnrowsが変わる→ワークスペースサイズも変わる)。

### 4.2 プレフィルのダブルバッファリング

- 現状: プレフィル中のミスはremap op内の**同期**H2Dで捌かれ、GPUが待つ(-6%罰金の原因)。
- 目標: プレフィルをチャンク(ubatch)単位で見て、チャンク内の層 l のGPU計算中に層 l+1 (以降)の必要エキスパートを別ストリーム/別スレッドで先読み。FreeTokenは「層l計算中に層l+1の全エキスパート」だが、このフォークではチャンクごとのtop-k union(設計メモ §5 Phase 2 参照)で必要分だけ転送するのが現実的。
- 転送にはダブルバッファ(ステーティング2面)を使い、コピーと計算をオーバーラップ。
- ggmlがバックエンドのcompute streamを露出していない問題: 必要なら `ggml/src/ggml-hip/` に最小APIを追加してよい(所有権内)。または「グラフとグラフの間(ubatch境界)」に同期転送を詰めるだけの簡易版から始めてもよいが、その場合は隠蔽率が下がることを数字で示すこと。
- **デコードパス(既存の `llama_moe_gpu_expert_slot_prefetch`)は変更しない**。プレフィル専用パスとして設計すること(コーダーAがデコード側を改造中)。

### 4.3 エラスティックVRAMサイジング(autoモード)

- `--moe-gpu-expert-slot-num auto`: モデル重み+KVキャッシュ確保後の空きVRAMを問い合わせ(`ggml_backend_dev_get_memory` 等)、エキスパート単価(層あたりバンクサイズ/n_slots)で割ってスロット総数を自動決定。決定根拠(空きVRAM、単価、採用したスロット数)を起動ログに出す。
- 安全マージン(例: 空きVRAMの80%まで)を設け、上限capもenvで指定可能にすること。
- 将来の「再起動なしの動的再配分」(FreeToken本来のelastic管理)はストレッチゴール。今回は起動時autoまででよいが、構造上再配分が不可能にならない設計を選ぶこと。

## 5. ファイル所有権(コーダーAとの衝突回避)

- **あなたが所有**(自由に変更可): `ggml/src/ggml-cuda/argsort.cu`, `ggml/src/ggml-hip/`(ストリーム露出・キャプチャ関連), `src/llama-context.cpp`(プレフィルフック), `src/llama.cpp` の preload/bank_ensure 周辺と引数パース部, `common/arg.cpp`(`--moe-gpu-expert-slot-num` の auto 対応)
- **触ってはいけない**: `ensure_resident` / LRU / evict ロジック、remap op のミス判定本体(コーダーA所有)。あなたのパスからは既存の公開API(`preload_or_assign_slot`, `materialize_cb` 等)経由でキャッシュを使うこと。
- 両方が `src/llama.cpp` と `src/llama-model.h` に用がある場合は、**関数単位で所有権を分け**(B: preload/bank確保/引数、A: ensure_resident/LRU/prefetch本体)、マージ衝突はリベースで解決する。

## 6. 検証ラダー(機械が空いている時にこの順で)

1. ビルド: `cmake --build build-hip` がエラー0。`llama-server --version` 起動確認。
2. **グラフ修正の検証**: `probe_health.ps1` 相当の起動 + 3130トークン以上のプレフィルをグラフON(= `GGML_CUDA_DISABLE_GRAPHS` なし)で実行。クラッシュゼロ + ログにキャプチャ成功を確認。
3. ベースライン再測: Qwen3.6-35B-A3B でスロット無効/slot30の pp・tg を再測し、改変による劣化がないことを確認。
4. ダブルバッファONで pp ≥ 135 t/s かつ tg ≥ 11.84 t/s を計測。
5. `--moe-gpu-expert-slot-num auto` で起動し、算出ログと性能を確認(手動slot30比較)。
6. 計測スクリプトは `bench_ft.ps1` / `smoke_moe.ps1` を拡張して使い、結果は `bench_results.txt` の形式(タブ区切り1行1構成)で追記。

## 7. ルール

- コミットメッセージは既存スタイル(`moe : ...` / `ggml : ...` / `llama : ...`、動詞は現在形)。機能単位で細かく分割。
- 新機能はすべてオプトイン(envまたはCLI引数)。既定の挙動を変えない(グラフ修正はバグ修正なので常時ONでよい)。
- 診断用のfprintfを残さない(前セッションで31箇所剥がしたばかり)。
- 設計ドキュメントは `docs/moe-slot-cache-async-design.md` の文体・構成(現状地図→発見→フェーズ設計→検証ラダー)に倣う。
- PR/成果物には「ベースライン比較表 + 実測コマンド」を必ず添付。数字のない完了報告は受け付けない。
