# Stage 2 設計: graphs 互換 paging — 静的調査結果と移植計画 (review)

- 日付: 2026-08-27
- 作成: review セッション (作業分配の review 担当分・user 承認済)
- 対象ブランチ: feat/qstar-r2-rebuild @e41fb61fc (調査時点)
- 目的: Stage 1 (pinned+async+早期 fire・ゲート tg≥10+h2d_gbps≥10) 通過後、 Stage 2 = CUDA graphs 再有効化 + paging 共存 (ゲート tg 40 t/s 級) に即着手できるよう、コード事実と実装計画を確定する。
- 設計の参照: FreeToken 論文 §4.1 (graph 互換 LRU)・上流 llama.cpp PR #25294 (device 側 slab キャッシュ+id-remap custom op+async I/O)。

---

## §1. 静的調査で確定したコード事実 (file:line は調査時点)

### 1.1 graphs 自動無効化パス (kill switch は 3 段)

1. **フラグ宣言**: `src/llama-model.h:732` — `bool graphs_disable_pending = false;` (`llama_moe_gpu_expert_cache` 内)。
2. **セット (2 経路)**: `src/llama.cpp:2055` (手動 init・qstar/global-lru 有効時) と `src/llama.cpp:1259` (auto init・elastic VRAM 経路)。 両所で INFO 「CUDA graphs will be disabled」 出力。
3. **消費**: `src/llama-context.cpp:468-474` — context 構築時に全 accel backend へ `ggml_backend_cuda_ext_set_graphs_enabled(backend, false)`。
4. **実体**: `ggml/src/ggml-cuda/ggml-cuda.cu:2729-2734` が `ext_graphs_disabled` フラグ (`common.cuh:1528`) をセットし、 `ggml-cuda.cu:4521` の `graph_compatible` 判定で読む。
5. **env**: `GGML_CUDA_DISABLE_GRAPHS` は `common.cuh:1271-1274` で process-wide kill (これは今後も維持)。

### 1.2 capture/replay 機構 (ポインタ安定性が全て)

- capture は `ggml_backend_cuda_graph_compute` (`ggml-cuda.cu:4503-4565`)。 キー = `cgraph->nodes[0]` (先頭ノードポインタ毎に 1 graph)。
- **warmup 規則**: ノードプロパティが変わらず 2 回連続呼び出しで capture。 プロパティ比較 (`ggml_cuda_graph_update_required`, `ggml-cuda.cu:2890-2930`) は各ノードの `ggml_tensor` 構造体 + 全 src の **data ポインタ**/ne/nb を memcmp = **ポインタが 1 つでも変わると warmup リセット→再キャプチャ** (毎ステップになると graphs は無意味化する)。
- **毎ステップ入力の更新方法**: llama 層は固定 device アドレスの入力 tensor へ `ggml_backend_tensor_set` で中身だけ書き換える (`llm_graph_input_embd::set_input`, `src/llama-graph.cpp:69-81`・呼び出し `src/llama-context.cpp:1394`)。 **キャプチャ後の graph へポインタ/スカラをパッチする ext API は存在しない** (内部の `cudaGraphExecUpdate` ラッパ `ggml-cuda.cu:2932-2957` は再キャプチャ時のフォールバック専用)。
- 結論: **graph 互換 paging の唯一の道は「キャプチャ対象の全ポインタを固定し、中身だけ更新する」= FreeToken §4.1 方式そのもの**。

### 1.3 remap op = graph 内ノードだが CPU 実行 (最重要発見)

- `build_moe_gpu_slot_ids` (`src/llama-graph.cpp:1837-1862`) が `ggml_map_custom1` として remap op を挿入。 コールバック (`llama-graph.cpp:234-268`) は host ポインタを直接参照し **CPU バックエンドで毎ステップ実行される**。
- `ggml_cuda_graph_check_compability` (`ggml-cuda.cu:2865-2877`) は MAP_CUSTOM ノードを拒否するが、コメント明記: 「scheduler が CPU backend へルーティングするので CUDA graph には現れない・ルーティングが変わったら bail」。
- **含意**: graphs を再有効化しても remap op (expert_id→slot_id 変換・LRU 更新・eviction 判定) は毎ステップ host 側でそのまま走る。 **graph の中身をキャプチャされるのではない**。 本当のブロッカーは以下の 2 点のみ:
  1. miss 時の**同期 H2D** が graph evaluation 中にネストする (§1.4)。
  2. bank tensor の **data ポインタが毎ステップ変わりうる** (= 1.2 の warmup リセットを引き起こす)。

### 1.4 同期 H2D の正体 (壁の全経路)

decode の miss 1 件毎の連鎖:
`ensure_resident` (`src/llama-model.h:1296-1376`) → `materialize_cb` = `llama_moe_gpu_expert_slot_materialize` (`src/llama.cpp:687`) → `llama_moe_gpu_expert_bank_copy_tensor` (`src/llama.cpp:484-514`) → `ggml_backend_tensor_set` (`:506`/`:512`) → `ggml_backend_cuda_buffer_set_tensor` (`ggml-cuda.cu:902`) = `cudaMemcpyAsync + cudaStreamSynchronize` (`:932-933`)。

- **既存の非同期代替**: `ggml_backend_cuda_ext_h2d_async` (`ggml-cuda.cu:2669-2681`・宣言 `ggml/include/ggml-cuda.h:52-53`) = 専用 `ext_copy_stream` 上での cudaMemcpyAsync。 event 系 API も完備 (`ext_event_create/record/query/synchronize/destroy`, `ggml-cuda.cu:2683-2727`)。 **ただし使用箇所は prefill ダブルバッファ経路のみ** (`src/llama.cpp:1110`・drain `:709-711`)。 decode 未使用 = Stage 1 がここを移植する。

### 1.5 毎ステップ host 側シーケンス (glru decode)

1. remap op 発火 (graph eval 中・CPU) → 2. LRU 更新 (`record_access`, `llama-model.h:1303,1316-1322`) → 3. eviction (global-LRU `:1331-1345` / per-layer `find_lru_victim` `:1347-1356`) → 4. slot 割当 (`preload_or_assign_slot` `:1273-1294`) → 5. materialize (同期 H2D・上記) → 6. `record_selections` (`:792-800`)。 step 間に `llama_moe_gpu_expert_slot_prefetch` (`src/llama.cpp:883-948`, 呼び出し `src/llama-context.cpp:2067-2072`・opt-in)。

---

## §2. Stage 2 実装計画 (Stage 1 の機構の上に積む)

### 2.1 前提 (Stage 1 が満たしているべきもの)

- pinned staging bank (hipHostMalloc) + `ext_h2d_async` + event による非同期 H2D (decode 経路移植済)
- **固定 VRAM スロットアドレス** (per-slot アドレス不変・`compute_tensor` は bank 固定ベースを返す)
- 早期 fire により miss コピーの大部分は使用前に完了済

### 2.2 変更点 (4 点)

1. **miss 処理の非同期化完了を保証する**: remap コールバック (CPU・graph launch より前に topological 順で走る) が、割当 slot の event 未完了なら **host 側で `ext_event_synchronize` して待つ** (graph 内に wait を入れない)。 pinned 15-20 GB/s なら 1 expert (1.95 MiB) の待ちは ~100-130µs・早期 fire 後はほぼゼロ。 同期 `ggml_backend_tensor_set` 経路は decode から完全撤去。
2. **graphs 再有効化スイッチ**: `graphs_disable_pending` のセット (llama.cpp:1259/2055) か消費 (llama-context.cpp:468-471) を新フラグ (例: env `LLAMA_MOE_GRAPHS=1` or CLI) でスキップ。 A/B 比較のため即時切替可能にしておく。 `GGML_CUDA_DISABLE_GRAPHS` は process-wide kill として維持。
3. **ポインタ安定性の監査と固定**: `compute_tensor` (`src/llama-graph.cpp:2492-2503` が bank tensor を swap) が固定ベースを返すことを確認。 `safe_unbanked_fallback` (`llama-model.h:1378+`) はポインタを変えるので **graphs モードでは発火禁止** (発火=graphs 無効化フォールバック or abort+WARN)。 slot_ids 出力バッファは固定 CPU メモリ = 毎ステップ captured H2D が中身を拾う (embd 入力と同一パターン・§1.2)。
4. **eviction の安全化**: 前ステップの replay は sampling 同期 (logits の host 転送) で完了済み = eviction は次ステップの remap 時点で安全。 ただし **inflight event 付き slot の eviction 禁止** (event query で確認 or 完了まで victim 選出から除外)。

### 2.3 やらないこと

- captured graph への in-place パッチ API 新設 (ext API 無し・`cudaGraphExecUpdate` は再キャプチャ補助のみ) — 固定ポインタ+中身更新方式で足りる。
- remap op の GPU 化 (CPU ルーティング維持 = §1.3 の compat 判定が通り続ける条件)。

### 2.4 リスクと検証項目

| リスク | 検知方法 | 対策 |
|---|---|---|
| ポインタ変動で毎ステップ再キャプチャ | capture 回数カウンタをログ追加 (warmup_complete 遷移) | 変動源を特定して固定 (fallback 経路・tensor 再バインド) |
| MAP_CUSTOM が CUDA にルーティング変更される | `ggml-cuda.cu:2870` の compat 判定が graphs を veto | 現状維持を確認・変更時はレビュー必須 |
| q* 経路 (`llm_moe_gpu_slot_remap_qstar`, `llama-graph.cpp:277+`) の [ids\|mask] 出力 | 値は毎ステップ変わるがバッファ固定 = 原理上問題なし | Stage 1.5 の q* ON スモークで確認 |
| replay と非同期コピーの競合 | copy stream と compute stream の無順序 | slot 使用前に event 待ち (§2.2-1)・eviction 制限 (§2.2-4) |
| ROCm 差分 | hipGraph capture は CUDA と等価セマンティクス (既知) | `hipEventQuery` 使用継続 (ROCm 7.1 に cudaEventQuery alias なし・既知罠) |

### 2.5 期待効果の予測

- 素ベースライン ~13 t/s と graphs OFF の差分 = graphs の寄与分 (**B が並行測定中** = graphs 帰属ラン・結果が出次第ここに数字を追記)。
- Stage 1 (bandwidth 2.12→≥10 GB/s) × Stage 2 (毎トークン host 起動オーバーヘッド除去) の積で 40 t/s 級を狙う。 A3B アクティブ 3B の計算床は 40 t/s を大幅に下回る (VRAM 644 GB/s) ため、この 2 段で届く物理的余地はある。

## §3. Stage 1 diff レビューのチェックリスト (A の 4 commit 用・review 担当)

- [ ] bank の data ポインタが全経路で固定 (slot 割当でポインタを返していないか)
- [ ] `ggml_backend_tensor_set` (同期) が decode miss 経路から消えているか
- [ ] `ext_h2d_async` の src が pinned メモリか (pageable なら cudaMemcpyAsync が内部同期して効果消滅)
- [ ] event の record/wait 対応が漏れなくあるか (inflight slot を読んでいないか)
- [ ] pinned 確保失敗時のフォールバックが安全か (同期経路戻り or 明確なエラー)
- [ ] llama-graph.cpp:1837 remap op / llama-context.cpp:471 graphs 無効化が**未変更**か (Stage 1 は触らない約束)
- [ ] stats に h2d_gbps が実測で出ているか (ゲート判定用)

---

## §4. 訂正 (2026-08-27 13:4x): B の実測で graphs 寄与 ≈0 → 本 doc の前提を降格

**B の graphs 帰属測定 (6575tok・glru なし・既存バイナリ) の結果、 graph 系機構の decode tg への寄与は ≈0 (ノイズ内):**

| run | 設定 | tg |
|---|---|---|
| default | llama reuse ON + ggml capture ON | 13.13 |
| ggml capture OFF | `GGML_CUDA_DISABLE_GRAPHS=1` (common.cuh:1272 で実際に読まれている・review ソース検証済) | 13.80 |
| llama reuse OFF | `LLAMA_GRAPH_REUSE_DISABLE=1` (llama-context.cpp:278) | 12.67 |

- ggml capture 寄与 ≈ -0.67 (≈0)・llama reuse 寄与 ≈ +0.46 (≈0)。 ラン間分散大 (8.66-13.80) なので単点 delta はノイズ内だが、方向結論 (≈0) は両機構で一致。
- `GGML_HIP_GRAPHS:BOOL=ON` (build-hip CMakeCache) で capture 機構はバイナリ入り・arch ゲートも gfx1101 は通過 = 「capture がそもそも走っていない」場合も含め、**どちらにしても再有効化のヘッドルームは ≈0**。
- 注: B が当初「GGML_CUDA_DISABLE_GRAPHS は未読」と結論したのは誤り (review が common.cuh:1272 の getenv を確認)。 `graphs reused`/`graph nodes` は llama レベルのカンタ (llama-context.cpp:4217/:702) で HIP capture の証拠ではない。

**計画への影響:**
1. **Stage 2 (graphs 再有効化) は「4x の本体」から optional / 無退行チェックへ降格**。 本 doc §2 の設計自体は有効だが (graphs を使うならこの設計)、 優先度は Stage 1 のオーバーラップ品質より下。
2. **40 t/s は Stage 1 の pinned+async+早期 fire のオーバーラップに全乗り**: pinned 15-20 GB/s でも毎トークン miss 転送 ~366 MiB ≈ 19-25 ms = 隠さなければ 40-52 t/s が天井。 早期 fire で転送を compute の裏に隠せるかが実ゲート。
3. 方法論教訓 (#151 実例): FreeToken の「graphs が 4x」は彼らの hybrid CPU レーンの毎トークン host オーバーヘッドが前提。 llama.cpp ベースラインは host オーバーヘッドが小さく graphs 寄与 ≈0 = 機構の移植性は環境依存・必ず自環境で測る。
