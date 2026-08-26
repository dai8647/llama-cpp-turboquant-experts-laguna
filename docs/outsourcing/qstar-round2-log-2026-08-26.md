# q* round-2 進捗ログ (2026-08-26)

## 着手 (Go from B)
- Bから go サイン受領 + 注意3点 + 大型prefill+glru クラッシュのA担当移管
- `feat/qstar-debug-round2` ブランチ push 済 (追跡設定)
- 関連: `docs/build-issues/rocm-71-msvc-attribute-pure.md` (B既知のmath_fwd.h問題)作成

## 受入バー (B 指示、固定)
1. `qstar_cpu > 0` をログで観測 (ゼロなら不採用)
2. 短文: 別プロンプト 3 種 × 3 ラウンド・1 回目破棄
3. 長文 6575 tok 完走 (REQUEST-FAILED 不可)
4. Ornith 同条件通過
5. 測定環境 (ビルドハッシュ / warmup / graphs 状態) を結果 docs に明記

## 担当拡大 (B から移管)
- 大型プレフィル + glru クラッシュ (B 発見、layer 18 無言死亡)
- 同一ブランチ `feat/qstar-debug-round2` で進捗 docs に逐次記録
- 単独再実装の場合は §8 mutex 契約 + §9 graphs スイッチ前提の設計レビュー先行

## ステップ 1: debug ビルド計画
- ROCm 7.1 + MSVC の math_fwd.h C2059 罠 → 別ディレクトリ `build-hip-debug` で
  クリーンリビルド試行
- 失敗したら `-O0 -g` + printf trace にフォールバック (B既知の回避策)
- まずは q* cpu_exec 1要素クエリの単独ハーネス (B 推奨の最小再現)
  を作ってサーバ起動を跨がない形にする

## ステップ 2 予定
- `991bf3042` (q*本体) を main から cherry-pick
  → 6c4f1c9b3 (plan) + 991bf3042 (body) + 1e3af73a2/8a0a55bbd (layout fix)
    の 4 コミットをスタック
- ただし 991bf3042 は A 独自実装、 main 側の `a414acc7f` で395行drop済
- 先に小さな再現関数 `qstar_cpu_exec_one()` を main に移植して
  死亡サイトを切り分けてから本体を戻すのが安全

## 既知の障害 (前回の症状)
- engine build は通る (layer 0 ready OK)
- calibrate 内 3回 qstar_cpu_exec ループでサーバ死亡
  - Huihui Q4_K mixed type=14+12
  - Ornith unified type=11
  - threads=1 にしても症状変わらず
  - threadpool alloc failed 警告は出ない
  - → threadpool は作れている、 compute 初回で死亡

## スケジュール目安
- step 1-2 (debug ビルド + 単独ハーネス): 30-60分
- step 3 (根本原因特定): 30-60分
- step 4 (修正 or revert 決定): 30分
- 大型 prefill+glru クラッシュ: 並行で着手、1-2h

## revert 判断の文書化方針
- 機能 revert も成果
- 理由 (compute path 別バグ等) を `docs/outsourcing/qstar-revert-reason-2026-08-XX.md` に書く
- main への戻し方 (cherry-pick -x 等) を明記

## ステップ 12-13: calibrate クラッシュ根本原因修正 + 性能分解 (2026-08-26, coder A)

### 12a. 死亡ノード特定 ([QNODE] trace)
`ggml-cpu.c` node loop に env-gated `[QNODE]` トレース追加 (LLAMA_QSTAR_NODE_TRACE=1)。
実行ログ (死亡時, 最終行):

```
[QSTAR-CALIB] rep=0 entering
[QSTAR-CPUEXEC] entry: layer=0 r=1 n_embd=2048 qstar_tp=0000000000000000 threads=4
[QSTAR-CPUEXEC] before ggml_graph_compute: n_nodes=5 work_size=39832 bytes=2039808
[QNODE] n=0/5 op=30 MUL_MAT_ID ne=[512,8,1] data=000002535D5D7FE0 src0data=00000253B724B3A0 src1data=000002535D5D5FE0
(プロセス消失 — node 0 の mul_mat_id カーネル内で即死)
```

### 12b. 根本原因 (3 層, コミット順)
1. **scratch dangling** (78c4a0486): `owned_tensor`/`own_node` ヘルパーが `mem.resize()` を
   テンソル毎に呼び、先に配った data() ポインタが realloc で全滅。
   → offset 集計 + 最後に resize 1 回 + 一括 rebind に書き換え。
   加えて view (gate_v/up_v/h_cols) は allocator 不在で data==NULL のまま → view_fixups で親 storage へ明示 bind。
2. **ids バッファ未初期化 = 本命の即死因** (3492f6f49): `cpu_exec` 内のローカル
   `std::vector<int32_t> ids_buf(r)`。r=1 でもグラフの t_ids_up は [r_max=8] で、mul_mat_id
   カーネルは id スロットを全部読む → 未初期化ヒープ 7 個分がゴミ expert index として
   weight を読みに行き node 0 で segv。→ ids_buf を layer_exec メンバに昇格
   (寿命問題も解消, B レビュー指示通り)、r_max 分確保して余りは最後の要求 id で pad。
   data バインドは exec_build に一元化し cpu_exec の per-call swap は廃止。
3. **calibrate 結果行がログに出ない**: LLAMA_LOG_INFO が早期ロード中に握り潰されるため
   `[QSTAR-CALIB] result:` fprintf を追加。

### 12c. 検証結果 (Huihui-Qwen3.6-35B-A3B Q4_K, slot32+glru+qstar threads=4)
```
[QSTAR-CALIB] rep=0/1/2 done ×3 → warmup loop exited
[QSTAR-CALIB] result: h2d=3.6 GB/s cpu=0.1 GB/s expert=1.95 MiB threads=4 budget_us=30000
(実リクエスト) [QSTAR-CPUEXEC] entry: layer=38/39 r=8 ... after ggml_graph_compute: status=0
```
クラッシュ解消 + 受入バー① (qstar_cpu>0 ログ観測) 到達。

### 13. 速度分解 (B 分析 review-qstar-slowness-analysis-2026-08-26.md の検証)
同一リクエスト ("Reply with exactly one word: hello", max_tokens=8, temp=0):

| 構成 | eval tok/s | 備考 |
|---|---|---|
| q* ON budget 300 (デフォルト) | 0.73 | B 分析通りの CPU 追い出し |
| q* ON budget 30000 | 1.45 | F1 実施。予測 ~13 t/s には届かず |
| q* OFF 対照 graphs 無効 | 4.34 | |
| q* OFF 対照 graphs 有効化試行 | 4.50 | **無意味だった**: glru はコード側で |

F2: `compile_commands.json` 全 obj に `-march=native` 付き → **AVX2 無効説は否定**。
cpu_bps 0.1 GB/s の別説明を発見: **測定過小バグ** — calibrate GEMV は ids pad により
r_max=8 experts 分計算するのに bps は r=1×expert_bytes で割るため真値の約 1/8 を表示。
真値推定 ~0.8-0.9 GB/s (threads=4)。表示修正は round-3 候補。

新発見: 起動ログ `q*/global-LRU expert paging is incompatible with CUDA graph capture;
set GGML_CUDA_DISABLE_GRAPHS=1` — glru 使用時はコード自身が graphs 強制無効。
graphs 条件は常に「無効」で統一されており対照比較としては公平。

**転送路理論上限の試算 (glru コピー経路限定)**: MoE 層 ~36 × r=8 × expert 2.04MB ÷
h2d 3.6GB/s ≈ 163ms/token → **この経路の**天井 ~6 t/s。budget 30000 の実測 1.45 t/s
(オーバーヘッド込み) と整合。※全 miss expert を H2D コピーする glru paging 経路に限った
上限であり q* 全体 (CPU 直計算経路含む) の上限ではない。
監査 tg=12.91 は `--cpu-moe`・glru 無しの CPU 直計算経路の記録なので、素ベースライン
判定は B の stage-a 完全再現 bench (P1) 待ちで行う。

### 受入バーへの影響と次手
- バー① は観測済みだが cache 冷 (全 miss) 状態の観測であり健全性の証拠でない (B 指摘通り)。
- F3 (ポリシー堅牢化: b_cpu degenerate 判定 / 予算条項の転送追い出し見直し) と
  F4 (pinned 一括 copy 再較定) が次の本丸。cpu_bps 表示バグ修正込み。
- 大型 prefill+glru クラッシュ調査は本 crash fix の上で再開予定。

### コミット
- 78c4a0486 moe : fix q* scratch dangling + bind view data + env-gated [QNODE] trace
- 3492f6f49 moe : own q* expert-id buffer in layer exec, pad to r_max, log calibrate result
- 両方 push 済 (feat/qstar-r2-rebuild)

### F3 実装 + 検証 (2026-08-26 夜, commit 7850c38db)
- **予算条項を分割判定から削除** (llama-graph.cpp): budget 超過 → 転送拒否 → CPU 追い出し
  という倒錯を解消。deferred は「host GEMV が H2D copy に本当に勝つ時だけ」。
  budget は [q*] stats 行の budget_left 表示として統計存続。
- **degenerate guard**: b_cpu < 0.5 GB/s なら cpu_ok=false (毒された EMA が host 経路へ
  流すことを構造的に禁止)。
- **cpu_bps EMA の分母修正** (llama.cpp): ids pad により r_max 列全部計算するため
  bytes を r_max ベースに。従来は ~r_max/r 倍過小 (表示 0.1 GB/s ↔ 実測 1.1 GB/s)。

検証 (Huihui Q4_K, slot32+glru+qstar threads=4, budget デフォルト 300):
```
[QSTAR-CALIB] result: h2d=3.8 GB/s cpu=1.1 GB/s expert=1.95 MiB threads=4 budget_us=300
同一リクエスト eval: 0.73 → 1.54 tok/s、推論中 [QSTAR-CPUEXEC] 発火ゼロ (calibrate 3 回のみ)
```
= 全 miss が設計通り転送路へ。対照 4.50 t/s との残余差分は remap op 内の同期コピー
コストで、glru-slot160 監査 (tg≈1.1-1.3) と同じ壁 = 次フェーズ (非同期 overlap) 領域。

### コミット追記
- c2911218d docs : 天井 ~6 t/s を glru コピー経路限定に表現修正 / 監査 tg=12.91 は
  --cpu-moe・glru 無し CPU 直計算経路の記録、素ベースライン判定は B の stage-a 再現待ち
- 7850c38db moe : q* split policy robustness (F3)
