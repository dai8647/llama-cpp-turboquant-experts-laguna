# 解析: q* 有効時 0.24 tok/s の原因特定 (review side, 2026-08-26 — GPU 不使用, ログ + ソースのみ)

対象: feat/qstar-r2-rebuild の trace run (port 8102)。 ログ `qstar-trace-8102.log.stderr` と r2 ソースの静的解析のみで特定した。 GPU も build も触っていない。

## 結論

遅いのは「q* の CPU 実行が重いから」ではなく、 **分割ポリシーが「ミスを全部 CPU へ」判定しているから**。 それはバグではなくコード通りの動作で、 デフォルト設定値同士の不整合が原因。

## 計算 (すべて実測値)

calibrate 実測: `h2d=3.2 GB/s / cpu=0.1 GB/s / expert=1.95 MiB / threads=4 / budget_us=300`

| 項目 | 計算 | 値 |
|---|---|---|
| expert 1 個の H2D 転送時間 | 2.04 MB ÷ 3.2 GB/s | **637 µs** |
| expert 1 個の CPU GEMV 時間 | 2.04 MB ÷ 0.1 GB/s | **20,400 µs** |
| 1 層 1 step の転送予算 | デフォルト固定 | **300 µs** |

判定式 (`src/llama-graph.cpp:349-352`):

```
transfer = t_xfer <= bytes/b_cpu * 1e9 && budget_used + t_xfer <= budget
```

- 条件 1 (転送が CPU より速いか): 637 ≤ 20400 ✓
- 条件 2 (予算内か): 637 > 300 → **最初の 1 個で既に予算超過 → 常に false**

→ ミス expert は**例外なく `qstar_deferred` (CPU 実行) へ**。 CPU は転送の 32 倍遅いため、 1 token あたり ~20 層 × r=8 × 20 ms ≈ 4 s/token ≒ **0.24 tok/s** と整合。 ログ裏取り済み: リクエスト中に `[QSTAR-CPUEXEC] entry: layer=0 r=8 ... bytes=16318464` (= 8 × 2039808) が実発火。 旧監査 ~13 t/s は `qstar_cpu=0` (実質全部転送) の環境であり、「q* が初めて本気を出した瞬間に壊れる」構造として一貫。

## 根本原因 2 層

1. **budget デフォルト 300 µs < expert 1 個の実測転送 637 µs** (`llama-model.h:760`, `llama-model.cpp:2506` ハードコード)。 このマシンの h2d では 1 expert も予算に収まらず、 予算条項が「転送禁止令」として機能している。 FreeToken 本家の設計意図では予算は許容レイテンシであり、 転送単価未満に設定される想定はない。
2. **cpu_bps 0.1 GB/s が異常に遅い** (bandwidth-bound GEMV の正常値は数 GB/s)。 最有力仮説: **ROCm 配布 clang ツールチェーンで ggml-cpu が native SIMD 無し (generic x86-64) でビルドされている** — Q4_K GEMV が ~0.3 GFLOPS しか出ない事象と整合。 次点: dedicated threadpool の wake/barrier オーバーヘッド (n_nodes=5 の mini graph では並列化効果も薄い)。

## 提案 (確認・修正の順序)

- **F1 (秒で試せる検証)**: budget を引き上げて同一リクエスト再実行 (`--moe-qstar-budget-us 30000` 相当)。 **予測: 条件 2 が通過し全 miss が転送路へ → 吐速が旧監査水準 (~13 t/s) に復帰**。 復帰すれば上記メカニズムが確定。
- **F2**: ggml-cpu のビルドフラグ確認 (`compile_commands.json` / CMakeCache で `-march` 系を grep)。 AVX2 が効いていなければネイティブフラグ追加で再ビルド → cpu_bps 再較定。 2+ GB/s が出れば CPU 経路が初めて現実的選択肢になり q* が意味を持つ。
- **F3 (ポリシー堅牢化, A 判断)**: (a) b_cpu が degenerate (例: <0.5 GB/s) なら `cpu_ok=false` 扱いにする、 (b) 予算条項で「速い経路 (転送)」を潰して「遅い経路 (CPU)」へ追い出す現状の式は本末転倒なので、 予算は転送側の同時投入数制限などに使う形へ見直し。
- **F4 (副次)**: h2d 3.2-3.7 GB/s も理論比低め (probe が bank tensor 毎の個別 copy)。 pinned 一括 copy での再計測も余裕があれば。

## 受入バーへの影響

バー ① (`qstar_cpu>0` のログ観測) は既に満たているが、 それは**「q* が暴走して全部 CPU に投げている」状態の観測**であって健全性の証拠ではない。 バー ②③④ の速度基準は現状設定では絶対に落ちるため、 **F1 (+可能なら F2) を入れた後の数字で B bench GO を出すべき**。 B 側の受入テンプレ (@b5c1eebd3) 自体は変更不要。
