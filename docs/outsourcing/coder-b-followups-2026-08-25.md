# コーダーB への依頼 — 2026-08-25 夜

## 受入基準 全達成 (A 報告 2026-08-25)

| 基準 | ターゲット | 実測 | 余白 |
|---|---|---|---|
| Huihui 短文単発 (qstar+slot96) | ≥12.38 t/s | 13.78 / 16.81 / 16.76 (avg 15.78) | +27% |
| Huihui 長文後デコード (qstar+slot96+B prefill) | ≥11.84 t/s | 13.43 (pp=23.0, 6575 tok) | +13% |
| Ornith 低局所性 (qstar+slot96) | ≥12.4 t/s | 13.24 / 13.01 / 12.96 (avg 13.07) | +5.4% |

feat/prefill-double-buffer 4 commits + 私の逆マージ 8fa0959bc は origin に push 済み。
docs/frequency-placement-findings.md の add/add 衝突は解消(補足セクション方式)。
main へマージ予定(現状 main HEAD = d46a2a887、衝突ゼロ確認済)。

## B にお願いしたい 2 件

### 1. ggml_cuda_graph_check_compability への検出追加(優先度: 中)

**背景**: 現在 q*/global-LRU paging を graphs-ON で動かすと `b54973372` の警告ログが出る
だけで、HIP graph capture がランタイムでコケる。一時凌ぎとして
`GGML_CUDA_DISABLE_GRAPHS=1` を運用しているが、q* 経路は eval 時に residency が
動的に変わるので captured graph は本質的に使えない。

**お願い**: `ggml-cuda.cu` の `ggml_cuda_graph_check_compability` に、remap op
(`GGML_OP_MAP_CUSTOM1` または独自の op type 識別子)を検出して、capture 対象
cgraph に含まれていたら警告ではなく **明示的に false を返す** か、
警告対象 op を含む sub-graph は capture から除外する経路を実装してください。

これで運用が「警告ログに怯えながら env を手動設定」から「Q* + graphs が普通に動く」へ
移行できます。検出ロジックは A 側の op type 命名と相談して決めてください。

### 2. q* path × prefill-PF の cache_mutex 順序取り決め(優先度: 低)

**背景**: A 静的レビューで `cache_mutex` が host GEMM 中に保持されるため
HOL blocking が判明。q* 経路 1 ユーザー(n=1 検証)では問題なしだが、
B の prefill_PF インフライト H2D が同 mutex を取る場合、デッドロックや
意図しないシリアル化のリスクがある。

**お願い**:
- 現状(B 側 PF 実装)の `cache_mutex` 取得範囲を教えてください(per-layer / per-expert / per-step?)
- q* 経路が host GEMM 中に持つ mutex を B 側 PF が避けるか、B 側 PF が
  q* 経路の host GEMM を回避するかの **どちらの方針を取るか** を
  決めてもらえると、A が spec を固められます。

n=1 検証範囲では priority: low ですが、n=2+ へ拡張する前に決めるべき論点です。

## 報告待ち

- (1)/(2) の対応可否とスケジュール
- ご自身の `ggml/tests/capture_probe*.cu` と `vb_last.json` を main に取り込むか
  (現状 feat worktree に未追跡で残っています。残すなら理由を、要らなければ削除します)
- 次サイクル(ハイパーパラメータ sweep・auto sizing の自動調整等)の方向性

## 監査で気になった点(参考、修正不要)

- `bench_results.txt` に A が追記した受入基準ログ(7行)が未コミット。
  私の PR コミットに含めるか、A 側で別途 commit してください。
- Huihui 短文 1 回目 13.78 t/s と 2-3 回目 16.81/16.76 t/s の乖離が
  大きく見えます(初動 vs 定常)。ウォームアップ 1 発目特有の
  q* 経路 cold-cache 性かとは思いますが、再現性確認できると安心です。
