# B へ: 待ち時間タスク — bench 即応パッケージ (GPU 不使用) (review side, 2026-08-26)

STATUS: standby 解除ではない。 A の r2 バイナリ (`feat/qstar-r2-rebuild`) がまだ存在しないため **実行は一切しない**。 GPU も build-hip も触らない (単一所有者ルール coder-a-build-coordination @d2016c9dc 継続)。 スクリプトと doc の準備のみ。 全タスク GPU フリー。

## Task 0 (確認): 大型プレフィル再現手順 doc の進捗

coder-b-repro-prep-2026-08-26.md (@49b11ee0e) の件。 完了済みならコミットハッシュの報告だけで OK、 未完ならこの待ち時間に仕上げて push。 A は現在 q* 優先なので急ぎ度は変わらず「低〜中」だが、 完っておくと A が詰まった時の保険として機能する。

## Task 1: bench_glru_qstar.ps1 に -BinaryPath パラメータ追加

現状スクリプトは main バイナリ前提だが、 `-Mode qstar` は `--moe-qstar` を直接渡すため main バイナリでは実行不可 → A の round-2 ブランチバイナリを指せる必要がある。

- `-BinaryPath <path>` パラメータを追加 (省略時は現状動作を完全維持)
- ハードコードローカルパス慣行はそのまま (repo 慣行通り)

## Task 2: 受入バー実行テンプレ doc

固定の受入バー ① `qstar_cpu>0` 観測 ② 短文 3 プロンプト ×3 ラウンド・r1 破棄 ③ 長文 6575 tok 完走 (REQUEST-FAILED=不可) ④ Ornith 同条件 ⑤ env 記録、 をそのまま埋められる結果テンプレを作る:

- 結果表: モデル × プロンプト × ラウンドの t/s + `qstar_cpu` 発生回数
- 比較列に旧監査値を事前記入: **13.78 / 16.81 / 16.76 / 13.43 / 13.07 t/s** (撤回根拠ラウンド)
- env 欄: ビルドハッシュ / warmup 有無 / graphs 状態 / `LLAMA_MOE_QSTAR_STATS`
- 短文 3 プロンプトの文面をこの場で確定 (互いに別トピック・同程度の長さ)
- Ornith 用起動引数案 (モデルパス以外は Huihui Q4_K と同一条件)

## Task 3: qstar_cpu 集計レシピ

`LLAMA_MOE_QSTAR_STATS=1` の stderr ログから `qstar_cpu` 発生回数と分布を集計する grep/awk 一式を doc 化。 受入判定①の自動化。

## 完了条件

Task 1–3 の commit+push (docs/outsourcing/ 配下)。 **GO トリガーは変更なし**: A が `qstar_cpu>0` + 受入パスを報告した時点で review 側から B へ実行指示を出す。 それまでは実行しない。
