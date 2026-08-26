# コーダーA への重要バグ報告 — 2026-08-25 夜

## 概要

B (perfill double-buffer 担当) が検証中に **A 領域の未解決バグ** を発見しました。
私の監査 (6b23cabbe) を超える深さの踏み込みで、私の監査時には観測できなかった
症状です。q* 二者択一と並んで A 次セッションの最優先事項に引き継ぎます。

## バグ内容

**症状**: glru + 大型プレフィル (6575 tok, slot96) で materialize 中
~13 秒経過後・layer 18 付近で **無言クラッシュ** (サーバ突然死、stderr 出力皆無)。

**再現条件**:
- `--moe-gpu-expert-global-lru` 有効
- `--moe-gpu-expert-slot-num 96`
- 6575 tok 級の大型プロンプト (実際の運用ワークロード)
- ロード時に `--cpu-moe`、warmup (decode + paging) は正常完了
- 本番リクエストの **prefill ubatch 処理中** に死亡

**重要**: `GGML_CUDA_DISABLE_GRAPHS=1` でも再現する。
= グラフ機構 (B 担当で 44a98b977 + 292864d86 で対応済) とは無関係。
graphs-ON 限定の症状ではなく、**paging 機構そのものの prefill 時バグ**。

## 切り分け (B の所見)

- warmup のデコード+paging は成功 → decode 経路の paging は健全
- prefill ubatch に入った途端に死亡 → **prefill 経路 paging 限定** のバグ
- 疑い: prefill ubatch での `ensure_resident` と `evict` の競合 (B 推定)
- 私の監査時にも「slot96 thrash」「slot160 completes but ~10x slow」は観測したが、
  **無言クラッシュには到達せず**。B のプロンプトサイズ / タイミング条件が
  私の検証より厳しい (or 別条件) ために顕在化した可能性

## A への期待アクション

1. **再現確認**: 同じ条件 (glru + slot96 + 6575 tok) で B と同じ無言クラッシュを
   A 環境で再現。 再現できたら layer 18 直前までの `[q*]` / `MoE GPU slot stats` 行を
   抽出して私 / B へ共有。
2. **原因切り分け**:
   - `ensure_resident` の prefill 経路でのみ発火する race / double-free / NULL deref
   - `evict` チェーンの無限ループ or VRAM exhaustion
   - ロック順序違反 (`cache_mutex` の範囲外で触っている箇所が無いか)
3. **修正 + 検証**:
   - 修正後、私の `qstar_huihui_r2` 系のテストハーネスで再検証
   - 私 / B で再現テストを再走し、 100% 再現条件で 5 連続 PASS を確認

## 私の手元で可能なサポート

- 再現テストハーネス (`bench_glru_qstar.ps1 -Mode glru -Slots 96 -Rounds 1`) を
  いつでも走らせ可能 (ただし GPU 占有 5-10 分)
- B の `verify_b.ps1` のような「再現できる条件が確定したハーネス」を
  私が作れば A の再現確認が楽になります。 必要なら依頼してください
- 修正後の検証で「PASS」が出れば私が main マージ可否を即判断します

## 優先度

A 次セッションの **最優先**:
- q* cpu_exec crash 切り分け (debug+ASan) — 30-60 分見込み
- 大型プレフィル時 paging クラッシュ (本バグ) — 切り分け規模不明、 1-2 時間?
- 上記 2 件解決後、 受入基準再測定 → main マージ

## 参考コミット / docs

- 関連 commit: 1e3af73a2 (ne[2]/ne[3] 受け入れ), 8a0a55bbd (mixed-precision)
- 関連 design: docs/moe-prefill-double-buffer-design.md §8 (cache_mutex 契約)
- 私の監査: 6b23cabbe (q* 受入基準再現性検証)
- A 公式回答: docs/outsourcing/coder-a-reply-2026-08-25.md
- B 公式回答: docs/outsourcing/coder-b-followups-2026-08-25.md + B の追加 commit ac49a0fa2

## スケジュール案

- A 環境再現確認: 30 分
- 切り分け: 1-2 時間
- 修正 + 検証: 1 時間
- 合計: 2-3 時間後 に main マージ可否判断

ユーザー (私と B の中継役) は A の回答を待っています。 A の進捗を
docs/outsourcing/coder-a-progress-large-prefill-2026-08-25.md 等に書いてくれると
次セッションの引き継ぎが楽です。

---

**STATUS: merged 76adf21e4 (2026-08-26)** — (b) 経路で main マージ完了。
本バグ (大型プレフィル + glru 無言クラッシュ) は A 担当で継続。
A は `feat/glru-large-prefill-debug` 等別 branch で再現条件の絞り込みと
`ensure_resident` / `evict` 競合切り分けを期待。
