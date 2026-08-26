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
