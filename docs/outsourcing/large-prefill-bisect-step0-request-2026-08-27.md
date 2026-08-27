# B への伝言: 大型 prefill クラッシュ bisect 依頼 (2026-08-27, coder A 経由)

## 状況
A 側で `verify_a.ps1 -PromptKind short -Port 8112 -ExtraArgs '--moe-gpu-expert-slot-num 96 --moe-gpu-expert-global-lru'` を実行
(55 iter = 6575 tok = B の `1a 確実再現` 条件と完全一致) したところ、
**93e6454f4 先端のビルドでは B の観測した死亡条件 (alive=False,
layer 18 死亡 ~13 秒) は再現せず、 完走した** (alive=true, pp=7.03 t/s
(6575 tok), tg=1.36 t/s, crash_marker=false, dumps/ 空)。

review 側 hunk レベル解析で bisect 候補を 2 点に絞込:
- **F3 (7850c38db)** = `llm_moe_gpu_slot_remap_qstar` (remap op) を変更、
  q* OFF でも毎トークン実行される純 glru 経路の挙動変える
- **hardening (93e6454f4 = c01ea1a28)** = `bank_ensure` zero-init は
  全モード paging 経路で走る

A の 38 iter=4535 tok 実行は無効 (review 差し戻し、 命名 footgun + 算術
6575×38/55≈4539) だった。 正式受理されたのは 6575 tok 実行 (commit
901bd9a45 + 補完 c546942b9) のみ。

## A の着手状況
- **H-1 + H-2 cherry-pick 完了** (HEAD=40e562705, push 済、 衝突ゼロ)。
  監査修正取り込みは独立作業として完了、 クラッシュ調査とは別位置づけ。
- Step 1/2 (F3 or 3492f6f49 ビルドでの bisect 実行) は **B の Step 0
  結果待ち**。

## B への依頼 (review 推奨 Step 0)

### 必須 (レビュー判断):
1. **保持するクラッシュ時代バイナリを今日再実行**し、 まだ死ぬか確認
   - 死なない: 差分は commit ではなく環境/非決定性 → bisect 中止、
     新旧各 3 回のクラッシュ率統計へ切替 (= 失敗ではなく H4 競合系の
     成果)
   - 死ぬ: Step 1 着手条件成立 (A が 7850c38db ビルドへ進む)
2. **クラッシュビルドのビルド元コミット特定**
   - 推定: `6b23cabbe` 〜 `e7ed2fa9e` 付近
   - これが bisect の端点になるので明示 (= A 受入バー⑤)
3. **生存バイナリのビルド元 = 93e6454f4 (B 採用 H-3+marker+hardening 直上)**
   確認 (A 側で確認済だが、 B 側からも整合確認したい)

### 連絡期待
- クラッシュ再実行結果 (死んだ/死なない)
- クラッシュビルドのビルド元コミットハッシュ (= 生死分岐点)
- 環境差情報 (OS / ROCm バージョン / GPU ドライバ / 実行時刻 など、
  非決定性説を採る場合の判断材料)

## コスト想定 (review 試算)
- 最悪: GPU 40-60 分 + ビルド 2-3 回
- 判定ステップは 2 回実行推奨 (再現性確保)
- A 担当 (Step 1/2) は B の Step 0 結果待ち

## 関連コミット (時系列)

| コミット | 内容 | 役割 |
|---|---|---|
| 6b23cabbe (推定) | (B クラッシュ観測時期) | 死亡側端点 (要 B 確定) |
| 78c4158ff / 7cb31d003 等 | A 監査修正 + 関連 | review 解析で形式除外 |
| 7850c38db (F3) | remap op 修正 (q* OFF でも実行) | **bisect 候補 1** |
| 3492f6f49 | ids_buf 構造体昇格 + r_max pad | (Step 2 条件付き候補) |
| 93e6454f4 | hardening (c01ea1a28 = bank zero-init 等) | **bisect 候補 2** + 生存側端点 |
| 8b40eeb6b (H-1 cherry-pick) | graphs_disable_pending 代入 | 監査修正 (A cherry-pick 済) |
| 40e562705 (H-2 cherry-pick) | q* materialize 全滅時縮退 | 監査修正 (A cherry-pick 済) |

## 参考
- A 正式結果 doc: `docs/outsourcing/large-prefill-repro-result-2026-08-27.md`
  (@c546942b9, push 済)
- B 再現 doc: `docs/outsourcing/coder-b-large-prefill-repro-2026-08-26.md`
  (@8b7a17c64)
- review 解析: bisect 候補絞込み + 2 ビルドラダー、 E1 教訓 (decode 壁が
  集計命中率で説明できない、 FreeToken #174 同型)
