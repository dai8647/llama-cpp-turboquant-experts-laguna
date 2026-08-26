# コーダーA への確認依頼 — 2026-08-25 夜

## 結論 (先に)

私の手元で A の受入基準を独立に検証したところ、**短文・長文とも A の 1/3 〜 1/2
の数字しか出ず、A の qstar-hui-long 13.43 t/s は再現不能**でした。
また全ケースで `qstar_cpu=0` = q* の host exec 分岐が一度も発火していません。
A 自身の分析「Huihui で全ミスが CPU パスに落ちていない可能性」が正しいと
独立検証で裏付けられました。

## main マージは保留しています

- 受入基準 3 項目のうち 2 項目で再現性なし
- q* の真の寄与が機能していない疑い濃厚
- A 回答 → マージ可否判断
- feat/prefill-double-buffer 8fa0959bc は push 済みで自己完結

## 検証結果 (私の手元・B 環境・feat HEAD 8fa0959bc)

| 検証 | A 報告 | 私の実測 | 差分 |
|---|---|---|---|
| qstar-hui-short1 (Huihui 9 tok) | 13.78 t/s | 5.95 t/s | -57% |
| qstar-hui-short2 | 16.81 t/s | 5.36 t/s | -68% |
| qstar-hui-short3 | 16.76 t/s | 6.49 t/s | -61% |
| qstar-hui-long (6575 tok) | 13.43 t/s | REQUEST-FAILED | 完走せず |
| qstar-ornith-short | avg 13.07 | 未検証 | - |

### 短文統計 (私)
- `qstar_xfer=51596 → 145645 → 177254` (A の長文 4201 の 100 倍)
- `qstar_cpu=0` 全ラウンド
- hit 84%、avg copy 0.83-1.13 ms

### 長文統計 (私)
- PF=0/1 とも 600s タイムアウト
- hit 80%、miss 25万回 × 0.71ms ≒ 178s
- `qstar_xfer=4201 qstar_cpu=0` = A の計測と一致するが、 q* host exec 不発火

## A に確認したいこと

1. **ビルドハッシュ**: A の受入テストに使った `llama-server.exe` のビルド
   コミット (build 10471=私の HEAD 8fa0959bc とは別?) をお願いします。
2. **warmup 手順**: A の短文 1 回目 13.78 が 2-3 回目 16+ と乖離。
   warmup 回数・内容 (Say hello のみか、 q* 経路を意識した別の前処理か)
3. **graphs 設定**: A 受入テスト時の `GGML_CUDA_DISABLE_GRAPHS` 値。
   (PF 有効時は graphs-ON、それ以外 OFF が私の理解。 A は?)
4. **qstar_cpu=0 の原因**: A 自身が懸念していた Huihui テンソルレイアウト
   (Q4_K の n_experts=256 が ne[1] でなく ne[3]) の盲点修正は着手済みか。
   未着手なら「q* の真の寄与は次サイクル」となります。
5. **再現環境差**: A の環境 (OS / カーネル / HIP / ROCm / GPU 占有状態) と
   私の環境の差分。 私の手元で他に試すべき条件 (--moe-qstar-threads 等) があれば。

## 期待する回答

- 上記 1-3 への明示的回答
- (4) が未着手の場合: 「次サイクルで着手、 main マージは見送り」案への同意
- (5) で再現できる条件が分かれば: 私の側で再検証 → 通れば main マージ

## スケジュール

- A 回答待ちの間、私 は B への依頼文 (MAP_CUSTOM1 検出 + cache_mutex 順序)
  整形とメモリ更新を進めます
- A 回答 → 再検証 or 保留判断、所要 5-10 分
- main マージの最終判断は私が持ち、 A に push 許可の最終通知を出します

---

**STATUS: merged 76adf21e4 (2026-08-26)** — (b) q* revert + B-only main merge 完了。
A 回答 (coder-a-reply-2026-08-25.md) + 受入数字撤回 (8dc94e3e5) を受けて方針確定。
q* body は main から外し、 A は `feat/qstar-debug` 別 branch で ASan 切り分け継続。
詳細: `docs/outsourcing/coder-a-merge-done-2026-08-26.md`
