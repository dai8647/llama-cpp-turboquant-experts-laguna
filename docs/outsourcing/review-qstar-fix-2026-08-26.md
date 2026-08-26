# レビュー: q* calibrate クラッシュ修正 (ids_buf) への注記 (review side, 2026-08-26)

対象: feat/qstar-r2-rebuild ワーキングツリーの cpu_exec ids 修正。 根本原因特定を承認する — 「src0/src1 は正常なのに MUL_MAT_ID 内で即死」のシグネチャに対する説明 (未初期化 ids 7 個分 = ゴミ expert index での weight 読み) は観察事実と完全整合であり、 数日悩んだ calibrate 3-loop 死の決着として妥当。

## 1. パッチ自体は正しい。 ただし寿命の設計がスコープ依存

`src/llama.cpp` 実読の結果:

```cpp
std::vector<int32_t> ids_buf(ex.r_max);
...
ex.t_ids->data      = ids_buf.data();
ex.t_ids_down->data = ids_buf.data();
```

`ids_buf` はスタックローカルで、 そのヒープポインタを `qstar_exec_by_layer` の永続構造体に保存している。 現状は「同一呼出内で graph_compute が完結し、 次回呼出が必ず .data を上書きする」前提でしか安全ではない。

**推奨**: ids バッファを `llama_moe_qstar_layer_exec` のメンバ (`std::vector<int32_t> ids_pad`) に昇格し、 `memcpy` で埋める (隣の `t_x` と同じ扱いに揃える)。 ポインタ差し替えが消えて寿命問題が構造的に消えるうえ、 将来このパスを非同期化する時の地雷も除去できる。 r=0 時の `experts[r-1]` (=-1) 参照についても 1 行ガードか注記があると尚良。

緊急度は低い (現状動作は確認済み) が、 受入前に直すなら数行で済む。

## 2. 0.24 tok/s をそのまま受入バーに持ち込まないこと

旧監査 ~13 t/s は `qstar_cpu=0` の実質 plain paging。 今回初めて host exec が本当に出たので速度が落ちるのは当然だが、 0.24 は「遅すぎて効果測定にならない」水準。 バー① (qstar_cpu>0) とは別に、 バー②〜④のベンチ前に分解計測を推奨:

1. ログから `[QSTAR-CPUEXEC]` 発火回数と平均 r を集計 (B のレシピ @3dee9cac9 がそのまま使える) — CPU に流れる量が妥当か
2. `--moe-qstar-threads` 4 → 8/16 に上げて同リクエスト再実行
3. 同一環境 (slot32 + glru + PF) で `--moe-qstar` 無しの対照走行 — q* 有無の差分だけを見る

calibrate が選ぶ分割が妥当でも、 threads=4 では CPU 側が律速になり分裂の意味が潰れる。 対照走行なしに「q* は遅い」と結論しない。

## 3. コミット規律

ids 修正 (根拠 fix) と [QNODE]/calibrate 計測 (計装) は**別コミット**で早めに push。 特に ids 修正は今回の主目的の決着点なので、 単独コミット + メッセージに死亡シグネチャと原因を書いておくことで後日の監査性が変わる。

## 4. 次の順序

1. build6 で calibrate 結果行を出しバー①を確定 (qstar_cpu>0 のログ観測)
2. §2 の分解計測 (発火集計 → threads 引き上げ → 対照走行)
3. 数字が揃ったら受入報告 → review 側から B へ `-Mode qstar` の GO

(大型 prefill glru クラッシュは本件と別物・継続案件。 gflags 判別手は review-large-prefill-hypotheses-2026-08-26.md のまま有効)
