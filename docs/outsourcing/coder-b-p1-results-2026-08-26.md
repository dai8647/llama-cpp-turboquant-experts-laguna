# P1 ベースライン決着ベンチ — 結果報告 (coder B, 2026-08-26 夜)

実施者: コーダーB / 仕様: review-p1-baseline-decisive-bench-2026-08-26.md (@d3663914e)
バイナリ承認: @1994008ee (`build-hip/bin/llama-server.exe` 2026-08-26 15:50 ビルド = F3 版 7850c38db 相当)
ランナー: `p1_stage_a_bench.ps1` (@f8b4b14e0 + -lv 追加、本コミット)
測定プロンプト: 受入テンプレ (@b5c1eebd3) の確定短文 P1-P3 ×3 ラウンド (r1 破棄) + 長文 6575 tok

## env 記録 (受入⑤)

| 項目 | 値 |
|---|---|
| バイナリ | build-hip/bin 15:50 F3 版 (= 7850c38db 相当, 承認済み) |
| warmup | 有 ('Say hello.' n_predict=16 自動 + 各プロンプト r1 破棄) |
| graphs | arm a/c = OFF (`GGML_CUDA_DISABLE_GRAPHS=1`) / arm b = ON (env 削除のみ) |
| LLAMA_MOE_QSTAR_STATS | 未設定 (q* 無効構成)。代わりに `[QSTAR-]` 行数=0 を全アームで検証 |
| ctx / port | arm a/b = 32768 / arm c = 8192, port 8091 |

## 結果一覧 (tg t/s・128 tok 生成)

### arm a — stage-a 完全再現 (slot96, glru/q* 無し, graphs OFF) — **mandatory**

2 回実施 (初回は証跡取得前に INFO が抑制されることを発見したため lv4 付きで再実行)。

| プロンプト | r1 (破棄) | r2 | r3 |
|---|---|---|---|
| P1 | 12.95 / 12.31 | 13.25 / 12.66 | 13.00 / 11.10 |
| P2 | 13.20 / 12.62 | 13.71 / 12.82 | 13.46 / 13.05 |
| P3 | 13.57 / 1.57† | 13.51 / 11.89 | 13.53 / 11.88 |

kept 6 値の短文中央値: **第1回 13.48 / 第2回 12.28 t/s**
長文 6575 tok: 第1回 pp=188.0 / tg=13.37 完走, 第2回 pp=164.5 / tg=12.33 完走

† 第2回 P3 r1 のみ一過性の 1.57 t/s (直後の採用ラウンドで即回復)。破棄ラウンドにつき判定に影響なし。

### arm b — 同構成で graphs ON (optional)

| プロンプト | r1 (破棄) | r2 | r3 |
|---|---|---|---|
| P1 | 13.79 | 13.98 | 13.78 |
| P2 | 13.85 | 13.23 | 12.54 |
| P3 | 14.00 | 14.16 | 14.15 |

短文中央値 **13.88 t/s** (全アーム中最速)。長文 pp=197.8 / tg=13.67 完走。

### arm c — A 対照推定構成の再現 (slot32 + glru, -c 8192 -t 8, graphs OFF) (optional)

glru 有効化ログ確認済み: `global LRU slot pool enabled (32 slots shared across layers)` +
`initialized MoE GPU expert slot cache with 32 slots`。

| プロンプト | r1 | r2 | r3 |
|---|---|---|---|
| P1 | 1.45 | 1.47 | 1.45 |
| P2 | 1.38 | 1.45 | 1.43 |
| P3 | 1.35 | 1.38 | 1.36 |

短文中央値 **1.44 t/s** (デコード ~690ms/tok = 同期 H2D コピー支配のスラッシング経路)。
長文は prefill **1.20 tok/s** (2048/6575 tok 時点) で実質停止のため B 側で切断
(監査時の「glru + 長文はタイムアウト」前例と同型。完走見込み ~90 分のため GPU 節約)。

## 証跡 (review doc §3 必須項目)

- `load_tensors: MoE GPU expert slot mode enabled; ignoring n_gpu_layers for MoE placement`
- `llama_model_load: initialized MoE GPU expert slot cache with 96 slots (requested 96)`
  → **ゲート無効化 (B0 前例) は発生していない**。slot96 は実働
- `[QSTAR-*]` 行数: **全アーム 0** (最終防波線クリア、q* 偶発有効化なし)
- 注意: F3 バイナリの新ロギング系は既定 verbosity で INFO を抑制する
  (verbosity=3 初期表示を確認)。slot 有効化行・stats 行の取得には `-lv 4` 以上が必須。
  ランナーに `-Lv` パラメータとして追加済み (既定 4)
- 小事後発見: `LLAMA_MOE_SLOT_STATS=1` でも lv4 で `MoE GPU slot stats` 行が出ない
  (出力条件が変わった?)。活性証明としては上記初期化ログで十分だが、要観察

## 判定 (review doc §4 表)

**短文中央値 12.28–13.48 t/s = ≥11 バンド → 「PASS: 退行なし」**

1. **監査 stage-a 12.91 系は今日の F3 バイナリで健在** (むしろ graphs ON なら 13.88)。
   r2-rebuild/prefill-double-buffer 系にコード退行はない。
2. **A の「~4.5 = この環境の素ベースライン」は否定**。対照構成 (glru 付き) の実測は
   さらに低い 1.44 t/s であり、4.5 が素ベースラインである可能性は消えた。
   グラフ ON/OFF は結果をほぼ左右しない (12.28-13.88) ので、A の「graphs on 試行 4.50」は
   別要因 (対照構成混入等) の疑いがさらに強まった。
3. **天井 ~6 t/s 試算の適用範囲限定が確定**: CPU 直計算経路の実力は ~12-14 t/s。
   q* の価値基準は「glru スラッシング 1.4 t/s → どれだけ持ち上げられるか」となり、
   受入ライン引き直し (P2) への材料が揃った。
4. 前提の置き場所: stage-a は「ミス expert をコピーせず CPU 直計算」の経路数値という
   review 側の構造分析 (BENCH_RESULTS 全文精読) と整合する。

## 後続アクション

- P2 (受入パスライン引き直し) — review 側着手時の基礎データとして本 doc を使用可
- arm c の長文完走データが必要になった場合のみ別ウィンドウで再実行 (~90 分覚悟)
- F3 exe 承認 (@1994008ee) の前提 ([QSTAR-]=0) は全アームで成立
