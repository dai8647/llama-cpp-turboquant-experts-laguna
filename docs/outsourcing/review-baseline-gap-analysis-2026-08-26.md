# 解析: 対照 4.5 t/s vs 監査 ~13 t/s のギャップ + budget 実験の続きへの注記 (review side, 2026-08-26)

対象: A の F1/F2 検証 + 対照走行結果。 GPU 不使用、 BENCH_RESULTS.md とスクリプト突合のみ。

## 現状の数字表 (A 実測 + BENCH_RESULTS.md 由来)

| 構成 | eval tok/s |
|---|---|
| q* ON, budget 300 (default) | 0.73 |
| q* ON, budget 30000 | 1.45 |
| q* OFF 対照, graphs OFF (slot32+glru, 単発 curl) | 4.34 |
| q* OFF 対照, graphs ON | 4.50 |
| **監査ベースライン**: slot96, glru無し, graphs OFF, -t 6, **-fa on, KV q8_0, 32K ctx** | **12.91** (pp 166.5) |
| 監査 slot30 + LRU | 11.84 |

## 1. budget 30000 では推論中の CPU 送りは構造的にゼロのはず

1 layer-step のミス上限 = r_max = 8 → 転送コスト最大 8 × 637 µs = **5.1 ms ≪ 予算 30 ms**。 条件 1 も転送優先 (637 µs < 真値補正後の CPU ~2270 µs)。 よって budget30000 run の推論中 `[QSTAR-CPUEXEC]` 発火は 0 であるべきで、 それでも 1.45 < 4.34 が残るなら差は「分割ロジック」ではなく **q* 有効化自体の経路オーバーヘッド (remap/mask/PREP)** か単発測定のばらつき。

ops 注意: `.qstar-on` ログは `rm -f` が `mv` に先行したため消失している。 今後は保存を先に (`mv a b ; rm a` の順ではなく `cp a b ; : > a`)。 次回 q* ON run の冒頭で `grep -c "\[QSTAR-CPUEXEC\] entry"` を取る習慣を。

## 2. 「対照 4.5 がこの環境の実値」はまだ言えない — 監査との条件差が多すぎる

run-qstar-trace.ps1 現行引数 (-m, -ngl 999, **-c 8192**, **-t 8**, slot32, glru, qstar系) と監査環境 (BENCH_RESULTS.md 行 68/90: **32K ctx, KV q8_0, --cpu-moe, -fa on, -t 6**, GGML_CUDA_DISABLE_GRAPHS=1, **slot96**) の差分:

- **`-fa` 未指定** (監査は on) — RDNA3 + graphs 無効で効く
- **KV 量子化なし** (監査は q8_0) — メモリ帯域に効く
- ctx 8K vs 32K (これは今の方が軽い)、 -t 8 vs 6、 slot32 vs 96/30、 glru ON/OFF
- **測定方法**: 単発 curl 1 発 vs ベンチハーネス (複数ラウンド・r1 破棄) — 単発は初回ペナルティ込みで数値が低く出る

特に -fa と KV 量子化は 2 倍級の差を生み得る。 加えて「eval tok/s」の分母 (生成長) も単発と ハーネスで違う。

## 3. 指示 (優先順)

- **P1**: 今日の r2 バイナリを `bench_glru_qstar.ps1 -BinaryPath <r2>\llama-server.exe` で **q* 無し** 監査相当フラグのまま回す。 **~12 再現 → ギャップは設定差だけ** / **~5 前後 → (b) 採用 feature 群 (gate lift / argsort capture / elastic VRAM / prefill PF / graphs) のどれかに退行** → その場合は bisect へ (そちらが q* より上位の案件になる)。
- **P2**: 受入合格ライン (≥12.38 等) は監査環境前提の数字。 P1 の結果次第で **ラインの再設定が必要になり得る** — B テンプレ変更は P1 確定まで待つこと。
- **P3**: cpu_bps × 8 過小表示の指摘は承認 (パディングで 8 expert 分計算しながら r=1 分で割るため、 真値 ~0.85 GB/s 推定)。 ただし補正後でも転送 637 µs < CPU 2270 µs なので条件 1 は依然転送優先 = CPU 経路は予算枯渇時にしか選ばれない。 native GEMV の単体マイクロベンチ (threadpool 有/無・AVX2 確認込み) で 0.85 GB/s 自体の妥当性を見ると、 host exec の将来性判断が付く。
