# DeepSeek-V4-Flash-0731-reap-150b-Q3_K_M チューニング結果 (RX 7800 XT 16GB)

機種: AMD Radeon RX 7800 XT (gfx1101, 16GB VRAM) / Ryzen 5 5500 (6C/12T) / RAM 95.8GB / ROCm 7.1
計測: 日本語プロンプト、n_predict=200、llama-server /completion の timings.predicted_per_second
共通: `-m <Q3_K_M.gguf> --host 127.0.0.1 --port 8091 --no-webui -c 8192 -np 1`

## 結果一覧 (config -> pred tps)

| Run | 構成 | pred tps | 備考 |
|---|---|---|---|
| 1   | `--cpu-moe -fa on -ctk q8_0 -ctv q8_0 -t 6` | 5.519 | ベースライン |
| 2   | 上記 + `--spec-type draft-dflash --spec-draft-ngl 99` | 5.093 | DFlash未発動 (draft model未指定) |
| 3a  | 1 + `-ctk turbo4 -ctv turbo4` | 4.652 | TurboQuant単体は悪化 |
| 3b  | 1 + `--spec-type draft-dspark --spec-draft-model dspark-...MXFP4.gguf --spec-draft-n-max 3 --spec-draft-ngl 99` | 7.038 | DSpark有効化 1.28x |
| 3c  | 同上 n-max 4 | 8.502 | acc/pos (0.95,0.93,0.93,0.93) |
| 3d  | 同上 n-max 5 | 9.429 | acc/pos 全位置0.97 |
| 3e  | 同上 n-max 6 | 7.249 | pos5以降0.0 で悪化 |
| 3f  | 3d + `-t 8` | 8.907 | -t 6が最適 |
| 3g  | 3d + `--no-mmap` | 10.360 | **10 tps突破** |
| 3h  | 3g 再実行 | 10.673 | 再現性確認 |
| 3i  | 3g + `-ctk turbo4 -ctv turbo4` | 11.663 | 最良 |
| 3j  | 3i 再実行 | 10.755 | 平均 ~11 tps |

## 最良構成 (3i)

```
--cpu-moe -fa on -c 8192 -np 1 -t 6 -ctk turbo4 -ctv turbo4 --no-mmap
--spec-type draft-dspark
--spec-draft-model C:\Users\dai86\.lmstudio\models\ggml-org\DeepSeek-V4-Flash-0731-GGUF\dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf
--spec-draft-n-max 5 --spec-draft-ngl 99
```

- メインモデル: 44/44層 GPU オフロード (6.97 GB)、expert は CPU (60.7 GB, --cpu-moe)
- DSpark ドラフト: 4/4層 GPU (10.32 GB)
- 合計 VRAM ~17.2 GB (GTT オーバーサブスクで動作)
- DSpark acceptance ~0.88-0.95, mean len ~5.4-5.7

## ホットエキスパート (frequency配置) の状態: 解除済み (2026-08-25)

旧記述(DeepSeek-V4-Flash時代の `decode() failed: resource deadlock would occur`)は
 stale。その後の修正で解決済み:

- `5edf04758` moe : bump slot cache clock under cache_mutex in prefetch
- `e684d233c` moe : fix frequency placement correctness and remove dead slot-map code
- `ba24f2254` llama : pool MoE slot-remap userdata in the expert cache
  (remap userdata の寿命問題を構造的に解決、gfx1101 ビルド修正込み)

Pass 1 収集は Qwen3.6-35B-A3B で正常動作済み(`ft_freq.json`、約6.9kトークンサンプリング、
40層×256エキスパート)。Pass 2 適用も下記のとおり動作確認済み。

### 運用上の罠: ratio と slot 数の関係

`--moe-gpu-expert-ratio` の既定値 1.0 は whitelist = 全エキスパートを意味する。
whitelist が slot 数より大きいと:

1. プリロードが whitelist 順(頻度降順)にスロットを埋め回った結果、
   **最後に残るのは最も冷たいエキスパート**(最悪の初期配置)
2. 全選択が「whitelist 内」と扱われ、ミスごとに 0.81ms の同期コピーが発生

slot30 + ratio 1.0 の実測: ヒット率 51% で頭打ち、ミス=エビクト、6575トークンの
プリフィルが 600s タイムアウト(コピー総量 417 GB)。**ratio は必ず
「(1層あたり確保したいslot数 × 層数) ÷ 総エキスパート数」以下に設定すること**
(slot30/40層/256エキスパートなら 0.117 程度)。

# Huihui-Qwen3.6-35B-A3B (Q4_K 20.2GB) MoEスロットキャッシュ (RX 7800 XT 16GB)

計測: bench_ft.ps1 (コードレビュー風 6575トークンプロンプト + 256トークン生成)、
32K ctx, KV q8_0, --cpu-moe, -fa on, -t 6, GGML_CUDA_DISABLE_GRAPHS=1
(グラフONだと長文プリフィルで argsort キャプチャクラッシュのため。
docs/moe-slot-cache-async-design.md §8.3 参照)

| Run | 構成 | pp t/s | tg t/s | 備考 |
|---|---|---|---|---|
| B0 | `--moe-gpu-expert-slot-num 30` (whitelistなし) | 157.6 | 12.00 | **キャッシュはゲートで無効**(実質スロットレス)。docs/frequency-placement-findings.md 発見1 |
| F1 | B0 + `--moe-expert-placement frequency --moe-freq-report-in ft_freq.json` (ratio 1.0 既定) | TIMEOUT | - | hit 51%、ミス=エビクト、コピー417GB。上記「運用上の罠」+ findings 発見2/3 |
| F1b | F1 の ratio を 0.117 に修正(≒top-30/層) | TIMEOUT | - | hit 51%で変わらず。whitelist不一致時の致命的悪化の実例 |

(過去セッション計測、4670トークンプロンプト・tg200: スロット無効 12.38 tg /
135.4 pp、slot30 LRU 11.84 tg(+11%) / 127.8 pp(-6%)、slot30+prefetch100ms 11.06 tg。
詳細は docs/moe-slot-cache-async-design.md §8)

# Global-LRU ページング最初の実測 + TDR切り分け (2026-08-25 午後, GPU独占)

コーダーA/Bの実装(q*: 991bf3042/b54973372/f80401d2d、capture修正: deb32dd9e、
以降 8ca1025a6 auto / 3b49ca50a prefill二重バッファ)をレビューエージェントが
GPU独占状態で検証。`bench_glru_qstar.ps1`(main)を使用。

| Stage | 構成 | 結果 |
|---|---|---|
| a | slot96, graphs OFF, global-LRU無し | **pp=166.5 / tg=12.91** — TDR無し。歴代最高ベースライン |
| b | a + `--moe-gpu-expert-global-lru` | **TDR無し**(両ラウンド600sタイムアウトまでサーバー生存)。ただし応答なし: hit 79.4%でも miss 471k×0.74ms ≈ 純コピー349秒、cross_evict=60k の激しいスラッシング。96グローバルslotでは動作セット(40層×top-8=320同時必要)に全く足りない |
| b' | 同 + slot160(auto実測157相当) | **完走するが遅い**: r1 pp=15.9/tg=1.13、r2 pp=17.4/tg=1.31(ベースライン比~10倍)。hit率は89.2%に改善、TDR無し。miss 348k×0.71ms≈純コピー247秒がそのままwalltime — remap op内の同期H2Dが直列ボトルネック。予算増でミス率は下がるが同期コピーコストは残る |
| c | graphs ON + 4535tok ロングプレフィル (deb32dd9e 二重修正入り) | コーダーB計測: 全構成クラッシュゼロ・graphs reused 140回。slot無効 156.7/12.74、slot30 156.3/12.51、slot30+PF 156.5/12.70、**auto(実測157slot) 155.7/13.24** — 受入基準(pp≥135.4/tg≥11.84)全クリア |

**TDRの帰結**: 当日14:27/14:39の LiveKernelEvent 141(WATCHDOG)は、Bが発見した
rocPRIM問題(gfx1101がtarget_names不在→partitioning_threshold=64フォールバック→
キャプチャ中 memcpy_and_sync D2H同期)のグラフキャプチャ実験がドライバーごと巻き込み、
同時刻に走っていたAのglruサーバーを道連れにしたものと特定。GPU独占での再試行では
global-LRU経路は一度もTDRしていない(stage-b)。

**q* 初の実ラン = ロード即死(未修正)**: `-Mode qstar` で warmup 時
`GGML_ASSERT(ggml_can_mul_mat)` (ggml.c:3327)。src/llama-graph.cpp:2698-2700 の
host畳み込み `mul_mat(p_mat[n_embd,r], w_col[r,1])` は ne[0] 不一致で不正。
修正案: `ggml_mul_mat(ctx0, ggml_transpose(p_mat), w_col)` ([n_embd,1]、数学は同一)。
A適用後に再検証する。

**スラッシング対策の方向性**: stage-b のデータは FreeToken の q* 論点そのもの
(転送コストが勝つmissはCPU実行へ)。q*修正後、(1) -Mode qstar でCPUオフロード効果、
(2) auto相当(157slot)のglobal-LRUでミス率低減、を測るのが次。
