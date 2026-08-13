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

## ホットエキスパート (frequency配置) の状態: BLOCKED

- `--moe-gpu-expert-slot-num 30 --moe-freq-report-out freq_dsv4.json` (Pass1 収集) は
  実行時に `decode() failed: resource deadlock would occur` (HIP runtime error) で失敗。
  `--cpu-moe` 併用時はサーバーが接続リセット (クラッシュ)。
- 原因: フォークの eval-time remap (`build_moe_gpu_slot_ids` -> `ggml_cpy` +
  `ggml_map_custom1`) が HIP バックエンドでデッドロック。map_custom1 は ggml-cpu のみ実装で、
  GPU/CPU テンソル境界処理が HIP で破綻。
- Pass2 (frequency whitelist適用時) も同じ map_custom1 経路を通るため、同様に壊れている可能性が高い。
- 修正にはフォークの再ビルドが必須 (10-30分規模)。
