# DeepSeek V4 Flash - 速度検証記録 (2026-08-05)

## 目的
RX 7800 XT (16GB VRAM / 96GB RAM / ROCm 7.1) で DeepSeek V4 Flash のローカル実行速度を解析し、
モデル選び・ビルド設定のための知見を残す。

## 環境
- GPU: AMD Radeon RX 7800 XT (gfx1101), 16368 MiB VRAM, 16223 MiB free
- RAM: 95.8 GB total / 86.4 GB free (実行時)
- CPU: AMD Ryzen 5 5500 (6コア/12スレッド)
- ビルド: llama-cpp-turboquant-experts-laguna, merge-upstream ブランチ, build 10260 (8bb26f514)
- バックエンド: ROCm (HIP), ggml-hip.dll + llama-cli.exe カスタムビルド

## 重要な計測事実

### 1. K160 (REAP) の「73 t/s」は誤計測・壊れ出力
モデル: DeepSeek-V4-Flash-0731-REAP-K160-Q2_K.gguf (60.9 GB)

実行: `llama-cli -ngl 99 -fa on -n 300 -p "..."`
表示: `[ Prompt: 66.4 t/s | Generation: 73.6 t/s ]`

しかし verbose ログで実態が判明:
```
load_tensors: offloaded 44/44 layers to GPU
load_tensors:        ROCm0 model buffer size = 0.00 MiB      # 重みがVRAMに載っていない
load_tensors:    ROCm_Host model buffer size = 0.00 MiB       # RAMにも載っていない
common_params_fit_impl: projected to use 69397 MiB of device memory vs. 16189 MiB free
common_params_fit_impl: cannot meet free memory target of 1024 MiB
```
- 60.9GB の重みは VRAM (16GB) にも RAM にも確保できず、SSD から都度読み出し
- 実際の速度 (verbose 実行) は 7 tokens / 1.26 s = **実質 ~0.2 t/s 相当**
- 出力内容は `##---` `<?import#importpackage##B####一个` など**意味不明の壊れたテキスト**

### 2. なぜ「速く」見えたか
- モデル重みが RAM に入らないため、毎トークン SSD 読み出しが発生
- 生成は破綻しながらも次トークンへ進むため、t/s 数値は出る
- その際 SSD 使用率 100% が続き、その後の実行でシステムが落ちた
- ページファイルが自動管理のため、RAM 逼迫時に OS が SSD へ大規模スワップ

### 3. Ombro v1 (非REAP) の比較
モデル: DeepSeek-V4-Flash-0731-Ombro-v1.gguf (75.8 GB, 256エキスパート非REAP, IQ量子化)

実行: `-ngl 99` → OOM (80GB要求)
実行: `-ngl 99 --cpu-moe` → 動作するが 3.9 t/s (CPU-MoE)

## 結論
1. **60GB超のモデルはこのマシン(16GB VRAM / 96GB RAM)では VRAM に全量オフロードできない**
2. 16GB VRAM に収まるのは実質 ~12-14GB 程度のモデルのみ。Q2_K でも 160 エキスパート REAP は 60.9GB で不可能
3. 「t/s が出てるから速い」は信用できない。重みが VRAM/RAM に載っているか、
   出力が意味のあるテキストかを必ず確認する
4. SSD 100% 連続 → システムダウンの原因はモデル重みの mmap/SWAP 逼迫

## 今後の方針
- モデル選び: **総ファイルサイズ ~14GB 以下** を目安にする (16GB VRAM 収まり)
  - 参考: K160 Q2_K (60.9GB) は NG, Ombro (75.8GB) は NG
- 実行時の確認事項:
  - `load_tensors: ROCm0 model buffer size` が実数 (MiB) になっていること
  - `common_params_fit_impl: cannot meet` が出ていないこと
  - 生成テキストが意味をなしていること
- 巨大モデル (60GB+) を RAM 96GB で動かす際は SWAP 逼迫に注意 (ページファイル確認)
