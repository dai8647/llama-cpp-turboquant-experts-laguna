# Qwen3.8-Flash-Next 実務メモ (2026-09-16)

## 環境
- ROCm は **入っている**: `C:\Users\dai86\rocm-sdk-core\_rocm_sdk_core` (wheel 復元)
- ドライバ側は RX 7800 XT で HIP 実行確認済み
- 公式 `C:\Program Files\AMD\ROCm\7.1` は無い。上のパスを使う

## MTP
- この GSQ-RCO GGUF に **MTP / nextn テンソルは無い**
- メタデータの "mtp" は vocab の smtp 等の誤検出
- 公式モデルカードの 4B MTP は別ファイル。この量子化では speculative MTP は使えない
- 代替: `--spec-type ngram-simple`（文脈 n-gram、ドフト不要）は動くが短文ではほぼ効果なし

## 実測 (decode)
| 構成 | t/s |
|---|---|
| 本家 + MoE CPU 固定 | ~4 |
| `--moe-hot-expert` ctx 2048 | 13.5 |
| 同 + ctx 8192 | 12.1 |
| 同 + ngram-spec + ctx 8192 | 13.4 |

## 実務起動
```bat
C:\Users\dai86\llama-cpp-turboquant-experts-laguna\run-qwen4exp.cmd -p "hello" -n 64
```

- expert: RAM + VRAM hot cache (158 slot / 層, auto)
- dense: 全層 VRAM
- n-gram shard2: mmap でディスクのまま
- 文脈は削らない。8k で 12 t/s 前後、2k で 13.5 t/s

## 20 t/s まで
hit 率が ~31% (158/512) で頭打ち。PCIe miss が支配的。
- slot 増 (margin) は OOM しがちで 11 t/s に落ちた
- frequency pin はこのルーティングではほぼ変化なし
- 次: shared-expert を bank から外す / 別量子化 / ハードウェア側
