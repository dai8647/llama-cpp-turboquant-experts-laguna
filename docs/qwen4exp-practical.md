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

## Session A follow-up (2026-09-19)

Canonical measurement was attempted with `build-stage1\\bin\\llama-cli.exe`, `-fa on`, ctx 8192, `n_predict 256`, exactly three runs intended, `LLAMA_MOE_HOST_BANK=0`, and `LLAMA_MOE_SLOT_STATS=1`. The model loaded to the interactive prompt, but run 1 aborted on the first generation with the generic HIP/ROCm abort at `ggml/src/ggml-cuda/ggml-cuda.cu:108`.

No valid `Generation t/s`, slot stats, transfer-path counts, `compute_buffer` line, or median was produced. Consequently the following values are **not measured** in this attempt: gen t/s median, hit rate, `stage_nonhost`, `pageable`, effective bandwidth, and decode CPU percentage. The Session A success gate (bandwidth >= 10 GB/s and decode CPU <= 70%) cannot be evaluated.

Artifacts:

- `bench_sessionA_run_20260919.log`
- `bench_sessionA_cpu_20260919.csv` (model-load samples only; not decode CPU)
- `bench_qwen38_20tps_logs\\20260919-164005\\run-1.log`
- Detailed synchronization analysis: `docs/outsourcing/session-a-followup-sync-stall-2026-09-19.md`

The decode miss wait is on a dedicated copy stream event, not a device-wide or compute-stream synchronization. However, remap runs on the host before graph submission and currently holds `cache_mutex` while waiting, so the wait delays compute submission. Removing it is unsafe without a compute-stream wait-event API, in-flight-slot eviction protection, and event lifetime handoff. This overlaps Session B and was intentionally not implemented.

## 78 MiB/s transfer split (Session A follow-up)

The previously reported `78.3 MiB/s` is an end-to-end materialization rate, not isolated PCIe DMA bandwidth. Its denominator includes source preparation, H2D enqueue, copy-stream event wait, and event cleanup. The `stage=100%`, `stage_nonhost=0`, `pageable=0` result proves that the pinned-stage branch ran, but mmap-backed GGUF pages can still be file-backed while classified as host-backed.

New Session A telemetry separates the components in `MoE transfer split`: `stage_memcpy`, `stage_get`, `h2d_wait`, `h2d_bw`, and synchronous fallback timings. A fresh `build-stage1` canonical run is required before assigning the 78 MiB/s bottleneck to host memcpy/page faults, backend tensor reads, or HIP H2D. See `docs/outsourcing/session-a-78mib-split-2026-09-19.md`.
