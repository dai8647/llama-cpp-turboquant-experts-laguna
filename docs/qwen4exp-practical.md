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

### 2026-09-21 split smoke result (invalid throughput run)

A stage1-only run was started with `HOST_BANK=0`, `SLOT_STATS=1`, ctx 8192, `-n 256`, `-fa on`, and `-t 12`. It aborted before `Generation` output with a ROCm launch failure, so no gen t/s or cold/warm comparison is valid. The final telemetry emitted before abort was:

```text
copy_bw=39.4 MiB/s, stage=4864, stage_nonhost=0, pageable=0
stage_memcpy=6412.5 MiB / 7284.99 ms
stage_get=0.0 MiB / 0.00 ms
h2d_wait=6412.5 MiB / 26161.91 ms
h2d_bw=245.1 MiB/s
hit=9197 miss=4864 evict=17
```

The H2D event wait was about 3.6x the host memcpy time in this partial run, and `stage_get` was unused. This points to copy-stream/HIP H2D wait as the dominant measured component for this run, not non-host backend reads. Because the process aborted before generation, these values are diagnostic only and do not establish a throughput result or a cold-vs-warm page-fault comparison. Artifacts: `split_cold_20260921.log` and `split_cold_20260921.err.log`.

## MTP draft-mtp GPU test (2026-09-21)

The `build-mtp` GPU test was executed with `LLAMA_MOE_HOST_BANK=0`, `LLAMA_MOE_SLOT_STATS=1`, ROCm on an AMD Radeon RX 7800 XT, the Qwen3.8-Flash-Next GSQ-RCO Q2_0 target, and the generated 2.84 GiB Q8_0 draft GGUF. No other `llama-cli` or `llama-server` process was running before the test.

Result: **failed before MTP context creation; no decode t/s or acceptance rate**. The target model loaded far enough to allocate the MoE slot banks, but loading the draft failed because the draft GGUF is missing the required `qwen4exp.context_length` metadata key. The log first reports the draft metadata and `nextn_predict_layers = 1`, then fails with:

```text
error loading model: error loading model hyperparameters: key not found in model: qwen4exp.context_length
common_speculative_init_result: failed to load draft model
```

The failure is recorded in `C:\Users\dai86\llama-cpp-turboquant-experts-laguna\mtp_test.log`. There was no ROCm abort or `ggml_cuda_error`; no `creating MTP context`, draft KV allocation, acceptance statistics, or generation benchmark was reached. `LLAMA_MOE_HOST_BANK=0` was confirmed in the slot-bank log (`host_bank=0`).

The draft must be regenerated or metadata-fixed before a valid MTP performance measurement can be made. No source or MTP code was changed.

### MTP draft shape repair attempt (2026-09-21)

The first repaired draft had a writer bug: raw Q8_0 bytes were emitted without a logical tensor shape. The prep script was updated to preserve quantized byte-row shapes for 1D/2D/3D tensors, and the draft was regenerated. The resulting GGUF was verified with these loader-visible shapes:

```text
block_count=49
nextn_predict_layers=1
token_embd.weight       [2560, 248320] q3_K
output_hc_down.weight   [10240, 320] q8_0
output_hc_up.weight     [320, 10240] q8_0
blk.48.ffn_down_exps    [640, 2560, 512] q8_0
blk.48.ffn_up_exps      [2560, 640, 512] q8_0
```

`build-mtp` was rebuilt successfully. The retest no longer reports a tensor shape error, but still fails during draft `load_tensors` with:

```text
llama_model_load: error loading model: invalid vector subscript
```

The trace shows all 33 draft tensors being indexed with valid shapes immediately before the failure. There is no `n_layer_nextn = 1`, MTP context creation, draft KV allocation, acceptance statistic, or generation rate. No ROCm abort occurred. The remaining issue is an internal vector access in the qwen4exp draft loader path, not the original raw-shape bug. A stack trace or narrow logging around the `load_arch_tensors` vector accesses is required before changing the loader. MTP gen t/s and acceptance remain unmeasured.

### MTP loader root cause (2026-09-21, second pass)

The `invalid vector subscript` was `src/llama-model.cpp`'s `devices.at(layer_gpu)` where `std::upper_bound` on the device splits returned `devices.size()` at a split endpoint. The fix clamps the index to `devices.size()-1` and checks `gpu_buft_list` with `find` instead of `at`. After this fix the draft model loads correctly (`n_layer_all=49`, `n_layer_nextn=1`, `n_layer=48`, `mtp_only=1`, `loading MTP layer il=48 layers=49`) and `adding speculative implementation 'draft-mtp'` is reached.

Generation then fails with a new, non-loader error:

```text
ggml_backend_tensor_get_async OOB: tensor=l_last-47 offset=0 size=368640 nbytes=0
ggml-backend.cpp:272: tensor read out of bounds
```

The last trunk hidden-state tensor (`l_last-47`) is empty when the MTP graph reads it. This is a graph-construction/execution issue after loading. Acceptance stats and gen t/s remain unmeasured.

### MTP graph handover fix (2026-09-21)

Root cause: the qwen4exp trunk graph always gathered the last-layer residual through `inp_out_ids`, even when `embeddings_nextn_masked=false`. The target MTP path requests full-row nextn embeddings (`n_rows=ubatch.n_tokens`), but the gathered `l_last-47` tensor had only output rows; during prefill it could have `nbytes=0`. This caused:

```text
ggml_backend_tensor_get_async OOB: tensor=l_last-47 ... nbytes=0
```

The fix matches qwen3next: gather the last trunk layer only when `cparams.embeddings_nextn_masked` is true, and gather the final LM output separately when it is false. The smoke run with `build-mtp`, ctx 2048, `-n 1`, `HOST_BANK=0`, and `--no-warmup` then passed the previous OOB/assert path and reached draft-mtp initialization and token generation without OOB, null-buffer, or ROCm errors.

A canonical ctx 8192 / `-n 256` run also reached `adding speculative implementation 'draft-mtp'` and generated 8 tokens, but then stopped on a separate transient GPU resource failure:

```text
ROCm error: unspecified launch failure
function: ext_copy_stream_get
stmt: hipStreamCreateWithFlags(&ext_copy_stream, 0x01)
```

The canonical run therefore has no valid gen t/s median or acceptance result. Its partial output was 8 tokens at a reported predicted rate around `0.28 t/s`, but this is invalid because the run aborted and must not be used as the MTP benchmark result. Artifact: `mtp_canonical.log`.

### MTP copy-stream owner-device fix (2026-09-21)

The target MoE pinned-stage materialize path lazily created a backend-owned copy stream from CPU-side remap/post-graph work. On HIP, the current device is host-thread local, so the stream/event APIs were hardened to select the backend context's owner device on every stream/H2D/event operation and to serialize lazy stream creation with a mutex. No device-0 fallback was added.

A `build-mtp` no-warmup smoke (`ctx=2048`, `-n 1`, `HOST_BANK=0`) reached `adding speculative implementation 'draft-mtp'` and one generated token without the old OOB/null-buffer/copy-stream `current device=-1` failure. The one-token timing is not a performance measurement.

The subsequent canonical ctx8192/256-token run still did not complete: it reached draft-mtp initialization and prompt processing, then stopped on a later ROCm error with no valid acceptance or generation rate. See `docs/outsourcing/HANDOFF-mtp-stream-abort-2026-09-21.md` for the raw artifact paths and remaining capture work.

## MTP draft-mtp retest (2026-09-21)

The repaired draft was tested with the `build-mtp` binary only:

```text
build-mtp\\bin\\llama-cli.exe
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75
--moe-hot-expert -ngl 99 -fa on -c 8192 -t 12
LLAMA_MOE_HOST_BANK=0
LLAMA_MOE_SLOT_STATS=1
```

Draft path:

```text
C:\Users\dai86\Downloads\mtp-draft\mtp-Qwen3.8-Flash-Next-draft-q8_0.gguf
```

Result: **failed during draft model tensor loading**. The target model reached its metadata/tensor-loading phase, and the log emitted `loading draft model`, but draft loading terminated with:

```text
llama_model_load: error loading model: invalid vector subscript
common_speculative_init_result: failed to load draft model
srv llama_server: exiting due to model loading error
```

No successful MTP markers were reached:

- no `n_layer_nextn = 1`
- no `creating MTP context`
- no draft KV allocation
- no acceptance statistics
- no `Generation: ... t/s`
- no ROCm abort

Artifacts:

- `mtp_test2.log`
- `mtp_test2.stderr.log`
- `mtp_test2.stdout.log`

The process exited with code 1. This is a draft-loader/vector-shape failure, not a GPU transfer or ROCm runtime failure. Therefore gen t/s and acceptance are **not measured**. No code, `build-stage1`, or `build-mtp` files were modified during this retest.