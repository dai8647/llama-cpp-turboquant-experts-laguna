# ハンドオフ — MTP 実行時 ROCm abort (copy stream 初期化) (2026-09-21)

**まず計画書 §3 (共通ルール) を読んでください。**
`docs/outsourcing/PLAN-qwen38-20tps-delegation-2026-09-19.md`
HOST_BANK=1 絶対禁止 / 測定は build-stage1・build-mtp のみ / 判定は gen t/s のみ。

---

## 進捗 (ここまで来た)

MTP の trunk hidden state 受け渡しは修正済み (コミット `2dda4c769`):
- trunk graph が最終層で常に gather していたのを、`embeddings_nextn_masked` 時のみに限定
- target は full-row、MTP context は masked output-row で copy サイズ一致
- **結果: draft ロード成功、`l_last-47` OOB 解消、MTP draft 実行 + token 生成まで到達 (8 token)**

## 新ブロッカー: 8 token 後に ROCm abort

```
ROCm error: unspecified launch failure
function: ext_copy_stream_get
stmt: hipStreamCreateWithFlags(&ext_copy_stream, 0x01)
current device: -1
```

## 根本原因の分析 (コード確認済み)

`ggml/src/ggml-cuda/common.cuh:1514` の `ext_copy_stream_get()`:
```cpp
cudaStream_t ext_copy_stream_get() {
    if (ext_copy_stream == nullptr) {
        ggml_cuda_set_device(device);   // <- device コンテキスト設定を試みる
        CUDA_CHECK(cudaStreamCreateWithFlags(&ext_copy_stream, cudaStreamNonBlocking));
    }
    return ext_copy_stream;
}
```

**問題**: `current device: -1` = 呼び出し時に有効なデバイスコンテキストが無い。

copy stream は **遅延生成** (初回呼び出し時)。hot-expert の expert prefetch / H2D が **decode 中に、メインの compute デバイスが設定されていないスレッド/コンテキストから初めてこの stream を要求** したとき、デバイスが未設定 (`-1`) のまま `hipStreamCreateWithFlags` が失敗する。

**なぜ 8 token 後か**: MTP (draft-mtp) が起動すると、draft 実行が hot-expert の expert prefetch 経路を叩き、そこで初めて copy stream が必要になる。MTP なしの通常 decode ではこのパスが発火しない/別タイミングで発火するため、今まで顕在化しなかった。

## あなたの仕事

### 1. 原因の確定 (コード + ログ)
- `ext_copy_stream_get()` を呼ぶ全経路 (`ggml_backend_cuda_ext_copy_stream`, `ext_h2d_async`, `ext_event_record` = ggml-cuda.cu:2661/2680/2697) を洗い、どれが decode 中のどのスレッド/コンテキストから呼ばれるか特定
- `ggml_cuda_set_device(device)` が正しい device に設定できているか。`current device: -1` は `cudaGetDevice` が失敗 or 未初期化を示す
- 特に: MTP draft の推論が別バックエンドコンテキスト (draft 用) を作り、そこで ext_copy_stream が未初期化のまま要求されていないか

### 2. 修正方針 (推測で直さず、まず確定してから)
有力な方向性 (いずれも要検証):
- a) copy stream を **遅延生成ではなく backend 初期化時に生成** する (device が確定しているタイミング)
- b) `ext_copy_stream_get()` 内で device が -1/未設定なら明示的に primary device (0) にフォールバックしてから stream 生成
- c) MTP draft コンテキストでも ext_copy_stream が使えるよう、draft backend 初期化時に stream を用意

**どれが正しいかは「どのスレッド/コンテキストが device=-1 で呼んでいるか」の特定次第。先にそれを確定させること。**

### 3. 検証
- 修正を build-mtp のみに適用、再ビルド
- smoke: `--no-warmup -n 1` で abort しないこと
- canonical MTP: ctx 8192 / n 256 で abort せず完走、acceptance / gen t/s を計測
- 8 token を超えて安定生成すること

## 参照
- 前段の hidden state 修正: コミット `2dda4c769`
- copy stream 実装: `ggml/src/ggml-cuda/common.cuh:1510-1518`, `ggml/src/ggml-cuda/ggml-cuda.cu:2661-2700`
- Session A の H2D/prefetch 経路: `src/llama.cpp` (ext_copy_stream 使用箇所)

## 禁止
- build-stage1 の変更、HOST_BANK=1、git push --force
- H2D 78MiB/s の最適化 (別ブロッカー、MTP 完了後)
- 推測による大規模変更 (まず device=-1 の呼び出し元を確定)

## 結果 (2026-09-21)

### 呼び出し元と原因

canonical log の成功した materialize 直後に初回 stream が要求される経路は target model の pinned-stage materialize です:

```text
src/llama.cpp:999
  ggml_backend_cuda_ext_copy_stream(prefill_pf_backend)
```

`prefill_pf_backend` は target context の accelerator backend であり、draft context の backend ではありません。MTP 側は hot-expert cache を無効化していますが、draft verification が target MoE paging を追加で進めるため、この lazy stream path が MTP 実行中に初めて発火します。呼び出しは CPU-side remap/graph post-processing から来る可能性があるため、HIP current device は thread-local です。

### 修正

`ggml/src/ggml-cuda/common.cuh` / `ggml-cuda.cu` に owner-device handling を追加しました:

- per-backend extension stream の mutex 保護
- stream getter 毎回 `ggml_cuda_set_device(ctx->device)`
- H2D / event create / event record / event destroy の直前にも owner device を選択
- device 0 fallback は追加しない

これにより backend logical device と host worker の current HIP device を混同しません。

### Smoke

`build-mtp`, `--no-warmup`, ctx 2048, `-n 1`, `HOST_BANK=0`:

- `compute_buffer: device=ROCm0` / `GPU path active`
- `adding speculative implementation 'draft-mtp'`
- `Generation: 1`
- 旧 `l_last-47` OOB / null-buffer assert / `current device=-1` stream-create failure は再現しなかった
- 出力上の Generation 行は prompt=0.4 t/s / generation=1000000 t/s（1 token の timing sentinel であり、性能値ではない）

### Canonical

ctx 8192 / `-n 256` の canonical run は `adding speculative implementation 'draft-mtp'` と prompt processing (`17 tokens, 0.34 t/s`) まで進み、旧 `current device=-1` の `hipStreamCreateWithFlags` エラーは消えました。しかし run は後段の ROCm error で完走せず、acceptance と有効な gen t/s は未計測です。artifact:

```text
mtp_stream_canonical.log
mtp_stream_canonical.err.log
mtp_stream_canonical.out.log
```

詳細な HIP stmt 行は PowerShell の stderr ラッパーで末尾が欠落しており、次回は raw `cmd.exe`/binary stderr capture が必要です。今回のログでは `adding speculative`、GPU path active、prompt timing、大量の slot materialize は確認できています。

## 報告
- device=-1 で呼んでいるのは target backend の pinned-stage materialize/prefetch 経路
- 修正は owner-device selection + mutex、stream/event API のみ
- smoke は旧 abort/OOB なしで成功
- canonical は後段 ROCm error で未完走、acceptance/gen t/s は未計測
- 残課題: canonical 後段 ROCm error の raw stmt capture と完走確認
