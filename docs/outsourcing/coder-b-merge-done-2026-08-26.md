# コーダーB への通知確認 — (b) main merge 受領 (2026-08-26)

## 結果 (受領・検証済み)
- **main HEAD = 76adf21e4** マージを受領。内容レベルで B 側からも main を直接 grep し通過を確認した
- **graphs 関連 3 commit 通過** (cherry-pick 由来でハッシュは新規、元 → main 対応):
  - `44a98b977` → **`e4372a6ab`** ggml : per-context CUDA graphs kill switch + map_custom capture guard
    (`ext_graphs_disabled` / `ggml_backend_cuda_ext_set_graphs_enabled` を ggml-cuda.h / common.cuh / ggml-cuda.cu で確認)
  - `292864d86` → **`0ef48796b`** llama : disable CUDA graphs for q*/global-LRU paging via ext API
    (`graphs_disable_pending` → llama_context 適用を llama-context.cpp で確認)
  - `ac49a0fa2` → **`192b47eda`** docs : cache_mutex/graphs-paging design sections + capture_probe12.cu
    (`ggml/tests/capture_probe12.cu` も main に存在)
- 同様に capture-safe sorts (`87efeff5a`) / elastic VRAM auto (`7a212bd76`) / prefill PF (`2d442cdb0`) も内容確認済み

## A 領域への転送事項 (受領、B は対応不要)
- 大型プレフィル + glru 無言クラッシュ (6575 tok, slot96, materialize ~13 秒・layer 18) は
  `docs/outsourcing/coder-a-bug-large-prefill-2026-08-25.md` 経由で A 担当に転送済みとのこと。
  B として補足: env 方式 (`GGML_CUDA_DISABLE_GRAPHS=1`) でも再現するため graphs 機構とは無関係、
  warmup decode+paging は成功するので prefill-paging 特化。再現手順が必要なら B 側で提供可能。

## B 側ステータス
- feat/prefill-double-buffer は役割終了 (内容は全て main へ反映済み)。feat/b-only 温存方針に従う
- q* drop (a414acc7f) も受領。q* 再起時は design doc §8 の cache_mutex 保持時間契約と
  §9 の per-context graphs スイッチが前提になる旨、ここに明記しておく

`[STATUS: merged 76adf21e4 acknowledged by B, standing by for next round]`
