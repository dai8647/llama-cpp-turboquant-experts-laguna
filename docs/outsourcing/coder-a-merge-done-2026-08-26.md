# コーダーA へ — (b) main merge 完了通知 (2026-08-26)

## 結果
- **main HEAD = 76adf21e4** に (b) q* revert + B-only マージを push 済
- feat/b-only (a414acc7f) も push 済、温存
- リモート検証済 (origin/main = 76adf21e4)

## 採用コミット (cherry-pick 11 + q* drop 1)
- `11818978a` moe : lift collection-mode gate so global-LRU paging is reachable
- `deb32dd9e` ggml : make cub/hipcub sort workspaces stream-capture safe on HIP graphs
- `8ca1025a6` moe : elastic VRAM sizing for expert slots (q* 行 drop 済)
- `2d442cdb0` (= 3b49ca50a) moe : prefill double buffering
- `d7034040c` docs : MoE prefill double-buffer design notes
- `8fa0959bc` docs : bring forward main measurements and merge-resolution update
- `6b23cabbe` docs : q* acceptance reproducibility audit
- `8dc94e3e5` docs : q* acceptance numbers invalidated (A 撤回)
- `44a98b977` ggml : per-context CUDA graphs kill switch + map_custom capture guard
- `292864d86` llama : disable CUDA graphs for q*/global-LRU paging via ext API
- `ac49a0fa2` docs : cache_mutex/graphs-paging design sections + capture_probe12.cu
- `e7ed2fa9e` docs : forward B's large-prefill paging crash to A + organize handoff
- **`a414acc7f` (新) moe : drop q* bandwidth-adaptive split for (b) main merge** — 395 行削除

## 数字で見る監査結論 (B 環境)
- 短文 r1/r2/r3 = 5.95 / 5.36 / 6.49 t/s (A 報告の 13.78 / 16.81 / 16.76 を再現不能)
- 長文 (6575 tok): REQUEST-FAILED (PF=0/1 両方、 qstar_cpu=0 ずっと)
- `qstar_cpu=0` 全ラウンド = q* host exec 分岐が一度も発火していない
- A の `1e3af73a2` (ne[2]/ne[3]) + `8a0a55bbd` (mixed-precision) 適用後も calibrate で死亡 (Huihui 限定ではなく Ornith type=11 でも同症状)
- 結論: q* body の compute path に別バグ、A 環境 + ASan 切り分けが必要

## A への次のお願い

### 1. q* cpu_exec crash 切り分け (継続)
debug ビルド + ASan で `qstar_cpu_exec` 3 回ループ死亡の原因を特定。
main には q* を入れないまま別 branch (`feat/qstar-debug` 等) で作業可。
完了条件:
- `qstar_cpu>0` 観測 + 短文 12+ / 長文 12+ / Ornith 12+ t/s の 3 項目 再現
- A 環境ビルドハッシュ + warmup プロトコル + graphs 状態を docs に明記

### 2. 大型プレフィル + glru クラッシュ (B 発見、A 領域)
6575 tok + slot96 + glru 有効で materialize 中 ~13 秒・layer 18 で無言死亡。
- 環境: `GGML_CUDA_DISABLE_GRAPHS=1` でも再現 (graphs と無関係)
- 暖機 decode + paging は成功 → prefill-paging 特化バグ
- 疑い: prefill ubatch での `ensure_resident` / `evict` 競合
- 元報告: `docs/outsourcing/coder-a-bug-large-prefill-2026-08-25.md` (B 作成、 A へ転送済)

## 連絡窓口
- 依頼 docs 末尾に `[STATUS: merged 76adf21e4, waiting for next round]` スタンプ追加済
- 質問・進捗は本ファイルへ追記 (私 = レビュー側、 ユーザ経由)
