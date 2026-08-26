# FreeToken 深掘りレポート × Laguna 実装 対照表 (2026-08-26, review side)

出典: ユーザ提供の deep-research 解説レポート (FreeToken, arXiv 2608.16157 の技術解説)。 本文はユーザ保管。 本 doc は「うちがどこまで追っていて、どこが違うか」の確定版マップ。

## 採用マップ

| FreeToken 技術 | Laguna 状況 | 根拠 |
|---|---|---|
| 共有 LRU エキスパートキャッシュ (§2.1.3) | ✅ プロジェクトの出発点 | Phase 1a/1b/1c, global-LRU slot cache |
| プリフィル全層ダブルバッファリング (§2.2) | ✅ main 採用済み | 3b49ca50a→2d442cdb0 (`LLAMA_MOE_PREFILL_PF=1`), (b) merge 76adf21e4 |
| エラスティック VRAM 管理 (§2.4) | ✅ main 採用済み | 8ca1025a6 (`--moe-gpu-expert-slot-num auto`) |
| q★ 帯域適応 CPU-GPU 分割 (§2.1.1–2.1.2) | ⏸ 監査で撤退 → r2 再検証中 | drop a414acc7f; `qstar_cpu=0` 全ラウンド; feat/qstar-r2-rebuild で再構築 |
| セマンティック CKPT・再帰状態 (§2.3) | ➖ 対象外 | 対象モデル (Huihui/Ornith, Qwen3.6 系 MTP) に再帰層がなく恩恵なし |

## 重要な交差検証: q★ 撤回は論文自身の主張と整合する

レポート §2.1.2 より: 「RTX 5090 デスクトップはキャッシュミスをほぼ全て PCIe 経由でルーティングし、 8GB ラップトップだけがほぼ全て CPU 実行で最大性能」— すなわち **q★ の効果はハードウェアバランス依存で、 デスクトップ級では ~0 も正当な測定結果**。

うちの監査 (RX 7800 XT 16GB + 96GB RAM、 `qstar_cpu=0` 全ラウンド) はこの予測どおり。 → 撤回判断 (8dc94e3e5) は「再現失敗」ではなく「**環境的不要の実証**」。 r2 再挑戦の受入バーに `qstar_cpu>0` を課す意味は、 「16GB VRAM / 96GB RAM の中間構成でも本当に割り振りが発生しないか」の反証テストである。 出なければ round-2→main マージ時の BENCH_RESULTS.md に「環境的不適合 (論文 §2.1.2 の HW 依存性と一致)」として記載して確定させる。

## ポジション差分

- **本家**: Linux x86_64 + NVIDIA 専用 (CUDA 13 / RTX 30/40/50)。 ROCm は Issue #82 コミュニティフォーク (gfx1201/RDNA4, bf16 3B 57 tok/s 動作報告) + Draft PR #132〜#137 の途上。
- **うち**: llama.cpp 直改修 C++/HIP、 gfx1101 (RDNA3) 実機 + GGUF/Q4_K 量子化経路。 公式未対応領域の先行補完。
- レポート §5.3 の移植課題のうち warp サイズ問題 (GCN/CDNA 64 vs NVIDIA 32) は RDNA3 = wave32 のため該当せず、 将来 CDNA (MI シリーズ) 移植時に初めて効く。 TVM-FFI / FlashInfer 系の課題は Python スタック固有で llama.cpp 直改修方式とは無関係。

## 参考: 本家実測値との比較の際の注意

レポート記載の数値 (RTX 5090 prefill 6700 tok/s 等) は MXFP4/NVFP4 + PCIe 5.0 前提。 うちの環境 (PCIe 4.0 x16, Q4_K GGUF, RDNA3) とは転送帯域も量子化形式も異なるため、 数値の直接比較は無意味。 比較軸は常に「同一マシン上の before/after」(旧監査値 13.78〜13.07 t/s 等と同列) を使う。
