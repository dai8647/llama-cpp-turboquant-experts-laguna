# 静的監査 残穴 3 点の追補 (turbo 量子化マトリクス / freq テスト生存性 / SET_ROWS 可整性契約)

- 作成者: reviewer AI
- 日付: 2026-08-27
- 前提文書: `docs/research/static-audit-full-fork-2026-08-26.md` (@071e753e0) の §6-6/§8「残穴」
- 方法: 全部 grep + ソース直読 (静的)。実機実行なし。

---

## ① TurboQuant backend×op カバレッジマトリクス(完全化)

### 型一覧の確定 (`ggml/include/ggml.h`)

| 型 | ID | 由来 | 用途 |
|---|---|---|---|
| TQ1_0 / TQ2_0 | 34 / 35 | 上流 llama.cpp 本家 | 重み |
| TQ3_1S / TQ4_1S | **45 / 46** | fork 追加 (WHT 回転 Lloyd-Max, block=32, 単一スケール) | 重み |
| TURBO2_0 / TURBO3_0 / TURBO4_0 | **47 / 48 / 49** | fork 追加 (WHT + PolarQuant+QJL, arXiv 2504.19874) | KV cache 専用設計 |

カスタム op `GGML_OP_TURBO_WHT` も存在 (全バックエンドに実装、下表)。

### マトリクス(監査穴を埋めた実測)

| op | CPU | CUDA | Metal | Vulkan |
|---|---|---|---|---|
| MUL_MAT (vec_dot/MMQ) | **✓** type_traits 完備 + `ggml_vec_dot_turbo_f32_impl` (ggml-cpu.c:3597-3617) | **✗** MMQ テーブルに turbo なし (mmq.cu TURBO = 0 件)。MUL_MAT_ID 高速経路も同様 | 部分的: mul_mat 側で ktype==T2/T3/T4 を強制 nonvec 化する env `TURBO_FORCE_NONVEC` の実験経路あり (ggml-metal-ops.cpp:2986) | 未確認(裏取り未了 — flash 経路は下記の通り存在) |
| FLASH_ATTN_EXT (KV が turbo) | −(CPU FA 対象外) | **✓** fattn-vec インスタンス K×V 全組合せ (f16/q8_0/t2/t3/t4 の積集合、template-instances/)。dequant ヘルパー fattn-common.cuh | **✓ +専用高速化**: TurboFlash 二段融合アテンション (V=turbo3 且つ decode 単 token 且つ K∈{q8_0,turbo3}) kernel_turbo_flash_p1/p2 (dk/dv 64/96/128) | **✓** flash_attn_cm1 / dequant_funcs_cm2.glsl + turbo3 K 特例 (vulkan.cpp:4518) |
| SET_ROWS (F32→turbo 書込み) | **✗ 実装なし** (ops.cpp forward_set_rows に turbo 0 件 → KV turbo は GPU バックエンド必須) | **✓** set-rows.cu に符号化カーネル本体 (+TURBO_WHT_SIGNS テーブル) | ✓(m 系ファイル群に実装) | **✓** pipeline_set_rows_turbo{2,3,4}_0 (vulkan.cpp:5527-5529) |
| CONVERT/COPY (dequantize) | ✓ | ✓ convert.cu dequantize_block_cont_cuda 3 型 (806-811) | ✓ kernel_turbo4_dequant_f16 等 | ✓ copy_to_quant.comp / dequant_funcs_cm2.glsl |
| GET_ROWS | ✓ ops.cpp:5813-5815 (clamp とは別の quant 受容リスト内…と見えた位置だが当該 switch は clamp 内 — 要注意) | **✗ 未収載** (getrows.cu TURBO = 0 件; 同関数は TQ3_1S/TQ4_1S は受容・turbo は default→unsupported) | 未確認 | 未確認 |
| TURBO_WHT (カスタム op) | ✓ ggml-cpu.c:2127/2314/3044 | ✓ ggml-cuda.cu:2289/5464 | ✓ ggml_metal_op_turbo_wht (ops.cpp:1921, pipeline 済) | (shaders 側で処理 or 不要 — 未精査) |

**結論 (運用上の意味)**:
1. **turbo KV は事実上「SET_ROWS + FLASH_ATTN_EXT」の組合せ専用**。K/V の高速 mul_mat (MMQ) はどのバックエンドにもなく、fork 内の実際のグラフ構築も FA 経路しか組んでいない (fattn インスタンス群 + TurboFlash + cm1)。q8_0 KV と同じ非 FA 構成を turbo で狙うと即サポ外エラーになる (決定的失敗、サイレント破壊ではない)。
2. embedding 等を turbo にする用法は GET_ROWS 未対応で不可能 — これは設計通り (KV 専用型) なので benign。
3. **将来レビュー観点**: Vulkan の SET_ROWS ゲートは全 turbo 型で `ne[0] % 128` 要求 (vulkan.cpp:18187 のコメント「128-element block」)、CUDA は T2/T3=%64・T4=%128 (ggml-cuda.cu:5292-5302)。**同一 head_dim でも CUDA では通るのに Vulkan では通らない帯域がある** (%64〜%127 の head_dim、ただし現行主流モデルは 64/128 どちらかの倍数なので実害なし)。バックエンド横断で contract を揃えるか、明示的に документ化する価値はある。
4. 軽微な疑義 1 件: CPU `ggml_compute_forward_clamp` が K-quant ファミリ + turbo を受容する拡張が入っている (ops.cpp:5813-5815 はその case リスト)。用途が不明 — KV clamp 実験用と思われるが A に出典確認推奨。

## ② tests/test_moe_frequency_placement.py の生存性

- 実在し、`LLAMA_CLI` + `MODEL` (MoE GGUF) の両 env がないとき **exit 77 で自己 skip** する親切設計 (メッセージには実行例まで印字される)。整合性チェックの入口としては健全。
- 今回は実走していない (GPU 消費 + A/B のベンチ領域)。B の P1 ランナー資産 (`bench_glru_qstar.ps1 -BinaryPath` 経由の exe パス管理) と同じバイナリを使えば空き枠で実走可能。P2 の受入ライン引き直しツールチェーンに入れる候補。

## ③ 「getrows block_dim 64 契約」の正体確定

監査エージェント報告の「getrows block_dim 64 契約」は**場所の記載が不正確だった**:
- getrows.cu 自体には turbo 型のインスタンス化がゼロ (hole ① の「GET_ROWS ✗」と同じ事実)。
- 同ファイルの `template<int block_dim> k_get_rows_kq` (getrows.cu:169) は K-quant dequant 版 get_rows のブロック次元テンプレートで、turbo は対象外。
- 実際に head_dim 制約を持つのは **SET_ROWS の supports_op ゲート**: CUDA `ne[0] % 64` (T2/T3) / `% 128` (T4) at ggml-cuda.cu:5292-5296、Vulkan `% 128` 共通 at vulkan.cpp:18187。WHT は 64/128 グループ回転なのでこの制約は数学的に必然。

## P1 確定値との接続

B の P1 (main @1da028abc) で stage-a 再現 tg≈12.3-13.5・graphs ON 13.88・A-control glru 構成 1.44 t/s が確定したことにより、「sky ベースライン ~13 t/s 帯を q*/glru でどこまで持ち上げるか」が q* の価値基準に確定済み。本追補のマトリクスはその後のフェーズ (FA 経路前提の最適化) での後方互換性確認材料として使える。

## 変更履歴

- 2026-08-27: 初版 (reviewer AI)
