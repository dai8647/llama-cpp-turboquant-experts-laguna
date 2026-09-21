# Qwen3.8-Flash-Next (qwen4exp) 容量拡大評価メモ

**作成**: Session D (LlamaDock 統合) — 2026-09-19
**対象**: Qwen3.8-Flash-Next-GSQ-RCO-Q2_0 (61.9 GB, 512 experts, 48層 MoE)
**現状**: 158 slots/layer (auto), hit rate ~31% (158/512), 13.4 t/s @ ctx 8192

---

## 1. 現状の物理制約

| 項目 | 値 | 備考 |
|------|-----|------|
| 総 expert 数 | 512 / layer | 8 experts × 64 layers? (実質 48 MoE layers) |
| VRAM 空き (RX 7800 XT 16 GB) | ~14 GB 実効 | KV cache, compute buffers, OS で削減 |
| slot 単位サイズ | ~92 MiB | Q2_0 quant, 全層 expert 1 個分 |
| 最大 slot 数 | 158 / layer | auto 計算でこの値に落ち着く |
| 理論 hit rate 天井 | **31%** | 158 / 512 |

**結論**: 物理容量 (VRAM) がボトルネック。LRU アルゴリズムの問題ではない。

---

## 2. 容量拡大のレバー別見積もり

### 2-A. より小さい expert 量子化 (IQ2 系)

| 量子化 | 推定 expert サイズ | 推定 slots (同 VRAM) | 推定 hit rate | 備考 |
|--------|-------------------|---------------------|---------------|------|
| Q2_0 (現状) | ~92 MiB | 158 | 31% | baseline |
| **IQ2_XS** | ~70 MiB | ~208 | **41%** | 最大の現実的選択肢 |
| **IQ1_M** | ~55 MiB | ~265 | **52%** | 品質リスク大、要検証 |
| Q3_K_M | ~115 MiB | ~127 | 25% | 逆効果 |

- **IQ2_XS** が現実的: hit rate 41% まで改善可能、品質劣化は Q2_0 並みとの報告あり
- **IQ1_M** は 52% まで届くが、perplexity 劣化が大きく実用閾値割れリスク
- 別量子化モデル生成コスト: GGUF 再作成 + 品質評価 = 敼時間〜数日

### 2-B. KV cache 量子化で VRAM 確保

| 設定 | KV VRAM (ctx 8192) | 節約分 | 追加 slots | 推定 hit rate |
|------|-------------------|--------|-----------|--------------|
| q8_0 / q8_0 (現状) | ~2.8 GB | - | - | 31% |
| **q4_0 / q4_0** | ~1.4 GB | **~1.4 GB** | +15 | **34%** |
| **q4_0 / q8_0** | ~2.1 GB | ~0.7 GB | +7 | **32%** |

- 非対称 KV (K=q8_0, V=q4_0) がバランス良い
- 品質影響: q8_0→q4_0 で perplexity +0.02〜0.05 程度 (許容範囲内の報告多)
- **即効性あり**: 量子化変更のみで再ビルド不要

### 2-C. mmproj (視覚プロジェクタ) 切り離し

| 項目 | 値 |
|------|-----|
| mmproj サイズ | ~0.85 GB (mmproj-Qwen3.8-Flash-Next-*.gguf) |
| 現状 | `-ngl 99` で VRAM に常駐 |
| 切り離し時の節約 | **~0.85 GB VRAM** → +9 slots → hit rate **33%** |

- テキストのみ用途なら `--mmproj` 指定を外すだけで即効
- 視覚タスク時のみ別プロセスで読み込み可能
- **最もコストが低い** レバー

### 2-D. 層ごとの slot 非均等化 (router-aware)

- 現状: 全層一律 158 slots
- 実測: 浅い層 (0-15) は expert 分布が広い、深い層 (30-47) は集中
- 深い層に多め、浅い層に少なめの分配で実効 hit rate 改善可能
- 実装: `llama_moe_gpu_expert_slot_auto_pin` 付近で層ごと budget 計算を分岐
- **Session A/B の成果待ち** (コード変更必要)

---

## 3. 統合シナリオ別 hit rate 見積もり

| シナリオ | 対策 | 推定 hit rate | 期待 t/s (現状 13.4) | 実装コスト |
|----------|------|--------------|---------------------|------------|
| **S1: 即効 (設定のみ)** | KV q4_0/q4_0 + mmproj 切り離し | **34%** | ~15 t/s | 0 (設定変更) |
| **S2: 短期 (再量子化)** | S1 + expert IQ2_XS | **42%** | ~17 t/s | 敼時間 (GGUF 再作成) |
| **S3: 中期 (非均等 + S2)** | S2 + 層非均等化 | **50%+** | ~19 t/s | Session A/B 後 |
| **S4: 積極 (IQ1_M + S3)** | S3 + expert IQ1_M | **55%+** | **20 t/s+** | 品質リスク要検証 |

---

## 4. 20 t/s 達成への推奨パス

### Phase 1 (即時・設定のみ) — 目標 15 t/s
1. KV cache: `q4_0 / q4_0` (非対称も可: K=q8_0, V=q4_0)
2. mmproj: テキスト専用時は外す (`--mmproj ""` または指定なし)
3. これだけで +1.5〜2 t/s 見込み

### Phase 2 (短期・GGUF 再作成) — 目標 17 t/s
1. Expert 量子化を **IQ2_XS** に再作成
   - llama.cpp `convert-hf-to-gguf.py` + `llama-quantize` で IQ2_XS 生成
   - 品質確認: perplexity, MT-bench, 人間評価
2. Phase 1 の設定と併用

### Phase 3 (中期・コード変更) — 目標 20 t/s
1. **Session A**: pinned stage 確実化 + async H2D 帯域確保 (≥10 GB/s)
2. **Session B**: decode 早期 fire + layer-wise prefetch で miss 隠蔽
3. **Session B 追加**: 層ごと slot 非均等化実装
4. Phase 2 のモデル + Phase 3 のコードで **20 t/s 達成見込み**

---

## 5. リスクと判断基準

| リスク | 影響 | 対策 |
|--------|------|------|
| IQ2_XS 品質劣化 | 推論品質低下 | 事前に perplexity/MT-bench で定量評価 |
| IQ1_M 品質崩壊 | 実用不可 | **推奨しない**（保険としてのみ残す） |
| KV q4_0 精度低下 | 長文脈で幻覚増加 | 非対称 (K=q8_0) で緩和 |
| mmproj 切り離し | 視覚機能喪失 | 別プロセス/別サーバーで運用分離 |

**判断ゲート**:
- Phase 1 実施後、ベンチで **15 t/s 以上** なら Phase 2 へ
- Phase 2 で **17 t/s 以上** なら Phase 3 (Session A/B 完了待ち) へ
- いずれかで伸び悩んだら、ハードウェア側 (VRAM 24GB+ GPU) を検討

---

## 6. 次アクション (Session D として残すメモ)

1. [ ] **Phase 1 設定を LlamaDock にプリセット化** (params-schema.json / profiles.json に hot-expert 用プロファイル追加)
2. [ ] **IQ2_XS 再量子化手順を docs 化** (convert-hf-to-gguf.py + llama-quantize コマンド例)
3. [ ] **Session A/B 完了後、層非均等化の設計レビュー** (llama-graph.cpp の slot budget 計算箇所)

---

## 参照

- 計画書: `docs/outsourcing/PLAN-qwen38-20tps-delegation-2026-09-19.md`
- 現状実測: `docs/qwen4exp-practical.md`
- FreeToken 技術ギャップ: `docs/research/freetoken-gap-adoption-plan-2026-08-27.md`
- canonical ベンチ: `bench_qwen38_20tps.ps1`