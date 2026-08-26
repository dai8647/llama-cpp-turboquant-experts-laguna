# コーダーA へ — round-2 プラン承認 + 注意点 (2026-08-26)

qstar-round2-plan.md (6c4f1c9b3) をレビューした。**プラン全体を承認する**。
4 ステップ構成 (ASan/TSan 再現 → スタックトレース → 根本原因 → 修正/revert 判断)
で問題なし。特に step 2 の「旧 991bf3042 の `llama_moe_qstar_cpu_exec` を
単独ハーネスで最小再現」は、calibrate ループ死亡を engine から切り離して
観測できるので有効。

## 作業開始前の注意 3 点

### 1. ブランチを先に push せよ
`feat/qstar-debug-round2` (6c4f1c9b3) が **origin 未 push のローカルのみ**の状態。
作業着手前に `git push -u origin feat/qstar-debug-round2` すること。
消失リスク低減と、レビュー側からの進捗可視化の両方のため。

### 2. Debug ビルドで math_fwd.h エラーが出たら既知問題
本機でのクリーンリビルド (Release, Ninja + MSVC cl.exe) が
ROCm 7.1 ヘッダ `hip/amd_detail/math_fwd.h:88-136` の
`__attribute__((pure))` で C2059 多発して停止することを確認済み。
**q* とは無関係の既存問題**。対処と回避案は:
`docs/build-issues/rocm-71-msvc-attribute-pure.md`
Debug + ASan ビルド (`build-hip-debug`) で同一エラーに当たった場合は
同 doc の回避策 (HIP clang 経路 / -O0 -g + printf トレース代替) に切替。

### 3. 受入バーは監査プロトコル準拠で固定
再 main 提案の条件は plan step 4 記載の通り、かつ以下を厳守:
- `qstar_cpu > 0` をログで観測していること (ゼロなら不採用 = 寄与ゼロ)
- 短文: 別プロンプト 3 種 × 3 ラウンド、1 ラウンド目破棄
- 長文 (6575 tok): REQUEST-FAILED なしで完走
- Ornith (type=11 統一) も同条件で通過
- 測定環境 (ビルドハッシュ / warmup 手順 / graphs 状態) を結果 docs に明記

## 大型プレフィル + glru クラッシュも今ラウンドで

B 発見の「6575 tok + slot96 + glru で materialize 中 ~13 秒・layer 18 無言死亡」
(`docs/outsourcing/coder-a-bug-large-prefill-2026-08-25.md`) も A 担当。
q* debug と同じブランチでも別ブランチでも良いが、進捗は
`docs/outsourcing/` に逐次書くこと。手順メモ:
- 再現は graphs OFF 2 方法どちらでも出る (新キルスイッチ + GGML_CUDA_DISABLE_GRAPHS=1)
- warmup decode + paging は成功する → prefill-paging 特化
- 疑い箇所: prefill ubatch 中の `ensure_resident` / `evict` 競合

revert 判断になった場合も、理由を文書化すればそれはそれで正しい成果。
単独再実装をする場合は設計レビューを先に (§8 mutex 保持時間契約 +
§9 per-context graphs スイッチが前提条件)。

— レビュー側 (ZCode) / ユーザ経由
