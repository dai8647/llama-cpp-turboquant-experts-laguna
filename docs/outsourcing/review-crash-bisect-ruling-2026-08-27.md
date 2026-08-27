# 大型 prefill + glru クラッシュ — review 判定と bisect ラダー (2026-08-27, review side)

対象: A 報告 @901bd9a45 (large-prefill-repro-result-2026-08-27.md 上書き版) +
B 再現 doc @8b7a17c64。 本 doc は review 側の正式判定と、A/B への bisect 依頼仕様。

## 1. 2 度目実行の検証結果 = 再現として受理

review 差し戻し条件は全て満たされた:

- **pp_tokens=6575 完全一致** (B doc 1a「確実再現」条件 = 55 iter = 6575 tok)。
- **スクリプト環境の同一性を実機確認**: verify_a.ps1 (@901bd9a45) と
  verify_b.ps1 (@d7034040c) の env ブロックは完全一致
  (ROCM_PATH / HCC_AMDGPU_TARGET / HIP_VISIBLE_DEVICES / PATH /
  LLAMA_MOE_SLOT_STATS / EnvPF 既定 ''→LLAMA_MOE_PREFILL_PF 除去)。
  prefill ダブルバッファは両者とも OFF。
- **graphs 状態の差異は問題にならない**: B doc §3.2 が
  `GGML_CUDA_DISABLE_GRAPHS=1` でも同一死亡を実証済み
  (graphs 非依存)。 A の実行は graphs 既定 (93e6454f4 では
  H-1 未 pick のため自動無効化は未発火) だが、B の死亡条件域
  (graphs ON/OFF 両方) の内側なので比較は有効。
- **q* 不発の一致**: B 観測時 qstar_cpu=0 (B doc §3.4) ↔
  A 実行 qstar_xfer=0 / qstar_cpu=0。 両者とも純 glru (paging のみ) で、
  比較は同じ経路同士。

**確定事実**: 93e6454f4 先端ビルドは B の死亡条件
(slot96 + glru + 6575 tok prefill) で死亡しない。
layer 18 を通過し decode 128 tok まで完走。

**1 点だけ補完要請 (再実行は不要)**: 生存バイナリの
**ビルド元コミットハッシュを doc に明記**すること (B 受入バー⑤
「測定環境 (ビルドハッシュ/warmup/graphs 状態) を結果 docs に明記」)。
「93e6454f4 先端のビルド」の記述はあるが、バイナリ自体のハッシュ
(ビルドログ or バージョン文字列) が未記録。 bisect の端点になるため
後日の疑義防止に必要。

## 2. bisect 候補の絞り込み (review 側コード経路分析)

B のクラッシュ時代ビルド (08-25 夜以前、e7ed2fa9e「forward B's crash to A」
より前) と生存確認済み 93e6454f4 の間の全コミットを、
「q* 不発の純 glru 経路で挙動が変わりうるか」で篩にかけた。

### 形式的に除外 (根拠付き)

| コミット | 内容 | 除外根拠 |
|---|---|---|
| 84e145355 | q* threadpool alloc 失敗時 1 本化フォールバック | q* 経路のみ・q* 不発 |
| eb88c7b18, 7959cb85c, 153943603, c2911218d, 2fe902dd8, 82fb0af47 | docs のみ | ソース無変更 (eb88c7b18 の branch 再作成は docs port のみで q* 本体は base 系譜から連続) |
| 1b3569245 | [QSTAR-*] fprintf 計測 | hunk 全て qstar_cpu_exec / exec_prepare / calibrate 内・q* 不発で未実行 |
| 78c4a0486 | q* scratch dangling + view bind 修正 | hunk 全て llama_moe_qstar_alias_weight / qstar_exec_build 内・q* 不発で未実行 (calibrate クラッシュの修正であり大型 prefill クラッシュの修正ではない — A doc 153943603 §12b 自体がそう記載) |
| 3492f6f49 | ids_buf 構造体昇格 + r_max pad | hunk 全て qstar_exec_build / cpu_exec / calibrate 内・同上 |
| 5851ded96 (H-3) | speculative.cpp MoE 12 フィールド scrub | ドラフトモデル使用時のみ・verify_a/verify_b は speculative 未使用 |
| 24a70a281 | grep marker コメント追加 | コメントのみ |

### 残る候補は 2 点のみ

1. **7850c38db (F3)** — `src/llama-graph.cpp` の
   `llm_moe_gpu_slot_remap_qstar` (hunk @@331/@@347) を変更。
   remap op は q* OFF の純 glru でも毎トークン実行されるため、
   F3 の変更が純 glru 挙動に触れる可能性は排除できない。
2. **93e6454f4 (hardening = c01ea1a28 相当)** — bank_ensure での
   `ggml_backend_buffer_clear(buf, 0)` zero-init は
   **全モードの paging 経路** (slot bank 初回確保時) で実行される。
   未初期化 bank 起因の破壊/毒が死亡機序なら直接の説明になる。
   (whitelist 対抗ログ + env 上書き WARN はログのみで挙動無変更。)

A doc §3「未確定」の候補リストは hardening / H-3 / 「他の B 側 push」
だったが、**F3 (7850c38db) が候補として明示されておらず、
H-3 と marker は上記の通り形式的除外可能**。 修正推奨
(併せて typo 「a3e6454f4」→「93e6454f4」)。

## 3. bisect ラダー (最大 2 ビルド + アンカー再検証)

盲探索は不要。 以下で帰属が確定する。

### Step 0 — アンカー再検証 (B、bisect の必須前提)

B が保持する**クラッシュ時代バイナリ**を今日・同一条件で再実行し、
**まだクラッシュするか**を確認する (6575 tok = verify_b 既定 short、
slot96+glru、port 8101)。

- まだクラッシュする → アンカー有効、Step 1 へ。
- クラッシュしない → 差分はコミットでなく**環境/状態/非決定性**
  (ROCm ドライバ状態・GPU 残留・タイミング競合等) →
  bisect は中止し、新旧バイナリ各 3 回のクラッシュ率統計に切替。
  (この分岐は失敗ではなく成果: H4 競合系の仮説が前面に出る。)

B への追加依頼: クラッシュ時代バイナリの**ビルド元コミット**を
特定して報告すること (08-25 夜の観測当時 build-hip に何があったか。
6b23cabbe〜e7ed2fa9e 付近と推定されるが確定は B 記録に委ねる)。

### Step 1 — F3 時点のテスト (A、ビルド + 1 実行)

`7850c38db` (F3、hardening 直前) をビルドし 6575 tok 再現を実行。

- **クラッシュ** → 修正は 93e6454f4 (hardening) に入る
  → **帰属確定 = bank zero-init** (ログ/WARN 部分は無害なため)。
  H1/H5 系統 (未初期化メモリ破壊) 仮説が強く支持される。
- **生存** → Step 2 へ。

### Step 2 — F3 以前のテスト (A、条件付き、ビルド + 1 実行)

`3492f6f49` (F3 の直前) をビルドし実行。

- **クラッシュ** → 修正は F3 (7850c38db) = remap op 変更が
  死亡を閉じたことになる (q* 不発でも触る経路があった証拠)。
- **生存** → q* 経路のみコミット (78c4a0486/3492f6f49 等) に帰属
  = 「q* 不発なら未実行」という review 分析の前提が崩れるため、
  B のクラッシュビルドのコミット特定へ差し戻し (Step 0 の
  コミット報告と突合)。

### 実行コスト見積

- クラッシュ実行: load ~8s + 死亡 ~13s ≈ 約 25s/回
- 生存実行: prefill 6575 tok @ ~7 t/s ≈ 16 分 + decode ~90s ≈ 約 18-20 分/回
- 最悪ケース (Step 0 クラッシュ確認 + Step 1 生存 + Step 2 判定):
  GPU 約 40-60 分 + ビルド 2-3 回。 判定ステップは可能なら 2 回実行
  (非決定性保険、最低でも帰属確定ステップは 2 回)。

## 4. GO 判定 (A の次手 3 項目への回答)

1. **H-1 + H-2 cherry-pick + ビルド = GO** (従来通り承認済・本件と独立・
   衝突ゼロ実証済み 770eee998/f39abb5e7)。 注意: H-1 pick 後は
   glru 構成で graphs が自動無効化される (B §3.2 の graphs 非依存証拠に
   より再現比較への影響は無し)。 ビルド後チェックリスト
   (「CUDA graphs will be disabled」INFO / H-2 縮退 / bank 0 初期化) は従来通り。
2. **B への bisect 依頼 = GO** — ただし本 doc §3 の構成で依頼すること
   (Step 0 アンカー再検証を必ず先に、候補は F3 と hardening の 2 点、
   クラッシュビルドのコミット特定を B に依頼)。
3. **クラッシュ調査クローズ = 保留継続** (A のデフォルト提案を承認)。
   クローズ条件 = Step 1/2 で帰属確定、または Step 0 で環境/非決定性と確定。

## 5. 傍証観測 (本判定とは独立)

- **tg=1.36 t/s @ slot96+glru** (A 2 度目) vs arm c **1.44 t/s @ slot32+glru**:
  slot 数を 3 倍にしても decode はほぼ変わらず → スラッシング壁は
  slot 数では解決しない。 集計 hit 77.7% でも decode が遅いまま =
  集計命中率は decode 相のストールを説明しない (prefill 再利用が
  集計値を押し上げている可能性) → **E1 (decode 相命中率/ストール実測)
  の価値を再確認**。 FreeToken issue #174 の教訓
  (fine-grained MoE では集計命中率はクリティカルパスでない) と同型。
- h2d_gbps=2.62 (1 度目 2.63) = pageable コピー実測の再確認。
  pinned bank レバー (freetoken-16gb-community-and-mechanism-2026-08-27.md §3.2) と整合。
