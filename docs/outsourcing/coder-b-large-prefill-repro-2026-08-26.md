# 大型 prefill + glru 無言クラッシュ — 再現手順 (coder B, 2026-08-26)

coder-b-repro-prep-2026-08-26.md (@49b11ee0e) の成果物。
A が手詰まった時の保険として常設参照可。全コマンドは B 環境
(RX 7800 XT 16GB / ROCm 7.1 / gfx1101) での実績値付き。

## 0. 前提: 対象ツリーの選別

| ツリー | `--moe-qstar` | `--moe-gpu-expert-global-lru` | graphs キルスイッチ (自動) |
|---|---|---|---|
| **feat/qstar-r2-rebuild** | ○ (common/arg.cpp) | ○ (common/arg.cpp + common/common.cpp) | ○ (`graphs_disable_pending`, src/llama-context.cpp) |
| feat/prefill-double-buffer | - | ○ | ✗ |
| main | ✗ (q* drop 済) | ✗ | ○ |

→ 再現は **feat/qstar-r2-rebuild バイナリ** を使う。glru が生きている唯一の
push 済ブランチ系統であり、キルスイッチも内蔵している。
ビルドは単一所有者ルール (coder-a-build-coordination @d2016c9dc) に従い A が実施。

## 1. 再現コマンド一式

### 1a. 一次再現 (B が実際にクラッシュさせた条件 = 6575 tok)

```powershell
cd C:\Users\dai86\llama-cpp-turboquant-experts-laguna   # worktree が r2-rebuild をビルドした exe を指す場合はパス調整
# graphs OFF 版 (env 方法)
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
.\verify_b.ps1 -Tag gd_repro_env -ExtraArgs '--moe-gpu-expert-slot-num 96 --moe-gpu-expert-global-lru'
Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS
# graphs ON 版 (比較用)
.\verify_b.ps1 -Tag gd_repro -ExtraArgs '--moe-gpu-expert-slot-num 96 --moe-gpu-expert-global-lru'
```

- `verify_b.ps1` はプロンプトを内部生成する (55 iter の C++ コードレビュー風、
  **実測 6575 tok** — 外部ファイル不要)。ctx=8192 / n_predict=128 / port=8101。
- warmup (`Say hello.` n_predict=16) を自動実行後、本リクエストを投入する。
- 判定: `alive=False` + `pp/tg 空` + `crash=False` (stderr 無言) がクラッシュ再現の目印。

### 1b. bench_ft.ps1 経由 (6575 tok 系ワークロードの元祖・n_predict=256)

```powershell
.\bench_ft.ps1 -Tag ft_glru96 -ExtraArgs '--moe-gpu-expert-slot-num 96 --moe-gpu-expert-global-lru'
```

- `GGML_CUDA_DISABLE_GRAPHS=1` はスクリプト内で既定設定済み。

### 1c. bench_glru_qstar.ps1 経由 (-BinaryPath で別バイナリを指す)

```powershell
.\bench_glru_qstar.ps1 -Mode glru -Slots 96 -Rounds 3 -BinaryPath <r2 llama-server.exe のフルパス>
```

- 注意: 内部プロンプトは ~40 iter ≈ 4.8k tok と小さめで、**再現保証はない**。
  サイズ依存の切り分け (4.8k tok で死ぬか) を見る場合の補助用。確実な再現は 1a。

## 2. クラッシュシグネチャ (B 実測)

- ロード: load 6.8–8.2 秒で ready、warmup 完了 (paging 関連行も正常出力)
- 本リクエスト投入後 **約 13 秒** で死亡。materialize 連打中で、最後の進捗表示は
  **layer 18 付近**。最終 stderr 行は途中で切れている
- assertion / `hipError` / `abort` 文字列は一切なし (= crash_marker=False)
- verify_b 記録例 (bench_results.txt):
  ```
  gd_glru:     extra='--moe-gpu-expert-slot-num 96 --moe-gpu-expert-global-lru' prompt=short ctx=8192 | pp= t/s ( tok) tg= t/s ( tok) load=6.8s capture_ok=False crash=False alive=False
  gd_glru_env: 同一条件 + GGML_CUDA_DISABLE_GRAPHS=1 → load=8.2s で同一死亡
  ```

## 3. 切り分け済み事実 (B 確認済み範囲)

1. **warmup decode + paging は成功** → decode 経路の paging は健全。
   死亡は prefill ubatch 処理中に限定される。
2. **graphs 非依存**: `GGML_CUDA_DISABLE_GRAPHS=1` でも同一条件で同死
   (gd_glru vs gd_glru_env)。B の HIP graph capture 修正とも無関係。
3. **slot30 + glru 無しは完走**: gd_normal pp=172.09 t/s / tg=13.65 t/s / alive=True。
   クラッシュは glru + slot96 + 大型 prefill の組合せ条件で顕在化。
   (未記録: slot30+glru、slot160+glru。B 監査 6b23cabbe 時点では slot160 は
   ~10x slow だが完走との観測あり)
4. **q* host exec は未発火** (当時 qstar_cpu=0 を確認済み) → q* 分岐ではなく
   paging 機構側の問題の公算が強い。

## 4. B の疑い箇所と根拠

最有力: **prefill_pf_prefetch の完了 poll が `resident=true` を立てた slot に、
ensure_resident が別 expert を割り当てる競合** (仮説 1 系統)。

根拠:
- (a) 死亡タイミング (~13 秒 / layer 18 付近) が materialize + paging 高頻度区間と一致
- (b) decode 経路は prefetch 競合の機会が少なく健在 → prefill 限定の競合という整合性
- (c) graphs ON/OFF で挙動不変 → 純粋な paging パス内の破壊

5 仮説の全容は `large-prefill-crash-hypothesis-2026-08-26.md`
(A 作成)、review 側検証は `review-large-prefill-hypotheses-2026-08-26.md`
(H2 は実制御フローで解消、H4 はコード確認済み) を参照。

## 5. 再現時に採取するもの

- `vb_err.log` / `ft_err.log` の末尾 ~50 行 (layer 17–18 直前まで)
- 最後の `MoE GPU slot stats` 行 (residents / evict 推移用)
- layer 18 直前の `[q*]` 行 (あれば)
