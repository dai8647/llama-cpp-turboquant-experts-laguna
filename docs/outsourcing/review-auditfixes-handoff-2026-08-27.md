# 監査修正 3 点の実装 (review/audit-fixes ブランチ)

- 作成者: reviewer AI
- 日付: 2026-08-27
- 対象: 静的監査 `docs/research/static-audit-full-fork-2026-08-26.md` (@071e753e0, main) の High 3 件
- ブランチ: **`review/audit-fixes`** = `origin/feat/qstar-r2-rebuild` (82fb0af47) + 以下 4 コミット
  - `a6ed570f5` moe : actually wire graphs_disable_pending for q*/global-LRU paging
  - `78b4158ff` moe : q* degrade to all-host-exec when no slot can be made resident
  - `7cb31d003` spec : stop leaking target MoE paging flags into draft params
  - `c01ea1a28` moe : harden slot cache edges flagged by the static audit (**追加・optional**)

A の feat/qstar-r2-rebuild ツリーは直接触っていない。マージ可否・マージ先
(r2 ブランチ or main) の判断は A に委ねる。

---

## 1. graphs_disable_pending を実際に立てる (H-1)

**問題** (監査目玉 #1): フラグ宣言 (`llama-model.h`) も消費側
(`llama-context.cpp` ctor 内 `ggml_backend_cuda_ext_set_graphs_enabled(backend,
false)`) も存在したが、代入するコードが歴史上ゼロ件
(`git log --all -S "graphs_disable_pending"` 空だったのを、commit 292864d86 /
0ef48796b のメッセージ「q*/glru 時 CUDA graphs 自動無効化」が実態と違った)。
唯一の保護は llama.cpp の WARN。つまりこれまで glru/q* + graphs ON で走ると、
per-step の slot/mask 決定がウォームアップ時点に凍結されたグラフを素で再生していた。

**修正**: 有効化 2 箇所で代入するようにした。
1. 手動初期化パス: `moe_qstar` / `moe_gpu_expert_global_lru` 判定ブロック末尾
   (旧 WARN ブロックを置換)。CLI / env どちら由来でも通る。
2. `llama_moe_gpu_expert_slot_auto_init()` 末尾: env 経由の
   `LLAMA_MOE_GLOBAL_LRU` / `LLAMA_MOE_QSTAR` 用。この関数は ctx ctor 内で
   フラグ消費チェックの直前に呼ばれるので順序は安全。

旧 WARN 文内の「per-context API が無い」コメントは虚偽になっていたため撤去。
`GGML_CUDA_DISABLE_GRAPHS=1` 併用も従来通り有効 (process-wide kill)。
ログは WARN→INFO 変更 (自動処理されたことを INFO で明示する方が運用上正しい)。

## 2. q* dummy-slot ASSERT 死の縮退運転化 (H-2)

**問題**: `llm_moe_gpu_slot_remap_qstar()` は全 selection が初期化済みスロットを
読む保証として deferred 先頭 1 エキスパートの強制 materialize を行うが、それすら
失敗 (過渡的 VRAM 枯渇など) すると GGML_ASSERT で推論中プロセス死亡。

**修正** (`src/llama-graph.cpp`): 最後の確保も失敗した場合、

- `cpu_ok == true` → 全 selection を host-exec に退避して生成を続行:
  - `qstar_deferred` を空にして位置順に全 r 件入り直し (mask 全部 0)
    ※ exec_op 側の「k 番目 mask==0 ↔ deferred[k]」位置対応と整合
  - host stage が元テンソルから層全体を再計算するので出力値は正しい
  - mask 済み bank 読みは slot 0 へ向く (=未初期化 VRAM) が router weight は
    0 に zeroing 済み → 他の host-deferred selection と同種の扱い
  - WARN は `static std::atomic<bool>` で 1 回のみ (OOM 状態は token ごとに再発)
- `cpu_ok == false` → 正しい計算経路が存在しないので従来通り fail-loud (ASSERT 維持)

**残存リスク (正直に明記)**: 未初期化 bank データ × weight=0 は、そのデータが
dequantize で NaN/Inf にならない限り安全。完全閉じるなら bank 割り当て時の
zero-fill (bank_ensure で 1 回 memset、ロード時に数 ms 相当) という拡張があるが、
差分範囲を最小にするため今回は入れない。必要なら次フェーズで。

## 3. ドラフト params への MoE フラグ漏洩遮断 (H-3)

**問題**: `common_base_params_to_speculative()` が `common_params result =
params;` で全コピーするため、ターゲット側の MoE paging 設定 12 種
(slot num/auto/placement/global_lru/qstar*/ratio/freq_report×3/map_path) が
ドラフト context に漏れる。server-context.cpp がこの関数を 2 箇所で使用。
dense ドラフトは無害だが、MoE ドラフト (MTP/DSpark 系) ではターゲット横並びの
第 2 paging インスタンスが黙って起動しうる。

**修正** (`common/speculative.cpp`): コピー直後に 12 フィールドすべてを
disabled デフォルトへリセット。ドラフト CLI にはこれらの表面がないため、
デフォルト以外の値は純粋な漏洩に相当 (意図的な継承は存在し得ない)。

## 影響範囲 / 自己検証の範囲

| 修正 | q* 無効時 (P1 arm a/b/c 含む) | q* 有効時 |
|---|---|---|
| H-1 | pending は立たない・挙動変化ゼロ | ctx 構築時に graphs 自動 OFF (以前は手動 env 必須) |
| H-2 | コード到達不能 (q* remap op 内のみ) | 通常動作は現行と同一。materialize 全滅時のみ ASSERT→縮退 |
| H-3 | spec 未使用なら影響なし。dense ドラフトは挙動同一 | MoE ドラフトで二重 paging 不発に |

- 検証済み: diff 全文自己レビュー・フィールド名対照 (common.h:465-476)、
  exec_op の位置対応規則との整合確認、P1 プロトコル非干渉確認
  (arm a/c は glru/q* 禁止なので H-1/H-2 とも dead path)。
- **未検証**: 実ビルド・実行。含まれる exe は一切無い。挙動確認は A の
  次回ビルド以降に行うこと。まず見るべきログ:
  - H-1: `CUDA graphs will be disabled for this context` (INFO)
  - H-2: `running all N selections on the host` (WARN, 1 回のみ)
- compile リスク低評価: 追加 include `<atomic>` のみ・新規シンボル参照なし
  (フィールド/関数は既存のもののみ参照)。

## P1 ベンチへの影響

なし。GO 継続。B の p1_stage_a_bench.ps1 は [-ngl 999 ... --cpu-moe] で
glru/q* 不使用の Arm a/c が主であり、ここで今回のコードは一切発火しない。
F3 版 exe 承認 (@1994008ee) も変更なし。今後ビルドされる exe は
このブランチが入っていても受け入れ可能 (diff は上記の縮退系のみ)。

---

## 追記 (2026-08-27): c01ea1a28 = 監査 Med 由来ハードニング(任意採用)

A の H-1/H-2 cherry-pick 計画には影響しない追加コミット。pick するか否か
は自由(単体で完結、3 変更とも src/llama.cpp):

1. **bank_ensure で bank バッファ zero-init** — H-2 に残した「未初期化 bank ×
   weight 0 が NaN dequant されたときだけ毒」という残余リスクを構造的に閉じる。
   `ggml_backend_buffer_clear` は全実バックエンド実装済み確認済み
   (CUDA/HIP/Metal/Vulkan/CPU 他・`.clear =` 全列挙 grep)。
2. **frequency whitelist 適用時に対抗ログ**「global-LRU pool is inactive」—
   cache 内部では whitelist 非空で global 縮退(モデル.h:1331 の
   `global_lru && frequency_whitelist.empty()`)だが、直前の
   「global LRU slot pool enabled」ログと矛盾する状態を黙認していたのを正直化。
3. **env 上書き WARN** — LLAMA_MOE_QSTAR_THREADS / BUDGET_US が既解決値を
   置き換えるときに明示的に警告(旧来の無言上書き・「CLI flags win」コメント矛盾の
   半分だけを塞ぐ安全側修正。優先順位自体の変更はしない)。
