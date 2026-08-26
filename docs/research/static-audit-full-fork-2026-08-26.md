# フォーク全体静的監査報告 (2026-08-26 深夜, レビュー側)

宛先: user / cc: コーダーA・B
目的: FreeToken 思想をさらに取り込む前に、本家 llama.cpp からの全分岐改変を静的に点検し、
現役資産・死蔵コード・相互干渉・地雷を数字と file:line で確定させる。

## 0. スコープと方法

- 分岐点: `cbc360d19` (2026-07-24, "experts-first MoE + Laguna arch + frequency placement")。
  以降 **dai8647 名義 143 コミット (うち純コード系 ~115)、改変ファイル 242**。
- 方法: レビュー側が slot cache/glru/q*/prefill-PF/graphs 系を直接深読み +
  並行監査エージェント 2 体 (動的配置系 / spec・KV系) + turbo 量子化はエージェント 2 回落丁のため縮小自己監査。
- 対象ツリー: `feat/qstar-r2-rebuild` @ 2fe902dd8 (main 同期済みの q* 再構築線)。

## 1. 概念インベントリ総表

| # | 概念ファミリー | 主要フラグ | 判定 |
|---|---|---|---|
| 1 | expert slot cache 本体 | `--moe-gpu-expert-slot-num` / `auto` | **現役** (プロジェクト核) |
| 2 | global-LRU プール | `--moe-gpu-expert-global-lru`, env LLAMA_MOE_GLOBAL_LRU | **現役** |
| 3 | q* bandwidth-adaptive split | `--moe-qstar*` 3種 | **現役** (round-2 再構築後) |
| 4 | prefill 二重バッファ | env LLAMA_MOE_PREFILL_PF/MB/INFLIGHT (**CLI フラグ無し**) | **現役** |
| 5 | elastic VRAM | `slot-num auto` + env AUTO_SLOT_CAP/AUTO_VMARGIN | **現役** |
| 6 | frequency 配置 (動的エキスパート第1世代) | `--moe-expert-placement frequency` + freq-report in/out | **半死蔵**: Pass1 収集は server のみ保存可、Pass2 whitelist は手動 slot パスのみ。`is_in_frequency_whitelist()` は呼び出し元ゼロ |
| 7 | placement その他の値 (`cpu-moe`,`map`) + `--moe-expert-map-path` | — | **死蔵** (検証されるだけで未消費) |
| 8 | TurboQuant KV 量子化 (TURBO2_0/3_0/4_0 = 型ID 47/48/49, PolarQuant+QJL, arXiv 2504.19874) | `--cache-type-k/v turboN` | **現役**。CPU(ggml-turbo-quant.c 37KB)+CUDA(turbo-quant.cuh+fattn template instances)+Metal(turbo-wht.h)+Vulkan(copy_to_quant.comp) 全 4 バックエンドに実装。128×128 Walsh-Hadamard 回転テーブル (src/turbo-rotation-data.h 4103行) は kv-cache で共有生成 |
| 9 | speculative 拡張 (~40 flags, common/speculative.cpp 2787行) | `--spec-type` 11種, ngram-mod/simple/map-k/k4v, draft-* | **概ね現役**。ngram-mod が最完成。死蔵・半端あり (下表) |
| 10 | DeepSeek-V4 KV / MTP / DFlash (kv-cache-dsv4 2212行) | — | 上流 PR 取り込みが主体、フォーク手直しは軽微 |
| 11 | CUDA カーネル改変 (argsort/top-k hipcub 化, getrows kq) | — | **現役** (HIP/gfx1101 安定化目的, MoE ルータ用) |

## 2. High (早めの対処を推奨)

### H-1: CUDA graphs 自動無効化は「実装されたことにになっているが存在しない」
- コミット 292864d86 / 0ef48796b のメッセージは「q\*/glru 有効時、モデルロードが
  `graphs_disable_pending` を立て、context が CUDA バックエンドの capture を無効化する」と主張するが、
  **`git log --all -S "graphs_disable_pending = true"` = 0 件**。代入文は歴史上一度も書かれていない。
- 実態: `src/llama-context.cpp:468` の適用ブロックは永遠に不発の死コード。
  唯一の防御は `src/llama.cpp:1990-1994` の WARN (env var 設定を促すだけ) と
  `ggml-cuda.cu:2870` の map_custom capture bail-out (ただし remap op は CPU バックエンド行きのため通常発火せず)。
- 影響: A/B が信じている「glru 使用時はコード自身が graphs 強制無効化」は誤り
  (A が確認したのは警告ログのみ)。env var を付け忘れると graphs+paging が素で走る。
  **P1b アーム (graphs 解禁) はこの組合せの事実上初の計測になるため注意**。

### H-2: q* dummy-slot 強制転送経路が OOM 時に abort (レビュー側・監査エージェントが独立に同一検出)
- `src/llama-graph.cpp:394-402`: 層内で何も resident になれない時、deferred 先頭を強制転送して
  未初期化 VRAM 読みを防ぐ設計自体は正しい。しかし最後の `GGML_ASSERT(slot>=0 && s->resident)` は、
  VRAM 枯渇で materialize が全部失敗した場合に推理中 abort する。
  `safe_unbanked_fallback` (llama-model.h:1380-1385) が resident 0 の状態で返す 0 は resident=false。
- 推奨: assert を「mask 全部 0 + 全 miss host exec」への縮退に置換 (q* 自体が成立しない状況なので安全側へ)。

### H-3: speculative が MoE フラグをドラフトモデルに漏洩させる
- `common/speculative.cpp:2310` `common_params result = params;` が**全コピー**のため、
  `--moe-gpu-expert-slot-num/--moe-qstar/--moe-expert-placement/freq-report` 等がドラフト側にも適用される
  (置き換えられるのは devices/model/ngl/overrides/threads のみ — :2313-2321 実読確認)。
- ドラフトが dense なら「no MoE experts; ignoring」で無害だが、**DeepSeek-V4 Flash のような MoE ドラフト
  (DSpark/MTP 系) ではドラフト ctx でも expert paging が二重起動しうる**。意図仕様の記録なし。
- 推奨: 継承 whitelist 化 (moe_* を明示的に result から消すか、意図するなら文書化)。

## 3. Med (運用罠・サイレント無効化カタログ)

| # | 現象 | 根拠 |
|---|---|---|
| M-1 | B0 ゲート沈黙: slot<experts・whitelist 空・track_access 無・glru/q* 無 → キャッシュ完全無効、警告なし (INFO のみ)。BENCH_RESULTS B0「実質スロットレス」の正体 | src/llama-graph.cpp:2427-2439, llama-model.h:1310-1313 |
| M-2 | fingerprint 不一致 → whitelist 破棄+full-slot フォールバック (WARN は mismatch のみ)。explicit slot<experts と重なると M-1 と重畳して実質無効 | src/llama.cpp:1927-1931 |
| M-3 | ratio→gpu_count は層合計の大域値。whitelist>slots だと頻度順に埋めて冷たい expert が残存スラッシング。「ratio=0 でも 1 件は GPU 行き」 | src/llama.cpp:1933-1934 |
| M-4 | slot auto パスは freq-report/placement/CLI q* を一切読まない (env のみ)。同一フラグでも auto vs 明示で挙動が変わる | src/llama.cpp:1107-1215 |
| M-5 | 「CLI flags win」コメントと逆に、env (LLAMA_MOE_QSTAR_THREADS/BUDGET_US) がフラグ後に無条件上書き | src/llama.cpp:1960-1975 |
| M-6 | whitelist 非空なら glru cross-layer eviction は自動無効 (`global_lru && whitelist.empty()`)。初期化ログは「global LRU enabled」と出し続ける | src/llama-model.h:1331, llama.cpp:1904 |
| M-7 | `--spec-ngram-simple-min-hits` 解析されるが未消費 (docs には掲載)。`-lcd` は save 経路恒久無効で説明と不整合。spec-draft cpu-mask/range/prio/poll は server 未接続 (examples のみ有効) | ngram-map.h:24-27, speculative.cpp:2186-2187, server-context.cpp (attach なし) |
| M-8 | `--spec-type draft-eagle3/dflash` 等を draft model 無しで指定 → 警告ゼロで spec 無効 | server-context.cpp:1091-1095 |
| M-9 | freq report 書き込みは server シャットダウン時のみ。llama-cli では保存フック不在 → tests/test_moe_frequency_placement.py の collect テストは成立しない可能性大 (要実行確認) | server-context.cpp:999-1016 |

## 4. Low / 死蔵コード一覧

- 死蔵: `--moe-expert-map-path`, placement=`cpu-moe`/`map` 値, `is_in_frequency_whitelist()` (llama-model.h:900),
  `uses_compute_tensor()` (:1003), `estimated_weight_bytes` (常時 0, llama-moe-stats.h:12), `graphs_disable_pending` (H-1),
  `--spec-draft-p-split` (examples のみ)。
- Low: access_counts.size() の mutex 外読み (src/llama.cpp:2264, 終了時とはいえ規約違反) /
  find_touch→ensure_resident 再探索で n_hit 二重計上 (統計歪み, llama-graph.cpp:352-358) /
  eagle3・MTP の malloc null チェック欠如 (common/speculative.cpp:489,1343) /
  MTP-only 時ユーザの -ctk/-ctv が draft 既定 F16 で無視され得る (speculative.cpp:2299-2325 は has_draft 無関係に上書き) /
  `--moe-gpu-expert-ratio` stof try/catch なし (arg.cpp)。
- 品質良好の記録も残す: slot_at/release_slot は範囲チェック完備、cache_mutex→access_mutex の一方向ロック順と
  hold-time 契約は model.h:818-835 に明文化、prefill-PF の slot_busy drain (llama.cpp:1046-1061) は
  in-flight copy 破壊を正しく回避、TurboQuant の WHT group-size global 漏れバグは既に構造修正済み (ggml-turbo-quant.c:22-29)。

## 5. 干渉マトリクス (優先順位の確定)

1. slot mode (slot-num>=0/auto) は `-ngl` を無視して expert tensor を強制 CPU 配置 (llama-model.cpp:1310,1726)
   → `--cpu-moe` (-cmoe, buft_override 経由) と同方向ゆえ共存可能 (実運用も併用)。
2. placement=frequency + slot 未指定 → INT32_MAX 注入 (full-slot+whitelist)。
3. whitelist 非空 → glru 実効無効 (M-6)、collection-skip 解除。
4. q* decode は has_paging_dynamic かつ decode (n_tokens==1) かつ n_expert_used>=2 でのみ活性 (llama-graph.cpp:2441-2445)。
5. spec は別レーンだが H-3 の漏洩経路で接続する。

## 6. 推奨処置 (優先度順)

1. **H-1 修正**: `graphs_disable_pending = true` を q*/glru 有効時の load パスに 1 行ずつ追加
   (llama.cpp:1902-1912 と 1963-1967 の近傍)。または警告文言を「必須」である旨に強化。~5分の修正。
2. **H-2 修正**: dummy-slot ASSERT を mask 全 0 縮退へ (VRAM OOM = q* 不成立状況の安全側)。
3. **H-3 修正**: `common_base_params_to_speculative` のコピーから moe_* を除去 (MoE ドラフトを使う日前までに)。
4. **M-4/M-5**: auto パスの flag 読み追加 or env>CLI の仕様確定とコメント修正。制御面 (CLI 13 flag + env 13種) の優先順位を docs に一枚化。
5. **死蔵掃除**: 第6世代 frequency 配置の残骸 (map-path, is_in_whitelist 等) は「FreeToken 前史」として
   docs に一言書いて削除候補に。M-7/M-8 の docs 不整合も同時に。
6. **turbo 量子化の残検証** (今回縮小監査の穴): バックエンド×op の完全マトリクスと
   fattn template instance の型組合せ網羅性は次回 GPU 不要作業で追加確認を推奨。

## 7. FreeToken 方向との関係 (user の問いへの答え)

- **足場として使える現役資産**: slot cache / glru / prefill-PF / elastic VRAM / q* (再構築後)。
  コード品質は防御的で契約文書化も進んでいる。FreeToken §2.1.2 (デスクトップ GPU は miss を PCIe で)
  と現在の転送路設計は整合。
- **足かせになるもの**: H-1〜H-3 と M-1〜M-6 のサイレント挙動群。特に「graphs 自動無効化があると思っている」前提で
  ベンチを組むと測定が静かに崩れる (P1b がまさにその危険地帯)。
- **用済み寄り**: frequency 配置第1世代 (Pass1/Pass2) は q*/glru に機能を譲った形跡がコード上明白
  (whitelist は事前配分のヒントに降格、residency 判定には使われない)。削除ではなく docs での位置づけ固定を推奨。

## 8. 不明点 (次の監査 or 実測で)

- test_moe_frequency_placement.py が実際に FAIL するか (実行禁止につき未確認)。
- turbo 量子の backend×op 完全マトリクス (§6-6)。
- getrows.cu kq の block_dim 64 契約 (QK_K との整合)。
- graphs ON + paging の実際の凍結リスク (コメント主張 vs stage-c 合格実績の矛盾) — P1b の実測待ち。

---
監査: レビュー側 (ox-alpha) / 直接読解: llama.cpp, llama-graph.cpp, llama-model.h, llama-context.cpp,
ggml-cuda.cu, ggml-turbo-quant.c / 監査エージェント: 動的配置系 (完了), spec・KV系 (完了),
turbo量子系 (ネットワーク障害で 2 回失敗 → 縮小自己監査で代替)
