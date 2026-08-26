# q* / glru decode 同期コピー壁の突破オプション — review 側フェーザビリティノート

- 作成者: reviewer AI(提案のみ・未実装)
- 日付: 2026-08-27
- 根拠: F3 実測 (calibrate h2d=3.8 GB/s / cpu=1.1 GB/s, q* ON 1.54 t/s vs 対照 4.34-4.50),
  監査 (@071e753e0), 残穴追補 (@58b2572cb), P1 確定値 (main @1da028abc:
  stage-a 12.28-13.48 / graphs ON 13.88 / glru 構成 1.44 t/s)
- 性格: **実装計画ではない**。オプション整理と判定ゲートの提示。エンジン側設計の
  最終判断は A。着手順序は大型 prefill glru クラッシュの切分け確定後を推奨(同領域)。

---

## 0. 壁の定義

glru/q* による動的 paging では、decode 各 token・各 MoE 層について
「router 出力で miss 判定 → **同じ remap op 内で H2D 同期コピー → 継続**」という逐次鎖になる。
限界推定は過去に複数回一致:
`壁 ≈ 層数 × n_expert_used × expert_bytes ÷ B_h2d`(36 層 × r8 × ~2MB ÷ 3.6GB/s ≈ 163 ms/token ≒ **~6 t/s 上限**)。
F3 後の 1.54 t/s はこの先さらに CPU exec 混在の損失。一方、paging 自体をしない構成
(stage-a 相当)は 12-13 t/s 実績。**q*/glru の価値条件 = この差分を、VRAM に入らない
場合にどれだけ埋められるか**。

## 1. 既存インフラの棚卸し(全部実在・調整済み)

| # | 資産 | 位置 | 本提案での役割 |
|---|---|---|---|
| 1 | `take_last_selections()` — 直近 token の層別選択 id取得 API | llama-model.h:802-811(remap op 内 `record_selections`:792 で記録) | 「次 token の当たり予測」の供給源 |
| 2 | inter-step 予測 prefetch(**同期**) | llama.cpp:891 〜(LLAMA_MOE_PREFETCH_MS, ctx 内 decode 後) | 存在するが下記 O-B の通り**壁は消えない** |
| 3 | prefill-PF 非同期機構 一式 | llama.cpp:965-1083(event 生成・`ggml_backend_cuda_ext_event_*` h2d_async enqueue・1 lock hold 契約・次 chunk 冒頭で完成拾って resident 反転) | **O-C の心臓部をまるごと再利用可能** |
| 4 | q* の mask/host-exec 分離 | llama-graph.cpp:276-424(split 判定)+ exec_op 位置対応 | O-C でも「当日読めなかった分」の正しい値を担保できる |

## 2. オプション評価

### O-A. 静的回避(ratio/whitelist/full-fit)— 対照であり続けるべき
P1 が示したとおり、page しなければ 12-13 t/s。VRAM に入る構成なら素朴な all-gpu +
whitelist が常に最速既定。**动 paging が必要なのは「full-fit できないが部分 fit なら
余裕がある」領域のみ**。以降のオプションの価値はこの領域で 12 t/s 帯に届くか。

### O-B. inter-step 同期 prefetch 強化 — 壁は消えない(要留意)
llama.cpp:891 の現行 prefetch は「ステップ間(GPU 遊休中)に前 token の選択を転送」。
ここで転送する byte は、何もしなければ次 token の remap op 内で壁として払っていた
byte と同一だから、**総転送量不変・時間も移動して来るだけで減らない**
(hit すれば当日の sync 分はゼロになるが、prefetch 分が丸ごと同じ時間先払いされる;
GPU が別仕事をしていない以上純ゲインほぼ 0、ホスト側 sampling 数百 µsとの僅かな
重畳しか得られない)。「prefetch ON で速くなった」と感じたら実は他要因を疑うこと。

### O-C. **decode-PF(prefill-PF 機構の decode 移植)— 本命**
骨子:
1. token n の forward 完了後(ctx デコード後フック、既存 :2064 相当の場所に併設)、
   `take_last_selections()` のうち当日 miss だったもの(または resident 外だったもの)を
   資産#3 と同じ手順で copy stream へ非同期投入(inflight 登録、resident=false のまま)。
   1 lock hold 契約(per-expert 短保持)もそのまま踏襲。
2. token n+1 の remap op 冒頭で該当層の inflight を完了ポーリング(`slot_busy drain`
   と同一パターン)してから split 判定 — 到着済みは「hit 扱い」になり、
   当日の sync 転送がその分**本当に消える**。
3. 到着していなければ現行同様(q* なら transfer-or-host の既存判定に自然フォールバック)。
   **尾行動作で正しさは壊れない**:未到着 = 既存経路そのまま。
4. q* 併存時の注意: split 判定は「到着済み hit」を知らないでよい形に一本化する
   (find_touch が drain 後に走れば後方互換)。

真となる削減量:`wall_h ≈ (1−h) × 全 miss bytes ÷ B_h2d`(h = 予測命中率)。
B_h2d=3.8GB/s・全 miss 163ms/token の設定で、h=0.5 ⇒ ~82ms ⇒ 単体で ~12 t/s 帯、
h=0.75 ⇒ ~41ms ⇒ 余裕を見て 15 t/s 方向。**少なくとも 1 token 分遅延で価値が出る
ターゲット帯(np=1 会話用途)なら理論上届く**。h の実測なしには GO しない、が結論。

### O-D. 層内先行転送 — 原理却下
同一 forward 内で layer L+1 の router 出力は L 計算前に存在しない =
転送対象不明。graph トポロジ変更で先行 exec してもデータ依存は解消されない。

### O-E. bank 側バイト削減(TQ3_1S/TQ4_1S への重み再量子化等)— 将来路線
転送 byte 自体を落とす方向。既存 followup doc の通り TQ\*\_1S は weight 用として
実装済み(MUL_MAT/GET_ROWS 対応)。精度影響の検証コストが大きいため本フェーズ外。

## 3. 推奨判定ラダー(review 提案)

- **E1(計測先行・diff 最小)**: 現行 inter-step prefetch のログに「予測 id 群のうち
  次 token で実際 hit した率 h」を [QSTAR-] 同型 1 行で吐くだけの instrument。
  GPU 枠ほぼ不要(任意の glru リクエスト 1 回)。ベースライン用途にも使えるので
  B の P2 ランナーへの統合が相性良。
- **E2 ゲート**: O-C 実装 GO の物差しは
  `(1−ĥ)×miss_bytes/B_h2d + 固定費 ≤ 目標 83ms(=12 t/s)` を実測 ĥ で判定。
  ĥ < ~0.35 程度なら、まず whitelist/ratio 増強(O-A 調整)の方が速い、という切り替え
  判断まで含める。
- **着手順序**: 大型 prefill glru クラッシュ(A 現行スレッド)確定 → crash fix push →
  audit-fixes 取り込み(H-1/H-2)→ この方向の設計合議 → E1。

## 4. 既知リスク・契約面

- prefill-PF が持つ契約群(cache_mutex per-expert 短 hold、resident=false 延期、
  slot_busy drain 同一パターン)は decode 側でも同じ前提で守れる — model.h:818-835 の
  文書を更新して差し込む形で矛盾を出さない。
- 大型 prefill + glru クラッシュ(unresolved)と同 code region を触るため、
  triage 前に実装着手すると障害分離が汚れる。順序厳守。
- CUDA graphs: graphs_disable_pending(a6ed570f5)により q*/glru 時は capture OFF が
  自動化済みなので、非同期 enqueue + 後追い remap の組合わせは replay-freeze 問題を
  新規には招かない。
