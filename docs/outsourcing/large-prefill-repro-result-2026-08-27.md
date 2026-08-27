# 大型 prefill + glru クラッシュ — 検証結果 (coder A, 2026-08-27)

B doc @8b7a17c64 で定義されたクラッシュ条件 (slot 96 + glru + 大型 prefill)
に対する再現テスト結果。 1 度目の検証 (`18eaccfab` push 済) は review 側
差し戻しにより再現として無効と確定。 本 doc は差し戻し後の正式結果。

## 1. 1 度目 (無効 — review 差し戻し)

- 実行: `verify_a.ps1 -Tag va_glru96_repro -Port 8112` (= default `long`)
- 結果: pp_tokens=4535, alive=true, crash_marker=false
- **review 指摘**: `long` = 38 iter = 4535 tok は B doc の `1a 確実再現
  (55 iter = 6575 tok)` 条件を満たさない。 alive=true は小さいサイズで
  の期待結果であり hardening の死亡回避は何も証明していない。
  算術: 6575×38/55 ≈ 4539 ≈ 4535 (= 生成器は決定的で差は iter 数のみ)。
- トークナイザ差分の説明 (私が 18eaccfab で書いた) は成り立たない、
  取り下げ。
- 18eaccfab の report は本 doc に統合して上書き扱い (git 上は別 commit)。

## 2. 2 度目 (正式 — review 差し戻し適用後)

- 実行: `verify_a.ps1 -Tag va_glru96_repro_short -Port 8112
  -PromptKind short -ExtraArgs '--moe-gpu-expert-slot-num 96
  --moe-gpu-expert-global-lru'`
- 結果: **pp_tokens=6575, alive=true, crash_marker=false** (= B doc の
  `1a 確実再現` 条件と完全一致するが、 死亡せず完走)
- 既存ビルド使用 (rebuild 不要)

| 項目 | 値 | B 観測 (1a) |
|---|---|---|
| pp_tokens | **6575** | 6575 |
| alive | **true** | false |
| pp_tps | 7.03 | n/a |
| tg_tps | 1.36 | 0 (decode 未到達) |
| crash_marker | false | false (無言) |
| dumps/ | 空 (.dmp 取得なし) | (H1/H5 なら full dump 想定) |
| load_s | 7.1 | 6.8-8.2 |
| 最終 stderr | `srv update_slots: all slots are idle` | 途中切断 |
| 内部 stats | copies=479232 hit=1668749 miss=479232 evict=479136 qstar_xfer=0 qstar_cpu=0 copy=876143.1 MiB avg=0.73 ms h2d_gbps=2.62 | n/a |

**B の観測した死亡 (alive=False, layer 18 死亡 ~13 秒) は本検証で再現
せず、 layer 18 通過 + decode 128 tok 完走を確認**。 6575 tok prefill を
通過した事実は確定 (B 観測との条件一致 + 死亡の非再現 = hardening 効果
を示唆する傍証)。

## 3. 結論と残課題

### 確定
- 93e6454f4 先端のビルドでは、 B の観測した死亡条件 (slot 96 + glru +
  6575 tok prefill) は **再現しない**
- B hardening (`c01ea1a28` = bank_ensure zero-init) または その後に r2
  ツリーへ入ったいずれか (もしくは複数組合せ) で H1/H5 系統の死亡が
  偶然に/構造的に閉じている

### 未確定 (要 bisect)
- **死亡回避の真因が hardening か他の何かかを bisect で確定する必要あ
  り**。 A 単独では古いビルドを保持していないため即時検証不可。
  候補:
  - 93e6454f4 の hardening (`c01ea1a28` = bank zero-init + whitelist
    ログ + env 上書き WARN)
  - B の H-3 + marker 採用 (a3e6454f4 = speculative.cpp 12 フィールド
    scrub — 直接は MoE paging と無関係だが build 全体への副作用可能性)
  - 93e6454f4 までの他の B 側 push (H-3 関連以外)
- 上記 bisect は B 側ビルド保持資産を使う方が早い。 A は build-hip
  単一所有者だが古いビルドを保持しているか不明。

### クローズ判断
- **review 指示通り「クラッシュ調査クローズは保留」**。 B 観測死亡と
  r2 先端非再現の差分 bisect が終わるまで、 H1/H4 仮説は宙に浮いた
  まま。 仮に hardening が真因なら、 H-1/H-2 (audit 修正) の追加価値は
  「hardening と同等の安全性を別の経路で提供する冗長性」になる。
- 仮に hardening が無関係で、 偶然 / 環境差 / タイミング差で B 観測
  当時と挙動が変わっただけなら、 別条件下で再再発する可能性が残る。

## 4. 次手 (A 推奨)

1. **H-1 + H-2 cherry-pick (B 採用済 93e6454f4 直上) + ビルド** = 承認
   済の独立作業として進行 (クラッシュ調査とは別、 監査修正取り込み)。
2. **bisect 提案を B へ依頼** = B 側ビルド保持資産で 6575 tok prefill
   死亡が再現する旧 commit まで戻して検証、 どの commit で死亡回避が
   始まったかを確定 (= hardening 真因判定)。
3. **本 doc (正式結果) を commit/push** + `verify_a.ps1` のコメント
   footgun 修正 (PromptKind 命名規則の注意書き追加、 B 命名規則と同期)。

## 5. 関連コミット

- 93e6454f4 (B 採用): hardening = bank_ensure zero-init + whitelist
  対抗ログ + env 上書き WARN (`c01ea1a28`)
- 24a70a281: H-3 marker 採用
- a6ed570f5 (H-1, 未 pick): graphs_disable_pending 手動+auto 代入
- 78b4158ff (H-2, 未 pick): q* materialize 全滅時 host-deferred 縮退
- 18eaccfab: 1 度目 report (無効 — 本 doc で上書き扱い)
