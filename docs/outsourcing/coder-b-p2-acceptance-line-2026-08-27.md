# P2 受入ライン数値化 — q* 価値基準のパスライン化 (coder B, 2026-08-27)

review 側から B に委譲された領域。P1 確定値 (coder-b-p1-results-2026-08-26.md @1da028abc) を
基に、q* が「価値がある」と言えるための数値パスラインを定義する。

## 1. P1 確定ベースライン (再掲)

| arm | 構成 | 短文中央値 tg | 長文 6575 tok |
|---|---|---|---|
| a | slot96, glru/q* 無し, graphs OFF (CPU 直計算経路) | **12.28–13.48 t/s** | pp 164.5–188 / tg 12.33–13.37 完走 |
| b | slot96, glru/q* 無し, graphs ON | **13.88 t/s** | pp 197.8 / tg 13.67 完走 |
| c | slot32 + glru, -c 8192 -t 8 (A 対照推定) | **1.44 t/s** | prefill 1.2 tok/s で 2048/6575 停止 |

## 2. q* 価値基準の構造的定義

review-p1-baseline-decisive-bench §4 で「q* の価値基準 = 12.91 超え」に確定したが、
監査 12.91 は **glru/q* 無しの CPU 直計算経路**の数値 (arm a/b)。q* が設計上
効くべきは **glru スラッシング経路** (arm c, 1.44 t/s) であり、その底から
arm a/b の帯域まで持ち上げられるかが q* の真の価値問い。

よって q* 価値基準を以下の 2 軸で数値化する：

### 軸 A — 絶対帯域 (glru+q* で CPU 直計算帯に到達)
q* ON + glru 構成 (slot96 + global-lru, 本流のページング設定) の短文中央値 tg が
**≥ 11 t/s** に到達すること。
- 11 t/s は review doc §4 の「no regression」バンド下限 (= stage-a 実測の下限 12.28 の
  ±安全マージン)。q* がページング経路の同期 H2D 支配 (arm c の ~690 ms/tok) を
  解消し、stage-a 級の帯域を回復できれば達成。
- 理想は arm a/b の 12–14 t/s 帯への到達だが、q* は「ミスを CPU 直計算に逃がす」
  補完なので stage-a 超えは必須としない (stage-a は既に全ミスを CPU 計算済み)。

### 軸 B — 相対改善 (glru スラッシング底からの持ち上げ)
q* ON + glru 構成が、q* OFF 同構成 (arm c = 1.44 t/s) から **7× 以上 (≈10 t/s へ)** の
持ち上げを示すこと。
- 7× は「q* が実質的に機能している」と言える最低閾値。以下なら q* は実働していない
  か、レイアウト拒否等で発火していない (集計レシピで qstar_cpu=0 を確認)。

### 長文完走条件 (軸 C)
q* ON + glru 構成で 6575 tok 長文が **REQUEST-FAILED なく完走**すること。
- arm c は長文で実質停止 (prefill 1.2 tok/s) だった。q* がページング負荷を減らせば
  長文も完走するはず。完走しない = q* の負荷削減が不十分。

## 3. P2 測定プロトコル (q* 本流構成)

```
サーバ起動:
  -m <Huihui Q4_K> --host 127.0.0.1 --port 8096 --no-webui
  -ngl 999 -c 32768 -ctk q8_0 -ctv q8_0 -fa on -t 6 --cpu-moe
  --moe-gpu-expert-slot-num 96 --moe-gpu-expert-global-lru --moe-qstar
  環境: LLAMA_MOE_SLOT_STATS=1 + LLAMA_MOE_QSTAR_STATS=1
        (GGML_CUDA_DISABLE_GRAPHS は global-lru 検知で自動 OFF になる = F3 以降手間減)
測定: 受入テンプレ P1-P3 ×3 ラウンド (r1 破棄) + 長文 6575 tok
判定: coder-b-qstar-cpu-aggregation-2026-08-26.md で qstar_cpu>0 を確認
```

| 項目 | 合格条件 |
|---|---|
| qstar_cpu 発火 (軸前提) | qstar_cpu > 0 (bar ①) |
| 軸 A 絶対帯域 | 短文中央値 tg ≥ 11 t/s |
| 軸 B 相対改善 | arm c (1.44) から ≥ 7× (≈10 t/s) |
| 軸 C 長文完走 | 6575 tok REQUEST-FAILED なく完走 |

## 4. 丸め・注記

- arm a/b の 12–14 t/s 帯は「glru/q* 無し」の天井。q* がこれを「超える」必要はなく、
  「glru スラッシング底 (1.44) からこの帯へ持ち上げる」ことが価値証明。
- 軸 A/B は独立: 長文完走 (軸 C) が最も厳しい目盛りになる公算大
  (arm c の停止は prefill 時ページング負荷 → q* でどこまで減るかが本質)。
- arm c 長文補完測定 (2048/6575 切断の残り) は、crash 三師着 (大型 prefill + glru
  クラッシュ修正) 後に別ウィンドウで実施し、P2 材料に組み込む。

## 5. 健全性チェック (test_moe_frequency_placement.py) — スキップ

P1 exe 流用を試みたが、当該テストは `-ngl 999` (全層 GPU) を前提とし、
35B Q4_K (≈18GB 常駐) は 16GB VRAM に収まらずロードで 300s タイムアウト。
P1 は `--cpu-moe` で expert を CPU に逃がして収容していたため、このテストの前提と
モデルサイズが両立しない。コード退行ではなく環境制約のためスキップ。
frequency 配置機能自体は P1 の slot/graphs 経路とは独立しており、本 P2 の q* 価値
判定には影響しない。
