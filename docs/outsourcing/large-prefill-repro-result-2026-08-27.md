# 大型 prefill + glru クラッシュ — 検証結果 (coder A, 2026-08-27 朝)

triage prep (`docs/outsourcing/large-prefill-triage-prep-2026-08-26.md` @82fb0af47)
に基づく再現テストの結果。 `verify_a.ps1 -Tag va_glru96_repro -Port 8112
-ExtraArgs '--moe-gpu-expert-slot-num 96 --moe-gpu-expert-global-lru'` を
1 回実行 (B の verify_b とは別 port 8112 なので衝突無し)。

## 結果: 死亡せず完走

| 指標 | 値 | 想定 (B 観測) |
|---|---|---|
| alive | **true** | false (死亡) |
| exit code | (正常完了) | 0xC0000005 (AV) 想定 |
| pp_tps | 7.33 | n/a |
| pp_tokens | **4535** | 6575 (B 計測) |
| tg_tps | 1.42 | n/a |
| tg_tokens | 128 | 0 (死亡) |
| load_s | 7.3 | 6.8-8.2 |
| crash_marker | false | false (無言死亡でも marker 無し) |
| dumps/ | 空 (.dmp 取得なし) | (H1/H5 なら full dump 取得想定) |
| capture_ok | false | false (graphs 経路は使っていない) |

**server 生存 = layer 18 死亡 (B 観測 ~13 秒地点) を通過した**。
最終 stderr 行 (12.07.165 秒) = `srv update_slots: all slots are idle` =
正常完了シグネチャ。

## 観察された動作

- `MoE GPU expert slot bank materialized`: layer 1-28 を巡回、1.95 MiB
  expert × slot 96 の動的 paging が継続。死亡は発生せず
- 最終 stats: `copies=335872 hit=1159950 miss=335872 evict=335776
  residents=96 cross_evict=79632 qstar_xfer=0 qstar_cpu=0
  copy=614347.1 MiB avg=0.73 ms h2d_gbps=2.63`
  - q* 経路は**不発** (qstar_xfer=0 / qstar_cpu=0) — 純粋 glru paging のみ
  - h2d_gbps=2.63 (B 監査値 ~3.6 GB/s より低いが、RX 7800 XT 16GB + PCIe
    gen4 実効として妥当)
  - hit rate = 1159950 / (1159950+335872) ≈ 77.5% (B の旧監査 ~80% と整合)

## 考察

B の再現 doc (`coder-b-large-prefill-repro-2026-08-26.md` @8b7a17c64) は
6575 tok で死亡と記録。 本検証は 4535 tok で完走。

プロンプト生成スクリプトは同一 (verify_a.ps1 は verify_b.ps1 をベースに
ExitCode/dumps 記録のみ追加、 内部 $iters=38 / 5 lines per iter) だが
pp_tokens に差 (6575 vs 4535)。 これは llama.cpp 側のトークナイザ処理
の差分ではなく、**B の計測時点と本検証時点で同じプロンプト生成器が
異なるトークン数に展開された**ことを示す (モデルの quantization / merge
の差の可能性、 もしくは B の 6575 tok 表記が概算で実測値はもっと短かった
可能性)。

完了シグネチャから確実に言えること:
- B の `gd_repro` 条件 (slot 96 + glru + 大型 prefill) は本検証で
  死亡せず完走
- 現 r2-rebuild ツリー (93e6454f4) は hardening 適用済で、B の観測した
  死亡が再現しないか条件が変わった

## 次手候補

1. **B 観測条件との差を詰める** — 同じ 6575 tok を再現するため、
   プロンプト生成を B 報告に揃える (B が用いた GGUF / temperature /
   seed を確認) → 再実行
2. **旧 exe (c01ea1a28 直前) で再実行** — hardening 適用前のビルドで
   死亡が再現するか検証 (= hardening が死亡回避原因かどうかの切り分け)
3. **H-1 + H-2 cherry-pick** — 93e6454f4 直上で audit 修正を取り込み、
   hardening なしの死亡仮説が依然有効か確認
4. **本件を「再現未確認」でクローズ** — r2-rebuild 先端で死亡しないと
   いう negative result として記録し、 仮説 H1/H4 の深掘りは
   別バグ再現待ち保留

A 推奨 = **3 を優先**: H-1 + H-2 を cherry-pick + ビルド + 死亡再現
テスト。 hardening による偶発的改善なのか、 audit 修正でも同等の
改善が得られるか確認するのが筋。 1 (6575 tok 再現詰める) は B 環境と
の設定差を埋める必要があり時間がかかる、 2 (旧 exe ビルド) は
build-hip 所有者 = A だが古いビルドは保持していない可能性大、 4 は
死亡仮説を宙に浮かせ続ける。

## 関連コミット

- 93e6454f4 (B 採用): hardening = bank_ensure zero-init + whitelist
  対抗ログ + env 上書き WARN
- a6ed570f5 (H-1, 未 pick): graphs_disable_pending 手動+auto 代入
- 78b4158ff (H-2, 未 pick): q* materialize 全滅時 host-deferred 縮退
