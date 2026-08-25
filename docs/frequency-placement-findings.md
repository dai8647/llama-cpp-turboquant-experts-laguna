# 頻度配置(Pass 1/2)実運用テスト結果 — 2026-08-25

Status: 実測完了(一部)。コーダーA(q*+global LRU)・コーダーB(prefill double buffer)
への引き継ぎ資料。計測はすべて main ビルド(09:45 build, commit 62f8d92c7 相当)、
RX 7800 XT 16GB + 96GB RAM、bench_ft.ps1(6575トークン・コードレビュー風プロンプト +
256トークン生成、32K ctx, KV q8_0, --cpu-moe, -fa on, -t 6,
GGML_CUDA_DISABLE_GRAPHS=1)。

モデル: Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated Q4_K (20.2 GB,
qwen35moe, 40 MoE層, 256 experts, top-8)
頻度レポート: `ft_freq.json`(2026-08-25 10:38生成、約6.9kトークンサンプリング)。
top-30/層で全選択の50.9%(層別33-61%)、top-96で85.1%をカバー。
247/256 エキスパートが最低1回選択される = ルーティングは拡散的。

## 結果

| Run | 構成 | 結果 |
|---|---|---|
| B0 | `--moe-gpu-expert-slot-num 30`(whitelistなし) | pp 157.6 / tg 12.00、**マテリアライズ0回**(統計不出力) |
| F1 | B0 + `--moe-expert-placement frequency --moe-freq-report-in ft_freq.json`(ratio 1.0 = 既定) | **600sタイムアウト**。hit 228k/miss 219k(51.1%)、miss=evict、コピー417 GB |
| F1b | 同上 ratio 0.117(≒top-30/層、whitelist 1198件) | **600sタイムアウト**。hit 262k/miss 253k(50.9%)、miss=evict、コピー464 GB |
| F2 | ratio 0.375 + slot96 | 未実施(GPU共有中につき延期。A/B実装後に再測推奨) |

## 発見1: whitelistなしのスロットキャッシュはゲートで無効化されている

`ensure_resident`(src/llama-model.h、main版)の冒頭:

```cpp
// collection mode (Pass 1 of the frequency workflow, slots < experts,
// no whitelist): count access only, no slot assignment/materialization
if (frequency_whitelist.empty() && n_slots < n_experts) {
    return expert_id;
}
```

この条件は `track_access` を見ていない。つまり **Pass 1 収集モードでなくても、
whitelist が無く slots < experts な起動はすべてキャッシュが silently 無効**になる。
B0 でマテリアライズがゼロだったのはこのため(B0 は実質スロットレス動作)。

**コーダーAへ**: 作業ツリーの q*+global LRU 実装には
`const bool global_lru = global_lru_enabled && frequency_whitelist.empty();`
があるが、上記ゲートがその手前で return するため、
**global LRU パスは現状到達不能(デッドコード)**。global LRU を
whitelist なしの通常キャッシュとして機能させるなら、ゲート条件を
`track_access && whitelist.empty() && slots < experts` に絞るか、
collection mode を明示フラグ化すること。導入コミット: 8c0c3bb4f。

## 発見2: whitelist モードは適応せず、ヒット率が静的カバレッジに張り付く

F1(ratio 1.0)と F1b(ratio 0.117)で、ヒット率がどちらも **50.9-51.1% =
top-30 静的カバレッジそのもの**に張り付き、miss == evict(常に満杯、
全ミスが誰かを追い出す)になった。LRU の再近接性で適応する気配がない。

- 可能性a: このワークロードのルーティングには頻度分布を超えた時間的局所性が
  元々無く、LRU に搾取する余地がない(なら q* の CPU 実行パスが本命になる)。
- 可能性b: whitelist モードの何か(プリロード由来の last_used 初期値、
  プリロードの反復経路等)が適応を阻害している。

判別方法: `LLAMA_MOE_SLOT_STATS=1` のヒット率をステップ経時で見る。
収束せず一定なら a、動くなら b。A の q* 設計の前提(「デコードのルーティングは
時間的局所性が強い」)に関わるので優先確認を推奨。

## 発見3: ratio 1.0 + 少スロットは最悪構成になる

`--moe-gpu-expert-ratio` 既定値 1.0 は whitelist = 全エキスパート(10240件)。
プリロードは whitelist 順(頻度降順)にスロットを埋め回るので、
**最終的にスロットに残るのは最も冷たいエキスパート**になり、
以降全選択がミス+同期コピー(0.76-0.81 ms/回)になる。

対策候補(実装はAの所有範囲):
- 起動時に `whitelist.size() > 総スロット数` なら警告 or ratio 自動クランプ
- プリロードを「スロットが埋まったら打ち切り」に変更(現状は全件なぞる)

## 発見4: whitelist と実ワークロードの不一致は致命的

ft_freq.json は日本語QA生成タスク(約6.9kトークン)で収集されたが、
bench_ft.ps1 のワークロード(6575トークンのC++コードレビュー風プロンプト)
ではルーティング分布が異なり、ヒット率51%・プリフィル中に464 GBのコピーchurn
でリクエストがタイムアウトした。一方、収集直後の同種ワークロードで測った
今朝のセッション計測では tg +11% の効果が出ている(4670トークンプロンプト、
slot30: pp 127.8 / tg 11.84。スロットレス: pp 135.4 / tg 10.67)。

つまり頻度配置は **レポートがワークロードを代表している時は効き、
外れるとスロットレスより大幅に悪化する**。運用ルール案:
- レポートは本番と同じ種別のプロンプトで収集する(生成のみでなくプレフィル込み)
- レポート収集時のプロンプト/トークン数をレポートのメタデータに残す
  (現状 `n_active_experts: 0` も未設定)

## 既知の課題(再掲)

- 長文プリフィルは HIP グラフキャプチャで argsort クラッシュ
  (docs/moe-slot-cache-async-design.md §8.3)。コーダーB担当。
- `generate_access_report` は全選択を `gen_selections` に詰め
  `prompt_selections` は常に 0(報告様式の未実装)。prompt/gen の分離収集は未着手。
