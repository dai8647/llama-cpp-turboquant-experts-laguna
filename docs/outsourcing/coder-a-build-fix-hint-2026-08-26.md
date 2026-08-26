# A へ: ビルド確定バグ予兆 — FA_ALL_QUANTS 欠落 + ログ取得レシピ (review side, 2026-08-26)

STATUS: 情報提供のみ。 single-owner rule (coder-a-build-coordination-2026-08-26.md @d2016c9dc) の通り review 側は build-hip に触らない。 急ぎの話は 1 つだけ。

## 1. このまま完走しても ggml-hip.dll のリンクで落ちる線がほぼ確定している

`run-cmake-hipclang.ps1` line 13 の cmake オプションに **`-DGGML_CUDA_FA_ALL_QUANTS=ON` がない**。 canonical `build-hip.ps1` はこのフラグを常時付けていた。

根拠は 2 つ:

- review 側の同構成 (HIP clang++ + llvm-ar、 FA_ALL_QUANTS 無し) ビルドは compile 全通過後のリンクで死亡: `undefined ggml_cuda_flash_attn_ext_vec_case<DV,cols_x,cols_y>`
- **リポジトリルートの過去ログに同一死亡の記録がある**: `build-hip-full.log` (2026-08-13) 行 722318-722344 に `lld-link: error: undefined symbol: ...flash_attn_ext_vec_case<64|128|256, 8|30, 1|8|30>` 計 9 シンボル。 8 月 13 日にも全く同じ場所で死んでいる。

つまり -j 1 で時間をかけても、 最後のリンクで同じ死に方をする可能性が高い。 対処は configure へのフラグ 1 個追加:

```
-DGGML_CUDA_FA_ALL_QUANTS=ON
```

(スクリプトは毎回 wipe のためフル再 configure になるが、 ccache=ON なので obj の大半はヒットする見込み)

## 2. 「エラーがログに残らない」のは出力経路のせい。 素リダイレクトで取れる

[149/417] subcommand failed の実エラーがどこにも無い件を review 側が検証: 最近の exec ログ全点で FAILED 行 **0 件**・最大 2780 バイト。 stderr がパイプ途中で消えている (PowerShell tee / バックグラウンド exec の組合せ。 review 側も同じ罠で「exit 0 の偽物」「エラー消失」を両方踏んだ)。

確実なレシピ:

```
ninja > ..\r2-build.log 2>&1
grep -nE "FAILED:|error:" ..\r2-build.log
```

ninja は FAILED: の直後に失敗コマンドと stderr 全文を出すので素リダイレクトだけで足りる。 `echo EXIT=$?` の追記は exit code をマスクするので禁止。

## 3. -j について

今走っている分を止めろという話ではない。 -j 1 は OOM 等 flaky 要因の切り分けに有効だが、 上記 1 が解決しない限り着地点はリンク死のままで時間だけ溶ける。 止めるなら FA_ALL_QUANTS 追加とセットで。 エラー本文が欲しいだけなら -j 4 + 素リダイレクトで deterministic かどうかも同時に判る。

## 4. 参考

- ログに大量に出る `[-Wignored-attributes] __declspec(dllimport)` warning は無害 (clang + GGML_API の定番ノイズ)。 失敗原因ではない。
- 1b3569245 (printf 5 箇所) 受領済み。 バイナリが出たら run-qstar-trace.ps1 → クラッシュサイト特定へそのまま進めて良い。
- review 側のビルド試行は前回で終了、 以後 build-hip 不触 (継続)。
