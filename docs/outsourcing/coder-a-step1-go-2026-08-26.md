# コーダーA へ — 受領、step 1 着手許可 (2026-08-26)

86f8c2308 受領。ブランチ origin push + ログ整備 + メモリ更新すべて確認した。
**step 1 (Debug build) の着手を許可する。**

## 1 点だけ先に報告: build-issue doc は main 側を A 版で統一済み

レビュー側が先に作っていた同名 `docs/build-issues/rocm-71-msvc-attribute-pure.md`
(453f423b4) を、A の書き直し版で **verbatim 差し替え**した (main 87e2063b1)。
理由: 将来 round-2 を main にマージする時、このパスで競合しないようにするため。
A 版の方が優れている点 (増分ビルドでは出ない / 回避策の優先度順 / 本機 config)
を評価した。旧版の内容 (MSVC が `__attribute__` を空マクロ扱いする原因分析等)
は git 履歴 453f423b4 に残っているので失われていない。

**影響**: A 側での対応は不要。round-2 マージ時にこのファイルは自動的に解決される。

## step 1 への追加情報

- main は A の分岐点 (76adf21e4) から 87e2063b1 まで進んでいるが、
  差分は docs のみ (coder-a/b 返信 2 件 + 上記 doc 統一)。コード差分ゼロなので
  debug ビルドの前提は変わらない。docs が欲しければ適宜 main を取り込めば良い。
- ASan が HIP clang で使えない場合は plan 通り `-O0 -g` + printf フォールバックで。
  その際は exec_prepare 直後と qstar_cpu_exec 入口/出口の 3 点を先に埋めると
  「calibrate 3 ループ目で死亡」の前後関係が最速で取れる (plan 記載の通り)。

進捗は引き続き `qstar-round2-log-2026-08-26.md` へ。数字なき完了報告は受け付けない
ルールはレビュー側も遵守する (クラッシュサイト特定報告にはスタックトレース or
printf ログ断片を添えてもらえると即検証できる)。

— レビュー側 (ZCode) / ユーザ経由
