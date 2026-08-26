# レビュー: 大型 prefill クラッシュ仮説カタログへの注記 (review side, 2026-08-26)

対象: `docs/outsourcing/large-prefill-crash-hypothesis-2026-08-26.md` (7959cb85c, feat/qstar-r2-rebuild)。 review 側が r2 ブランチの実コードを参照して確認した結果の注記。 優先順は変更しない (q* trace が本日の主目標、 本件は同じバイナリで後追い可能)。

## 仮説 2 について: 実害なしと判断 (降格推奨)

r2 ブランチ `src/llama.cpp:697-714` を実読した。 カタログの記述「`it = inflight.erase(it)`」だが、 実コードは line 704 で **戻り値を代入しない素の `erase(it)`**。 ただしその直後は必ず `matches` なら `return true` (line 710)、 非 match なら `break` (line 712) で **どちらの経路でもループを抜けるため、 erase 後に `++it` は評価されない**。 C++17 でも UB 不成立。 この関数に限っては無罪。

## 仮説 1 の絞り込み: 保護は既に存在する。 残る問いは「他の書き込み経路」

line 691-694 のコメントが示す通り、 ensure_resident 側には inflight 中 slot の保護が **既に実装済み** (同一 layer/slot の inflight を sync + destroy してから触る)。 つまり H1 が生きているとすれば穴は「**この保護を通らずに slot bank 領域へ書き込む別経路**」(eviction / decode 側 prefetch / materialize 側の完了処理など)。

直接証拠を出す計測の提案: 上記保護を持たない各書き込みサイトに、「その時点で同 (layer,slot) の inflight エントリが存在するか」を 1 行 fprintf するチェックを入れる。 クラッシュ前に 1 回でもヒットログが出れば H1 確定。

## 仮説 4 の補強: 長時間 mutex 保持は実在する

line 696 の `lock_guard` 下で line 701 `ggml_backend_cuda_ext_event_synchronize` を呼んでおり、 「cache_mutex 保持中の GPU 同期ブロック」は仮説ではなく実コードに存在する。 切り分け実験としては、 一時的に event 操作をロック外に出すパッチを当てて再現有無を見るのが最短。 (recursive_mutex は同一スレッドの再取得しか助けないので、 転送完了コールバック側が cache_mutex を取る設計ならデッドロック成立余地あり)

## 再ビルド不要の切り分け (追加提案)

Windows の「無言クラッシュ ~13 秒」に対して:

```
gflags /p /enable llama-server.exe /full   # full page heap: ヒープ破壊を即時 AV 化
```

(gflags.exe は Windows SDK 同梱。 exe がある build-hip\bin で実行、 無効化は `/disable`)。 加えて WER LocalDumps レジストリで .dmp 取得。 **ヒープ破壊系 (H1/H5) なら即死でダンプが出る。 H4 (デッドロック) ならプロセスが生き続ける** — これだけで仮説ファミリーの判別が付く。 Release バイナリのまま効くので再ビルド不要。

## 小さい訂正

RelWithDebInfo に MSVC 経路は不要。 今回リンクまで通った HIP clang++ + llvm-ar 構成のまま `-DCMAKE_BUILD_TYPE=RelWithDebInfo` に差し替えるだけで良い (math_fwd.h 罠は cl.exe 専用)。

## 推奨順序

1. q* trace (本日の主目標、 受入バー判定)
2. gflags 有効化 + 大型 prefill 再現 (再ビルドゼロでファミリー判別)
3. 判別結果に応じて H1 計測 printf か H4 ロック外しパッチ
