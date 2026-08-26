# 大型 prefill + glru クラッシュ 仮説整理 (2026-08-26)

## 観察事実 (B からの引継ぎ)
- 症状: glru + 6575 tok prefill + slot96 → layer 18 付近で無言クラッシュ
- `GGML_CUDA_DISABLE_GRAPHS=1` でも再現 → グラフ機構は無関係
- warmup decode + paging は成功 → prefill 経路 paging 限定
- タイミング ~13 秒 = prefill ubatch 内のページ競合

## 仮説 1: prefill_pf_inflight と materialize の slot 競合
- prefill_pf_prefetch (llama.cpp:940-) が inflight.copy を poll し、完了時に
  `slot->resident = true` (line 962) を立てる
- 別経路の ensure_resident (llama-model.h:1295-) が同 slot に **別の expert_id**
  を assign_slot する可能性 → 同じ slot バッキング領域に別 expert データが
  部分的書き込み (H2D 中の上書き) → ヒープ破壊 → 後の layer で死亡
- 6575 tok の大量ページングで衝突頻度が高まり、layer 18 までに蓄積

## 仮説 2: materialize 内 prefill_pf_inflight イテレート
- llama.cpp:697-714: `for (auto it = inflight.begin(); it != end; ++it)`
  で `it = inflight.erase(it)` (line 704) + `break` (line 712)
- 0 件 hit → ++it されず条件 false で脱出 → 実際は安全
- 1 件 hit → erase + break → 脱出 → 1 件のみなら安全
- ただし **std::vector** なので erase(it) で it 無効化、その後 ++it 評価は UB
  (C++17 までは、 C++20 以降は erase 戻り値+break で安全)
- 今回のビルドは C++17 (ggml 設定) → UB 経路

## 仮説 3: prefill_pf_prefetch 内 953-967 の問題
- 948 `for (auto it = inflight.begin(); it != end; )` で `++it; continue` or
  `it = inflight.erase(it)` を正しく使用 → これは安全

## 仮説 4: lock 順序違反
- cache_mutex は recursive_mutex で全経路統一
- prefill_pf_inflight への同時アクセスは cache_mutex 配下
- 一見 lock 順序 OK だが、 materialize 内の `it->event` 操作
  (line 701 `event_synchronize` / line 702 `event_destroy`) は
  キャッシュ mutex 保持中 → backend API の長時間ブロック → 別 thread 待機
- HIP stream 同期が driver 内 mutex を保持し、 driver 側でデッドロックの可能性
  (hipStreamSynchronize + event destroy 同時呼び出し)

## 仮説 5: メモリ pool 枯渇
- prefill_pf_inflight の max_inflight=64 を超えてバックプレッシャー
- それが引かずに queue 伸び続けると VRAM 圧迫 → driver 内部で OOM ハンドラ発火
- layer 18 で何かの CUDA call が NULL pointer に触れて無言死亡

## 次の切り分け
1. ビルド完了後、 qstar を **無効化** して大型 prefill + glru のみ実行
   - qstar を切り離して paging 単独のクラッシュか確認
2. `LLAMA_MOE_PREFILL_PF=0` で同じ prefill 実行
   - prefill_pf を切り離して plain paging のみでクラッシュか確認
3. `LLAMA_MOE_SLOT_STATS=1` で layer 17 までの stats を取得
   - 最後の正常 stats と n_resident_global の推移を記録
4. クラッシュダンプ取得のため `-DCMAKE_BUILD_TYPE=RelWithDebInfo` を試行
   - ただし本機の ROCm 7.1 + MSVC 経路が必要 → 時間があれば

## 修正案 (暫定)
- 仮説 1 が最有力: ensure_resident が prefill_pf_inflight と同 slot を扱う際、
  inflight entry の expert_id と要求 expert_id の **不一致チェック** が無い
- 修正: ensure_resident 内で `prefill_pf_inflight` をチェックし、
  inflight の slot_id と一致する場合は **inflight 完了を待つか、別 slot を割当**
- これは「q* 機能側」ではなく「paging 機構側」の修正 → A 領域として A 担当 OK
- 単独 reimplementation に該当しない (§8/§9 レビュー不要)

## 関連コミット
- 2d442cdb0 moe : prefill double buffering (prefill_pf_* API)
- 7a212bd76 moe : elastic VRAM sizing
- 3b49ca50a → 2d442cdb0 系 (prefill_pf_configure / prefill_pf_prefetch)
