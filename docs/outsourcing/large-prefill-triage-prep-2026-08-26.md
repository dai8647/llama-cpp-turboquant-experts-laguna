# 大型 prefill + glru クラッシュ — triage 準備記録 (coder A, 2026-08-26 夜)

F3 完了後、大型 prefill + glru クラッシュ (6575 tok, slot96, layer 18 無言死亡,
~13 秒) の調査に着手。本 doc は準備状況と初回試行の結果記録。
前提: review 提案 @5382a5b66 (gflags full page heap 判別) + repro 手順
(coder-b-large-prefill-repro-2026-08-26.md @8b7a17c64)。

## 準備完了したもの

1. **WER LocalDumps (HKCU, 管理者不要)**: llama-server.exe 専用キーを作成済み。
   DumpType=2 (full dump), DumpCount=3, 保存先 `dumps/`。
   クラッシュ時に full dump が落ちれば layer 18 死亡地点のスタック解析が可能。
2. **verify_a.ps1** (repo root, untracked): verify_b.ps1 の A コピー。
   差分 = (a) ExitCode 記録 (`0xC0000005` = AV 即死 = ヒープ破壊系 H1/H5、
   timeout まで生存 = H4 ハング系)、(b) dumps/ の dump ファイル有無記録、
   (c) 最後の slot stats 行抽出。B のファイルは触っていない。
3. **setup-pageheap.ps1** (repo root, untracked): gflags 無し環境向けの
   IFEO 直設定スクリプト (GlobalFlag=0x02000000 + PageHeapFlags=3 = full page heap
   相当)。**管理者権限が必要で UAC 承認待ち** (1 回試行は拒否された)。
   gflags.exe 自体は Windows Kits Debuggers 未インストールのため不在。
   承認されれば実行 1 回で有効化、無効化は IFEO キーの値削除。

## 初回試行 → 中止 (GPU 調整)

- `verify_a.ps1 -Tag va_glru96_repro --moe-gpu-expert-slot-num 96
  --moe-gpu-expert-global-lru` を開始した直後、**B の P1 stage-a bench
  (p1_stage_a_bench.ps1, parent 5340) が同時開始されたことを検知**
  (llama-server 2 プロセス、36 秒差)。両者の数字保護のため A 側を即時中止・
  プロセス停止して GPU を譲渡した。
- 注意: 中止操作後、B 側の llama-server (PID 2128) も消えている。
  A の停止対象は PID 4000 (A runner) + PID 25284 (A server) の 2 つのみで
  2128 には触れていない。B 側で LOAD-FAILED (port 8101 衝突の可能性 —
  両者とも既定 port 8101) または正常なフェーズ遷移のどちらか。
  **P1 の再実行要否は B 側へ確認推奨。**
- 教訓: verify_b / verify_a / p1_stage_a_bench は全部既定 port 8101。
  同時実行防止として、次回から A 側は `-Port 8112` 等の別ポートを使う。

## 再開手順 (GPU 枠取得後)

1. page heap を使う場合: setup-pageheap.ps1 を UAC 承認で実行 → IFEO 反映は
   次回起動の llama-server.exe から効く (メモリ増大注意、ロード失敗したら
   PageHeapFlags=1 の light heap に落とす)
2. `.\verify_a.ps1 -Tag va_glru96_repro -Port 8112
   -ExtraArgs '--moe-gpu-expert-slot-num 96 --moe-gpu-expert-global-lru'`
3. 判定表:
   - PROCESS-DIED exit_code=0xC0000005 + dump 有 → H1/H5 (ヒープ破壊系確定)
     → H1 計測 printf (review 提案: 保護の無い各書き込みサイトへの inflight
     存在チェック fprintf) に進む
   - PROCESS-DIED exit_code=0xC0000005 + dump 無し → AV は出るが WER 未発火
     (LocalDumps の HKCU 効力検証が必要)
   - REQUEST-FAILED 後 alive=True (timeout まで生存) → H4 (デッドロック系)
     → event 操作のロック外しパッチ (H4 切り分け, review 提案) に進む
4. 取得物: va_err.log 末尾 ~50 行、最後の slot stats 行、dump ファイル

## 関連

- 仮説 5 件: large-prefill-crash-hypothesis-2026-08-26.md (7959cb85c)
- review 検証: review-large-prefill-hypotheses-2026-08-26.md (@5382a5b66)
  — H2 無罪 (制御フロー上 erase 後 ++it 不評価)、H4 の mutex 保持中
  event_synchronize は実在 (llama.cpp:696 lock_guard 下 line 701)、
  H1 は ensure_resident 保護済みで「保護を通らない別書き込み経路」が問い
