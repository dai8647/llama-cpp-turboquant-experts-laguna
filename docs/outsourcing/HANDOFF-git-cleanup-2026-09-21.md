# ハンドオフ — git リポジトリ整理 (2026-09-21)

**あなたの仕事は git 履歴とブランチの整理だけです。コードの機能変更・ビルド・GPU 実行は一切しないでください。**

リポジトリ: `C:\Users\dai86\llama-cpp-turboquant-experts-laguna`
リモート: `origin` = https://github.com/dai8647/llama-cpp-turboquant-experts-laguna.git
(`upstream` = ggml-org/llama.cpp。upstream には触らない)

---

## ⚠️ 最重要ルール (事故防止)

1. **`git push --force` / `--force-with-lease` は一切禁止**。force push が必要に見える場合は必ず停止して相談。
2. **未コミットの作業中ファイルを絶対に消さない・退避させない**。特に以下は別AI/セッションの作業中:
   - `src/models/qwen4exp.cpp` (MTP 作業中、未コミット) — **触るな**
   - `src/llama-model.cpp`, `src/llama.cpp`, `src/llama-model.h` (Session A 作業中) — **触るな**
   - `src/models/bailingmoe3.cpp.disabled`, `src/models/kimi-k3.cpp.disabled` (意図的な無効化) — **消すな**
3. **生成物・ログは消してよいが、コミットには絶対入れない**。
4. 削除・rebase・branch 削除など破壊的操作の前に、必ず `git stash list` / `git status` / 対象 branch の差分を確認し、**実行計画を提示して承認を得てから**実行。

---

## 現状の問題 (整理対象)

### A. 作業中の未コミット変更とゴミの混在
`git status` に以下が混ざっている:
- **作業中 (保持)**: Session A の llama.cpp/llama-model.* 変更、run-*.cmd、AGENTS.md、qwen4exp.cpp
- **ゴミ (削除してよい)**: ルートの `_check_gguf.py`, `_diff_*.py/txt`, `_gguf_*.txt`, `_prep_*.txt`, `_tk_*.txt`, `_trunk_keys.py`, `_vd_*.txt`, `_verify_draft.py`, `_draft_keys.py`, `diag_*.cmd`, `diag_pid.txt`, `bisect-lazyoff.cmd`, `bench_sessionA_cpu_*.csv`, `mtp_test*.log`, `split_*.log`, `bench_qwen38_20tps_logs/`, `bench_qwen38_20tps_results_*.json` 等
- **残すべき成果物だが未追跡**: `bench_qwen38_20tps.ps1`, `docs/qwen38-20tps-bench.md`, `docs/qwen4exp-capacity-evaluation-2026-09-19.md`, `docs/outsourcing/*.md` (ハンドオフ群), `src/models/*.disabled`

### B. MTP ブランチが2つある (重複)
- `feat/qwen4exp-hot-expert` (カレント): `a45a839f8` (shape fix) + `3820c7e38` (port) — **これが正**
- `feat/qwen4exp-mtp`: `167285627` (WIP port、古い別系統)
→ どちらを残すか、古い方をどうするか (削除 or アーカイブタグ) を判断

### C. リモートとの乖離
- `origin/feat/qwen4exp-hot-expert` は `3820c7e38` 止まりで、`a45a839f8` (shape fix) が **未 push**
- カレントブランチに upstream (追跡先) が設定されていない

### D. 古い・不要なブランチ
`merge-upstream`, `merge-upstream-local`, `merge-upstream-next`, `feat/b-only`, `feat/prefill-double-buffer`, `feat/qstar-*` 等、過去の作業ブランチが origin に多数残っている。どれを消してよいか。

### E. upstream ブランチのノイズ
`git fetch` で ggml-org/llama.cpp の数百ブランチがローカル ref に残っている。`git branch -r` の出力が読めないほど多い。

---

## やること (順番に、承認を挟みながら)

### Step 1: 現状の棚卸し (読み取りのみ)
- `git status`、`git log --oneline --graph --all -20`、`git branch -vv`、`git stash list` を取得
- 各未コミット/未追跡ファイルを「作業中(保持) / ゴミ(削除) / 成果物(addしてコミット)」に分類した表を作る
- 2つの MTP ブランチの差分 (`git log feat/qwen4exp-mtp..feat/qwen4exp-hot-expert` 等) を調べ、どちらが最新か確定

### Step 2: 整理計画の提示 (ここで一度停止して承認を得る)
- 削除するゴミファイルのリスト
- コミットする成果物 (bench スクリプト、docs、.disabled) とそのコミット粒度
- MTP ブランチの統合方針 (hot-expert に集約、古い feat/qwen4exp-mtp はタグ保存して削除、等)
- 削除する古いブランチのリスト
- ローカル → origin への push 計画 (force なし)
- **この計画をユーザーに提示し、OK が出るまで実行しない**

### Step 3: 実行 (承認後)
- ゴミ削除 (作業中ファイルを絶対に含めないこと)
- 成果物を意味のある単位でコミット (例: `chore(bench): add canonical qwen38 20tps bench`, `docs: add session handoffs`, `chore: disable unused bailingmoe3/kimi-k3 model files`)
- MTP ブランチ統合
- 不要ブランチ削除 (ローカル + origin)
- upstream ブランチ ref の掃除 (必要なら)
- push (force なし)

### Step 4: 最終確認
- `git status` がクリーン (作業中ファイルのみ残る)
- `git log --oneline --graph` が読みやすい
- `git branch -a` が整理されている
- 結果の before/after を報告

---

## やらないこと
- コードの機能変更、ビルド、GPU/ベンチ実行
- force push
- 作業中ファイル (§最重要ルール 2) への干渉
- upstream (ggml-org) への操作
- HOST_BANK=1 など実行系フラグの変更

## 報告
各 Step で「何を見つけたか / 何をするか (計画) / 何をしたか / 残課題」を簡潔に。破壊的操作は全て実行前に承認を得ること。
