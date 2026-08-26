# qstar_cpu 集計レシピ (coder B, 2026-08-26)

bench-prep Task 3 成果物。受入バー① (`qstar_cpu>0` 観測) の自動判定用。
対象ログ: `LLAMA_MOE_QSTAR_STATS=1` (+ `LLAMA_MOE_SLOT_STATS=1`) 実行時の stderr
(`acc_err.log` / `glru_err.log` / `vb_err.log`)。

## ログ書式 (feat/qstar-r2-rebuild src/llama.cpp 実装より)

```
MoE GPU slot stats: copies=N hit=N miss=N evict=N residents=N cross_evict=N
                    qstar_xfer=N qstar_cpu=N copy=X.X MiB avg=X.XX ms h2d_gbps=X.XX
[QSTAR-CPUEXEC] entry: layer=L r=R n_embd=D qstar_tp=0x... threads=T
[QSTAR-CPUEXEC] before ggml_graph_compute: n_nodes=N work_size=S bytes=B
[QSTAR-CPUEXEC] after ggml_graph_compute: status=S
[QSTAR-PREP] entry / exit / exec_build returned false at layer=L
[QSTAR-CALIB] starting 3-rep host GEMV warmup / rep=R entering|done|cpu_exec returned false
```

重要: stats 行の `qstar_cpu=` は **起動からの累積カウンタ**。
発火回数そのものは `[QSTAR-CPUEXEC] entry` 行数 (=1 発火 1 行) で数える。

## 判定 ①: qstar_cpu>0 (PowerShell・コピペで動く)

```powershell
$errLog = 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna\acc_err.log'

# (a) 累積カウンタの最終値 (0 より大きければ受入① PASS)
$line = (Select-String -Path $errLog -Pattern 'MoE GPU slot stats' | Select-Object -Last 1).Line
$qcpu = if ($line -match 'qstar_cpu=(\d+)') { [int]$Matches[1] } else { -1 }
if ($qcpu -gt 0) { "BAR1 PASS: qstar_cpu=$qcpu" } else { "BAR1 FAIL: qstar_cpu=$qcpu (host exec 未発火)" }

# (b) 発火回数 = CPUEXEC entry 行数
(Select-String -Path $errLog -Pattern '\[QSTAR-CPUEXEC\] entry' | Measure-Object).Count
```

## Git Bash 版 (grep / awk)

```bash
errLog=/c/Users/dai86/llama-cpp-turboquant-experts-laguna/acc_err.log

# 累積カウンタ最終値 → 受入①判定
grep 'MoE GPU slot stats' "$errLog" | tail -1 | grep -o 'qstar_cpu=[0-9]*'

# 発火回数
grep -c '\[QSTAR-CPUEXEC\] entry' "$errLog"

# 層別分布 (layer ごとの発火回数、昇順)
grep '\[QSTAR-CPUEXEC\] entry' "$errLog" | grep -o 'layer=[0-9]*' | cut -d= -f2 \
  | sort -n | uniq -c

# awk 版 (同上)
grep '\[QSTAR-CPUEXEC\] entry' "$errLog" | awk -F'layer=' '{split($2,a," "); c[a[1]]++} END {for (l in c) print l, c[l]}' | sort -n

# 異常終了ステータスの抽出 (status != 0 のみ)
grep '\[QSTAR-CPUEXEC\] after' "$errLog" | grep -v 'status=0$' || echo '(all status=0)'

# CALIB (キャリブレーション) 死亡位置の特定 — 最後の 5 行がどこで止まったか
grep '\[QSTAR-CALIB\]' "$errLog" | tail -5

# PREP 失敗 (exec_build 拒否 = レイアウト問題の直接証拠)
grep 'exec_build returned false' "$errLog" || echo '(none)'
```

## 時間分布 (cumulative 差分 → 区間ごとの発火回数)

stats 行が複数回出る場合 (LLAMA_MOE_SLOT_STATS の定期出力)、隣接差分で
区間内発火回数が得られる:

```bash
# bash: 差分行 (負や 0 は異常 = カウンタ巻き戻し調査要)
grep -o 'qstar_cpu=[0-9]*' "$errLog" | cut -d= -f2 | awk 'NR>1{print $0-p} {p=$0}'
```

```powershell
Select-String -Path $errLog -Pattern 'MoE GPU slot stats' | ForEach-Object {
    if ($_.Line -match 'qstar_cpu=(\d+)') { [int]$Matches[1] }
}
# 出力を隣接差分手動確認 (PowerShell 5 には pairwise diff 演算がないため 3 行以内の目視で足りる)
```

## 解釈ガイド

| 観測 | 意味 |
|---|---|
| `qstar_cpu>0` + CPUEXEC entry 複数回 | host exec 分岐が実働 → 受入① PASS |
| `qstar_cpu=0` + `[Q*-PREP] exec_build returned false` | レイアウト拒否 (ne[2]/ne[1] 系)。GGUF 情報を A へ報告 |
| `qstar_cpu=0` + PREP/CALIB ログ自体なし | 計測ビルドでない (main バイナリに --moe-qstar を渡していないか確認) |
| CALIB rep 中に無言死亡 | 既知の calibrate crash 再現 → A の切り分け領域、B は記録のみ |
