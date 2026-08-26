# q* r2 受入バー実行テンプレート (coder B, 2026-08-26)

bench-prep Task 2 成果物。**GO トリガー前は実行しない**
(A が `qstar_cpu>0` + 受入パスを報告 → review 側が B へ実行指示 → 本テンプレに従う)。

## 受入バー (固定・5 項目)

| # | 判定項目 | 合格条件 |
|---|---|---|
| ① | qstar_cpu 発火 | `qstar_cpu>0` を観測 (集計は coder-b-qstar-cpu-aggregation-2026-08-26.md) |
| ② | 短文 3 プロンプト ×3 ラウンド | 各プロンプトの **r1 を破棄**し r2/r3 を採用 |
| ③ | 長文 6575 tok | REQUEST-FAILED でなく完走 |
| ④ | Ornith 同条件 | Huihui Q4_K と同一条件 (モデルパス以外同一引数) |
| ⑤ | env 記録 | 下記 env 欄を実行開始前に全て埋める |

## env 欄 (実行開始前に記入)

| 項目 | 値 |
|---|---|
| ブランチ / ビルドコミット | feat/qstar-r2-rebuild @ |
| llama-server.exe パス | (-BinaryPath で渡したもの) |
| warmup | 有 ('Say hello.' n_predict=16 + 各プロンプト r1 破棄) |
| graphs 状態 | OFF (`GGML_CUDA_DISABLE_GRAPHS=1`) |
| LLAMA_MOE_QSTAR_STATS | 1 |
| ctx / slots / threads / port | 32768 / 96 / 6 / 8096 |

## 短文 3 プロンプト — 文面確定 (変更禁止・互いに別トピック・ほぼ同長)

共通パラメータ: `temperature=0.3`, `n_predict=128`

### P1 コードレビュー (~110 tok)

```text
Review the following C++ function and point out any issues concisely.

std::vector<int> top_k(const float* scores, size_t n, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) { return scores[a] > scores[b]; });
    idx.resize(k);
    return idx;
}

List at most five findings.
```

### P2 金利カーブ分析 (~95 tok)

```text
You are a fixed-income analyst. Explain how a steepening of the 2s10s Treasury curve affects: (1) a leveraged duration-neutral curve trade, (2) negative convexity hedging costs for MBS, and (3) corporate bond relative value screening. Write three short paragraphs, each under four sentences.
```

### P3 日英技術翻訳 (~120 tok)

```text
Translate the following Japanese technical passage into natural English, preserving terminology. Provide only the translation.

「expert の一部のみを VRAM に常駐させ、残りは需要発生時にホストから転送する方式では、転送帯域と計算の overlap が性能を左右する。特に prefill 中の大量ページングはレイテンシ急増を招くため、LRU による常駐選択の精度が重要になる。」
```

## 結果表

### Huihui 短文 ×3 プロンプト ×3 ラウンド (受入②+①)

| プロンプト | r1 tg t/s (破棄) | r2 tg t/s | r3 tg t/s | r2/r3 avg | qstar_cpu 回数 | 旧監査値 |
|---|---|---|---|---|---|---|
| P1 | | | | | | **13.78** |
| P2 | | | | | | **16.81** |
| P3 | | | | | | **16.76** |

### Huihui 長文 6575 tok (受入③)

| 項目 | 値 |
|---|---|
| 完走 / REQUEST-FAILED | |
| pp t/s (tok) | |
| tg t/s (tok) | |
| qstar_cpu 回数 | |
| 旧監査値 | **13.43** |

### Ornith 短文 P1 使用・同条件 (受入④)

| ラウンド | tg t/s | qstar_cpu 回数 |
|---|---|---|
| r1 (破棄) | | |
| r2 | | |
| r3 | | |
| **r2/r3 avg** | | (旧監査値 **13.07**) |

旧監査値の出典: A 受入ラウンド (coder-a-confirm-2026-08-25.md)。**warmup バイアス込みで
過大評価と撤回済みの既知数値**であり、比較の参考列。合格判定には使わない。

**合格ライン** (coder-b-followups-2026-08-25.md 由来):
Huihui 短文 r2/r3 avg ≥12.38 t/s / 長文完走かつ tg ≥11.84 t/s / Ornith r2/r3 avg ≥12.4 t/s

## Ornith 用起動引数案 (モデルパス以外は Huihui Q4_K と完全同一)

```
-m C:\Users\dai86\.lmstudio\models\gbuzhf\Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Mini-v2D-lite.gguf
--host 127.0.0.1 --port 8096 --no-webui -lv 4 -c 32768 -np 1 -t 6 -fa on -ctk q8_0 -ctv q8_0 --cpu-moe --moe-gpu-expert-slot-num 96 --moe-qstar
```

env は `LLAMA_MOE_SLOT_STATS=1` + `LLAMA_MOE_QSTAR_STATS=1` + `GGML_CUDA_DISABLE_GRAPHS=1`
(`.\bench_glru_qstar.ps1 -Mode qstar -Slots 96 -Rounds 1 -BinaryPath <exe>` なら自動設定、
ただし内部プロンプトを使わず手動 POST するためサーバ停止のみ回避)。

注意: Ornith 側 GGUF が exec_build のレイアウト拒否 (Q4_K の n_experts が ne[2] / ne[1]
問題) を踏むと `qstar_cpu=0` のままになる。その場合は受入① FAIL として記録し、
GGUF レイアウト情報 (n_experts の実際の ne 位置・n_ff/expert 数) を A へ報告する。

## 実行手順 (server 手動起動 → ランナー → 集計)

```powershell
# [1] サーバ起動 (上記起動引数、Huihui / Ornith で -m を差し替え)
$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'; $env:HIP_VISIBLE_DEVICES = '0'
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:GGML_CUDA_DISABLE_GRAPHS = '1'; $env:LLAMA_MOE_SLOT_STATS = '1'; $env:LLAMA_MOE_QSTAR_STATS = '1'
$exe = '<llama-server.exe フルパス>'
$model = '<gguf フルパス>'
$outLog = "$pwd\acc_out.log"; $errLog = "$pwd\acc_err.log"
$p = Start-Process -FilePath $exe -ArgumentList '-m',$model,'--host','127.0.0.1','--port','8096','--no-webui',
    '-lv','4','-c','32768','-np','1','-t','6','-fa','on','-ctk','q8_0','-ctv','q8_0','--cpu-moe',
    '--moe-gpu-expert-slot-num','96','--moe-qstar' `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -WindowStyle Hidden
# /health ok 待ち → warmup ('Say hello.' n_predict=16)

# [2] 短文ランナー (P1-P3 × r1-r3、結果を目視で結果表へ)
$promptFile = 'acc_prompts.ps1'   # 上記 P1/P2/P3 文面を here-string で保存したもの
. .\$promptFile
foreach ($k in 'P1','P2','P3') {
  foreach ($rd in 1,2,3) {
    $body = @{ prompt = $prompts[$k]; n_predict = 128; temperature = 0.3 } | ConvertTo-Json -Compress
    try {
      $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8096/completion' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
      "{0} r{1}: tg={2:N2} t/s ({3} tok)" -f $k, $rd, $r.timings.predicted_per_second, $r.tokens_predicted
    } catch { "{0} r{1}: REQUEST-FAILED {2}" -f $k, $rd, $_.Exception.Message }
    Start-Sleep -Seconds 1
  }
}

# [3] 長文 6575 tok (verify_b.ps1 と同一ジェネレータ・55 iter = 実測 6575 tok)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('Review the following C++ code and summarize potential issues concisely.')
for ($i = 0; $i -lt 55; $i++) {
  $lines.Add("static void worker_stage_$i(context_t * ctx, const batch_t * batch, uint32_t flags) {")
  $lines.Add("    assert(ctx != NULL && ctx->magic == CTX_MAGIC_$i);")
  $lines.Add("    for (uint32_t j = 0; j < batch->count; ++j) { accumulate(&ctx->acc, batch->items[j].value, flags & FLAG_MASK_$i); }")
  $lines.Add("    if (ctx->pending >= THRESHOLD) { flush_stage(ctx, $i); ctx->pending = 0; }")
  $lines.Add("}")
}
$body = @{ prompt = (($lines -join "`n") + "`n"); n_predict = 128; temperature = 0.3 } | ConvertTo-Json -Compress
$r = Invoke-RestMethod -Uri 'http://127.0.0.1:8096/completion' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
"long: pp={0:N1} ({1} tok) tg={2:N2} t/s" -f $r.timings.prompt_per_second, $r.tokens_evaluated, $r.timings.predicted_per_second

# [4] 集計 → coder-b-qstar-cpu-aggregation-2026-08-26.md のレシピで qstar_cpu 判定
# [5] Stop-Process -Id $p.Id -Force
```

Ornith は [1]-[4] をモデル差し替えでもう一巡。
