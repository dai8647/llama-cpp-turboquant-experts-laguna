# FreeToken hot-expert mode (Qwen3.8-Flash-Next / RX 7800 XT)

目標: expert は RAM、VRAM には hot expert の LRU キャッシュだけ置く。

## 起動（推奨）

```powershell
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.1'
$env:HCC_AMDGPU_TARGET = 'gfx1101'
$env:GGML_ROCM_MAX_VRAM = '15360'

$M = 'C:\Users\dai86\.lmstudio\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'

llama-cli -m $M -ngl 99 `
  --moe-hot-expert `
  -lm mmap --lazy-mode on `
  -c 8192 -p "hello" -n 64
```

`--moe-hot-expert` は以下をまとめて有効化する:

- expert 重みを host (CPU/RAM) に固定 (`--cpu-moe`)
- slot 数を free VRAM から auto (`--moe-gpu-expert-slot-num auto`)
- global LRU ページング (`--moe-gpu-expert-global-lru`)

dense パス (attn / SSM / HC / embedding) は `-ngl 99` で VRAM。
n-gram shard2 は mmap のままディスクに置く。

## 確認ログ

起動時に以下が出れば hot-expert 経路:

```
MoE GPU expert slot auto sizing: ... slots_budget=... slots=...
MoE GPU expert slot bank pinned stage: layer=...
MoE GPU expert slot cache initialized with N slots (auto) global_lru=1
q*/global-LRU paging active - CUDA graphs disabled
```

slot 統計 (`LLAMA_MOE_SLOT_STATS=1`):

```
copies=... hit=... miss=... evict=... copy=... MiB avg=... ms
```

## online frequency pin のログ出し方 (HOST_BANK=0)

**禁止**: `LLAMA_MOE_HOST_BANK=1` での起動・実測。
ON は load 時に全層 expert を pinned へ詰め、CPU~95% / Disk~100% / ROCm error。
デフォルトは `HOST_BANK=0` のまま。実測は D / F セッションが排他で行う。

online pin は `--moe-hot-expert` だけで有効。`HOST_BANK=0` + `LLAMA_MOE_SLOT_STATS=1` で十分。

```powershell
$env:LLAMA_MOE_HOST_BANK = "0"   # 必須
$env:LLAMA_MOE_SLOT_STATS = "1"
# 既定で auto_pin_after_access=2000。上書きする場合のみ:
# $env:LLAMA_MOE_AUTO_PIN_AFTER = "2000"

llama-cli -m $M --moe-hot-expert -ngl 99 -c 8192 -t 16 --reasoning off -p "..." -n 128 2> pin_err.log
```

stderr に現れる順:

```
online frequency pin armed: after 2000 accesses, track_access=1
MoE GPU expert slot bank pinned stage: layer=...          # HOST_BANK=0 のときの通常経路
MoE GPU expert slot cache initialized with N slots (auto) global_lru=1
runtime frequency pin: P experts (K/layer max) after hit=H miss=M access_sum=S layers=L
MoE GPU slot stats: copies=C hit=H miss=M evict=E copy=X MiB avg=Y ms
```

ゲート:
- pin 1 回だけ (`auto_pin_done`)。ログに `runtime frequency pin` が 1 行
- pin 後 hit が増え miss/copy が減ること
- `MoE host expert bank pinned slab` が出たら **HOST_BANK が ON** になっている。すぐ停止して 0 に戻す

起動ヘルパ (`run-qwen4exp-bank.cmd`) は `HOST_BANK=0` 固定。
`Downloads/run-online-pin-bench.cmd` も同じ制約で使うこと。

## auto slot の意味

1 slot = 全 MoE 層それぞれの expert 1 個分の合計。
`slots = (free_vram * margin) / unit_bytes`。
`n_expert_used` を下回っても **OOM を避けて縮小**する (強制拡大しない)。

## 次のレバー (P1)

1. pinned staging (実装済み) の H2D 実測
2. decode 経路の async H2D + 早期 fire
3. frequency pin (Pass1 → `--moe-expert-placement frequency`)

## 参考

- docs/research/freetoken-gap-adoption-plan-2026-08-27.md
- docs/research/freetoken-16gb-community-and-mechanism-2026-08-27.md
