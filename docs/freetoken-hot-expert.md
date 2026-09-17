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
