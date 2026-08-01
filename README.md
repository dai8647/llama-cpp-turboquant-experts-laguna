# llama-cpp-turboquant: Experts-First + Laguna MoE

## 概要

TheTom/llama-cpp-turboquant (AMD HIP/ROCm対応llama.cpp fork) に、以下の機能を統合したfork:

- **Experts-first MoE GPU expert slot** — leoustc/llama.cpp-moe からの移植
- **Laguna architecture support** — upstream PR #25165 (merged 2026-07-22) の移植
- **Frequency-based expert placement** — Expert選択頻度に基づくGPU配置

## 対象ユーザー

- AMD Radeon GPU (ROCm HIP) を持つローカルLLMユーザー
- Laguna S 2.1 (118B) や Laguna XS 2.1 (33B) をローカルで動かしたい人
- MoEモデルでGPU expert配置を制御したい人

## 主要機能

### 1. Experts-first MoE GPU Expert Slot

Expert単位でGPU VRAMへの配置を制御。全ExpertをGPUに事前配置（full-slot）或いは頻度ベースの部分配置が可能。

```bash
# 全ExpertをGPUに配置 (full-slot mode)
llama-cli -ngl 999 --moe-gpu-expert-slot-num 999 -m model.gguf

# 頻度ベースの部分配置 (例: 上位50%だけGPU)
# ただしfrequencyモードは2-pass方式: Pass1で統計収集、Pass2で配置適用
# Pass1: 統計収集 (--moe-freq-report-path でJSON出力)
# Pass2: 統計JSONを読み込んでfrequency配置 (-nは省略可、書かないと自動継続)
llama-cli -ngl 999 --moe-gpu-expert-slot-num 999 --moe-expert-placement frequency --moe-gpu-expert-ratio 0.5 --moe-freq-report-path stats.json -m model.gguf -n 200
# または --moe-gpu-expert-slot-num を省略 = 自動全slot有効化

# Expert選択統計をJSON出力
llama-cli -ngl 999 --moe-freq-report-path stats.json -m model.gguf
```

### 2. Laguna Architecture

118B MoEモデル (Laguna S 2.1) と 33B MoEモデル (Laguna XS 2.1) をサポート。

- 256 routed experts (top-10) + 1 shared expert
- Sliding-window attention (hybrid full/SWA)
- Softplus attention gate, QK-norm
- Sigmoid-routed MoE with score-correction bias

### 3. Frequency-Based Expert Placement

Expert選択頻度を記録し、高頻度ExpertをGPUに優先配置。

#### 2パスワークフロー

1. **Pass 1 (計測)**: 全expertをロードして推論し、各expertの使用頻度を記録
2. **Pass 2 (配置最適化)**: 頻度JSONから高頻度expertだけをGPUに配置

| フラグ | 用途 |
|---|---|
| `--moe-freq-report-out PATH` | Pass 1: 頻度統計をJSONに書き出す (track_access有効化) |
| `--moe-freq-report-in PATH` | Pass 2: 頻度JSONを読み込み、上位expertをGPU配置 |
| `--moe-freq-report-path PATH` | [DEPRECATED] 両方を兼ねる旧フラグ |
| `--moe-expert-placement frequency` | 頻度ベース配置モード |
| `--moe-gpu-expert-ratio 0.6` | GPU配置割合 (0.0-1.0) |

使用例:
```bash
# Pass 1: 統計収集 (full-slotでもOK、track_accessだけ有効)
./llama-cli -m model.gguf --moe-freq-report-out stats.json -p "..." -n 256

# Pass 2: 高頻度expertだけGPU配置
./llama-cli -m model.gguf --moe-expert-placement frequency \
    --moe-freq-report-in stats.json --moe-gpu-expert-ratio 0.6 -p "..." -n 256
```

## ビルド

### 前提条件

- Windows + AMD ROCm 7.1
- Ninja build system
- CMake

### ビルド手順

```bash
# ROCm 7.1環境でビルド
set PATH=C:\Program Files\AMD\ROCm\7.1\bin;%PATH%
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.1

cmake -S . -B build-hip -G Ninja -DGPU_TARGETS=gfx1100 -DGGML_HIP=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --config RelWithDebInfo -j
```

## テスト結果

### Laguna XS 2.1 (33B-A3B, Q4_K_M)

| モード | Prompt t/s | Generation t/s |
|--------|------------|----------------|
| 通常 | 113.3 | 24.5 |
| slot (999) | 41.5 | **36.2** |
| frequency (0.5) ^1 | 124.4 | 24.5 |

### Laguna S 2.1 (118B, IQ3_XXS)

| モード | Prompt t/s | Generation t/s |
|--------|------------|----------------|
| 通常 | 24.1 | 4.6 |
| slot (999) | 18.2 | **9.7** |
| frequency (0.5) ^1 | 24.3 | 4.6 |

^1 frequency numbers are pre-fix (v0.1). The bug where frequency mode silently fell through to slot-off has been fixed; post-fix numbers will differ — re-run bench_frequency_placement.ps1 to get current values.

- 全モードで出力一致 (temp=0, seed=42)
- HIP/ROCm 7.1, AMD GPU
- flash-attn off (ROCm SWA層クラッシュ回避)

## クレジット

- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — ベースfork
- [leoustc/llama.cpp-moe](https://github.com/leoustc/llama.cpp-moe) — experts-first実装
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — upstream (Laguna PR #25165)
- [poolside/Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) — Lagunaモデル

## ライセンス

TheTom/llama-cpp-turboquant と同一 (MIT)
