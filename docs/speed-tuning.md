# DeepSeek V4 高速化チューニング（ds4 流）

このガイドは antirez の `ds4`（DwarfStar）が 128GB MacBook で DeepSeek V4 Flash を
~35 t/s で回している技術を、このフォーク（ROCm / CPU オフロード）に適用するための
手順書です。

## 1. なぜ Mac で 35 t/s が出るのか（前提の物理）

decode は「1 トークンごとにアクティブなパラメータをメモリから読み出す帯域」で決まります。

| ハード | メモリ帯域 | Q2 エキスパート想定 |
|---|---|---|
| M5 Max 128GB (統合) | ~500 GB/s | ~35 t/s（実測） |
| Strix Halo (Framework Desktop) | ~256 GB/s | ~15-18 t/s 想定 |
| 7800 XT 16GB + DDR4 96GB | VRAM ~624 GB/s / CPU ~51 GB/s | **CPU 律速なら ~5-9 t/s** |

つまり **35 t/s は帯域の物理**であって、7800 XT + DDR4 構成では同じモデルで
その数字は出ません。目標は「帯域消費を減らす」ことで、以下 2 つが本丸です。

1. **エキスパートだけ 2bit に落とす非対称量子化**（帯域消費をほぼ半減）
2. **DSpark 投機デコード**（1 回の読み出しで複数トークン進める）

このフォークは両方に対応済み（後述）。対応ツールは `scripts/` に同梱しました。

## 2. フォークに既にあるもの（再確認）

| 機能 | このフォーク | ds4 |
|---|---|---|
| 高速 2bit 型 TQ2_0 (2.06 bpw) | ✅ 専用デコードカーネル | IQ2_XXS を使用 |
| DSpark/MTP 投機デコード | ✅ 上流 #25784 由来 (`--mtp` / `--dflash` / `--spec-type draft-dspark`) | 同 DSpark |
| KV キャッシュ圧縮 | ✅ turbo KV (turbo2/3/4) | 独自圧縮 KV |
| ホットエキスパート配置 | ✅ `--moe-expert-placement {all-gpu,frequency,cpu-moe,map}` | エキスパートキャッシュ |
| imatrix | ✅ `tools/imatrix` | 自前ツール |

## 3. 手順 A: エキスパート 2bit 混合 GGUF を作る

```bash
# 前提: F16/BF16 の高精度 GGUF を用意（HF の元 GGUF など）

# デフォルト: エキスパート=TQ2_0, ルーター/共有エキスパート=F16, 他=Q8_0
python3 scripts/mixed_quant.py model-f16.gguf model-tq2.gguf

# 計画だけ見る
python3 scripts/mixed_quant.py --print-plan model-f16.gguf /dev/null

# 比較用: TQ1_0（さらに小さい 1.7bpw 級）
python3 scripts/mixed_quant.py --expert-type TQ1_0 model-f16.gguf model-tq1.gguf

# 品質重視: base も F16 に
python3 scripts/mixed_quant.py --base-type F16 model-f16.gguf model-tq2-f16base.gguf
```

テンソル割り当て（deepseek2/4, qwen3moe 等の標準 MoE 命名）:

| パターン | 型（既定） |
|---|---|
| `blk.*.ffn_{gate,up,down}_exps.weight`（ルーティングエキスパート） | TQ2_0 |
| `blk.*.ffn_{gate,up,down}_shexp.weight`（共有エキスパート） | F16 |
| `blk.*.ffn_gate_inp.weight`（ルーター） | F16 |
| その他（attention / 投影 / output） | Q8_0 |

`--keep-pattern` で任意の正規表現を F16 固定に追加できます。

### サイズ感（DeepSeek V4 Flash 想定）

- エキスパートがモデル体積の ~90% を占めるため、TQ2_0 化で全体 ~40% 減
- 96GB RAM 機なら「TQ2_0 エキスパート + Q8_0 ベース」で RAM に全搭載できるのが目標
- 品質確認は `tools/perplexity`（下記 手順 C）

### 注意

- 入力は F16/BF16 GGUF 推奨（既に Q4 等の低精度だと再量子化の劣化が蓄積する）
- マルチシャード GGUF は未対応（先に `gguf-split` で単一ファイル化）
- 変換はテンソル単位で逐次処理するので RAM は小さくて済む
- 選択できる型は gguf-py で量子化実装があるものに限定（Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K..Q6_K/
  IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS/BF16/TQ1_0/TQ2_0）。
  K 系と IQ 系は 2026-08 に quantize 実装を追加（C++ の ggml-quants.c 参照実装の移植で、
  K 系の出力は C 参照とバイト単位で一致）。IQ 系は imatrix 非対応の unweighted 版なので、
  --imatrix と完全一致させたい場合は C++ の llama-quantize を使う

## 4. 手順 B: DSpark の実測（基準と比較）

```bash
MODEL=model-tq2.gguf ./scripts/bench_dspark.sh
```

ベースライン / `--mtp` / `--dflash` / `--spec-type draft-dspark` の 4 ケースで
同一プロンプト（既定はコード生成）の decode t/s を比較します。

- 初回は DSpark サイドカー (~5.6GB) を HF からダウンロードするため時間がかかる
- コードや定型的な続きほど受理率が高く効果大。散文は効果が小さい
- 帯域律速環境では「受理トークンあたりの読み出しバイト」が減るため特に有効

手動でも:

```bash
llama-server -m model-tq2.gguf -c 216000 -ngl 99 --cpu-moe --dflash --temp 0
```

## 5. 手順 C: エキスパート型の比較（TQ2_0 vs TQ1_0 vs Q4_0）

同じ F16 ソースから複数種類を作り、perplexity と実速度を比較します。
Python 経由で選択できるエキスパート型は TQ2_0 (2.06bpw) / TQ1_0 (1.7bpw 級) /
Q2_K (2.625bpw) / Q3_K (3.44bpw) / Q4_K (4.5bpw) / Q5_K (5.5bpw) /
Q6_K (6.56bpw) / IQ2_XXS (2.06bpw) / IQ2_XS (2.31bpw) / IQ2_S (3.06bpw) /
IQ3_XXS (3.44bpw) / IQ3_S (4.44bpw) / IQ4_XS (4.25bpw) / Q4_0 などです
（IQ 系は imatrix なしの unweighted 量子化）。

```bash
# 候補の GGUF を作る
python3 scripts/mixed_quant.py --expert-type TQ2_0 model-f16.gguf m-tq2.gguf
python3 scripts/mixed_quant.py --expert-type TQ1_0 model-f16.gguf m-tq1.gguf
python3 scripts/mixed_quant.py --expert-type Q2_K model-f16.gguf m-q2k.gguf
python3 scripts/mixed_quant.py --expert-type Q4_0  model-f16.gguf m-q4.gguf
python3 scripts/mixed_quant.py --expert-type IQ2_XXS model-f16.gguf m-iq2xxs.gguf

# 品質（小さいほど良い、F16 の PPL が基準）
./build/bin/llama-perplexity -m m-tq2.gguf -f wiki.test.raw -c 4096 -n 2048
./build/bin/llama-perplexity -m m-tq1.gguf -f wiki.test.raw -c 4096 -n 2048
./build/bin/llama-perplexity -m m-q4.gguf   -f wiki.test.raw -c 4096 -n 2048

# 速度（各 GGUF で同じ手順を回す）
MODEL=m-tq2.gguf ./scripts/bench_dspark.sh
```

期待値:

- 品質: Q4_0 / Q4_K > Q3_K > Q2_K > TQ2_0 > TQ1_0（順位はモデル依存、PPL/KLD で確認）
- 速度: TQ2_0 / TQ1_0 はこのフォーク専用のデコードカーネルがあるため Q4_0 より速い
- K 系（Q2_K..Q6_K）は品質を重視しつつ C の llama-quantize と同一出力が欲しい場合の選択肢
  （エキスパート 2bit なら Q2_K、4bit なら Q4_K が典型）
- IQ 系（IQ2_XXS など）は 2026-08 に quantize 実装を追加済み。imatrix なしの
  unweighted 版なので、--imatrix 前提の品質を求めるなら C++ の llama-quantize を使う

## 6. その他の効くスイッチ

```bash
# 頻出エキスパートを VRAM に置く（16GB の範囲で）
--moe-expert-placement frequency

# KV キャッシュを turbo 圧縮（長文脈ほど効く）
--cache-type-k turbo4 --cache-type-v turbo4

# 共有/ルーティング投影の精度を保つための設定は mixed_quant.py 側で対応済み
```

## 7. 現実的な期待値まとめ

| 構成 | 想定 decode |
|---|---|
| 現状（Q4_K_M + DSpark なし） | ~5.5 t/s |
| + TQ2_0 エキスパート混合 | ~7-10 t/s（帯域律速分が改善） |
| + DSpark（コード等） | ~8-15 t/s（コンテンツ依存） |
| 35 t/s | M5 Max 級の帯域が必要（このハードでは物理的に不可） |

最大の一手は **エキスパート 2bit 化**です。次点で **DSpark 有効化**。
両方ともコードに変更は不要で、GGUF と起動フラグだけで適用できます。
