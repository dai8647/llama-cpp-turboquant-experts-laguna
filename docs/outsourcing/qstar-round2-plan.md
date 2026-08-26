# q* round-2 切り分け計画 (2026-08-26 開始)

## 状況
- main = `76adf21e4` (Bのb-only merge採用)
- q*本体は `a414acc7f` でdrop済(395行削除)
- ブランチ `feat/qstar-debug-round2` (76adf21e4起点) で再着手
- 旧 feat/qstar-global-lru (62f8d92c7) はq*本体実装があったがmainからは消えている
  → q*本体をcherry-pickで戻すのは要再設計。 まずは**最小再現**を目指す

## 既知の障害
1. **engine buildは通る**: layer 0 ready OK
   - Huihui Q4_K mixed: `n_embd=2048 n_ff=512 r=8 fused=0 type=14`
   - Ornith unified:    `n_embd=2048 n_ff=512 r=8 fused=0 type=11`
2. **calibrate内の3回qstar_cpu_execループで死亡**
   - 症状: サーバが `bank_ensure` 後にlistening を出さず終了
   - threads=3で死亡、threads=1 (`LLAMA_MOE_QSTAR_THREADS=1`) でも死亡
   - threadpool alloc失敗ではない(q* threadpool alloc failed 警告出ない)
3. **qstar_threads=3の threadpool_newは成功している**が、その後の
   `ggml_graph_compute` 初回実行で死亡と推定

## 切り分け手順
### 1. debug ビルド + ASan で再ビルド
```bash
cmake -B build-hip-debug -G Ninja \
  -DGGML_HIP=ON -DGGML_ROCM=ON -DAMDGPU_TARGETS=gfx1101 \
  -DCMAKE_BUILD_TYPE=Debug -DGGML_SANITIZE_THREAD=ON \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
cmake --build build-hip-debug
```
- 必要なら ROCm 7.1 HIP clang で `-fsanitize=address` が使えるか確認
- できない場合は -O0 -g + printf trace で代替

### 2. qstar_cpu_exec 1要素クエリのスタックトレース取得
- 死亡箇所を絞り込む: 1要素 `n=1` の `mul_mat_id` なのか、
  `ggml_graph_compute` 内のbarrierなのか
- 旧 `991bf3042` の `llama_moe_qstar_cpu_exec` を最小再現で呼び出す
  単独テストハーネスを書く

### 3. w_in0/w_in1 のテンソル型不一致を確認
- 旧 mixed-precision enable (`8a0a55bbd`) を再適用したか
  どうかに関係なくq* computeで死亡するなら、 q* エンジンの
  ミニグラフは**単一量子化のみ想定**なのかもしれない
- `w_in0` が Q5_1 / Q4_K / Q4_0 など混合時の `ggml_mul_mat_id`
  の挙動を ggml-cpu 実装で再確認

### 4. 受入基準の修正
- 短文tg: 「3つの異なる短文プロンプト、各3ラウンド、1ラウンド目破棄」
- q* host execがengaging しなければ q* は不採用
- 真の寄与ゼロ状態が観測された場合は **q*機能 revert + 単独再実装** へ

## 残TODO (次回セッション)
- [ ] debugビルドで再現
- [ ] crash siteの特定
- [ ] 根本原因(おそらくテンソル型不一致 or quant GEMM dead code)
- [ ] 修正 or 機能revertの判断
- [ ] 修正後 B の `bench_glru_qstar.ps1 -Mode qstar` で再検証依頼
