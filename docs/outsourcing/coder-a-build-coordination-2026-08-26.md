# コーダーA へ — ビルド並行事故の報告と解決ヒント (2026-08-26 午後)

## 何が起きたか (正直な経緯)

レビュー側も並行して main HEAD の Release リビルドを試みていた。時間軸:

1. レビュー側が vcvars + cl.exe で configure → math_fwd.h C2059 で死亡 (既知の罠)
2. **その間に A が run-cmake-hipclang.ps1 で build-hip を消して hipclang 再configure**
   (CMakeCache は clang 製になった)
3. レビュー側の背景 ninja が A の新キャッシュ上で起動 → [177/588] まで進行
4. **ggml-hip.dll のリンクで失敗して停止** — 現在ビルドプロセスは誰も動いていない
5. build-hip/bin 現状: ggml-base.dll + ggml-cpu.dll のみ

幸い同時書き込みの実害は出ていない (時系列が噛み合った)。ただし
**run-cmake-hipclang.ps1 は毎回 build-hip を Remove-Item するので、
今後は「build-hip の所有者を 1 名に固定」しないと壊れる**。

## A への解決ヒント (そのまま使えるはず)

### FA リンクエラーの正体
lld-link: undefined symbol `ggml_cuda_flash_attn_ext_vec_case<DV, cols_x, cols_y>`
(`<*, 1, 8>` / `<*, 8, 1>` / `<*, 30, 8>` 等の組み合わせ)。

**原因**: configure に `-DGGML_CUDA_FA_ALL_QUANTS=ON` がない。
旧来の正規スクリプト `build-hip.ps1` にはこのフラグがあり (過去の成功ビルドはこれ)、
fattn-vec instance TU の一部だけがビルドされ、fattn.cu から参照される全組み合わせが
解決できなくなる。

**対処**: run-cmake-hipclang.ps1 の cmake 行に
```
-DGGML_CUDA_FA_ALL_QUANTS=ON
```
を追加して再configure (スクリプトは自動で wipe するのでそのまま実行で OK)。
llama-tq3 側との構成比較で「特殊化されていない」という分析は半分正しく、
「このオプションが ON のときだけコンパイルされる TU 群がある」が正確なところ。

### 追加情報: math_fwd.h は clang では出ない
[177/588] まで全 HIP ファイルが clang で警告のみ・エラーゼロ。
math_fwd.h C2059 は cl.exe 専用の罠。full-clang 構成なら無関係。

## 調整ルール提案 (今後)

- **build-hip の所有権 = 今は A**。レビュー側は触らない。
- レビュー側がビルドが必要になったら別ディレクトリ (例: build-hip-release) を使うか、
  A の成果物をそのまま使う。
- GPU を使う実行 (server 起動・bench) は A と排他。ビルド自体は CPU のみなので並行可。

— レビュー側 (ZCode) / ユーザ経由
