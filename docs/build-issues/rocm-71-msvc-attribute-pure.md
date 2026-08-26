# ROCm 7.1 + MSVC + Ninja で math_fwd.h:88-136 C2059 エラー

## 症状
- クリーンリビルド (cmake -G Ninja + MSVC) で `C:/Program Files/AMD/ROCm/7.1/include/hip/amd_detail/math_fwd.h:88-136` の `__attribute__((pure))` が C2059 (構文エラー: 'constant') を多発して停止
- q* 機能と無関係、 llama.cpp 本体 ggml 経路で発生
- 増分ビルド (既に build-hip がある状態での cmake --build) では発生しない

## 影響範囲
- q* round-2 で debug ビルドを作る時にこのエラーが先に出る
- ASan 入り debug ビルドもクリーンリビルドが必要なので同じ罠

## 回避策
1. **既存 build-hip を流用して増分ビルド** (推奨)
   - `-DCMAKE_BUILD_TYPE=Debug` を最初に付けると
     既存 build-hip の中身が上書き/不整合になるので不可
   - 別ディレクトリに build-hip-debug を作る
2. **HIP clang 経路 (推奨度 低、本機未確認)**
   - `C:/Program Files/AMD/ROCm/7.1/bin/clang++.exe` を CMAKE_CXX_COMPILER に指定
   - ヘッダーパスも ROCm の include を先に通す
3. **`-O0 -g` + printf trace (推奨度 高)**
   - Debug ビルドを諦めて Release のまま -O0 相当に
   - qstar_cpu_exec の入口/出口に print 挟んでクラッシュサイト絞り込み
4. **直接 ASan 無しで死ぬ場合は minidump + stack walk**
   - 現状 q* クラッシュは「listening を出さず終了」なので
     dmp ファイルは出ない、 stderr にも assertion 無し
   - 唯一の頼りは exec_prepare 直後の printf 差し込み

## 既知の本機 config
- cmake: Ninja generator, MSVC toolchain, 既存 build-hip は Release -O3
- ROCm 7.1, gfx1101, Windows 11
- math_fwd.h:88-136 の __attribute__((pure)) は C2059 ではなく
  C2146 (識別子がない) や C2061 (構文識別子) が出ることもある
  出力に応じて切り分け
