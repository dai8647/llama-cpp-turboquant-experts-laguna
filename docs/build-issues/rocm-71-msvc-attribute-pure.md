# ROCm 7.1 + MSVC `__attribute__((pure))` 互換性問題

## 症状
- 環境: Windows 11, MSVC (C:/Program/VC/ = VS BuildTools), ROCm 7.1
- ターゲット: gfx1101 (RX 7800 XT)
- configure: `cmake -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1101 -DCMAKE_BUILD_TYPE=Release ..` → 成功
- build: `ninja -j8 llama-server llama-cli llama-bench` → 失敗

エラーメッセージ (math_fwd.h 88-136):
```
hip\amd_detail\math_fwd.h(88): error C2059: 構文エラー: ')'
hip\amd_detail\math_fwd.h(94): error C2065: 'pure': 定義されていない識別子です。
hip\amd_detail\math_fwd.h(94): error C2374: '__attribute__': 再定義されています。2 回以上初期化されています。
hip\amd_detail\math_fwd.h(136): fatal error C1003: プログラム内のエラーが 100 個を超えました。
```

## 原因
ROCm 7.1 のヘッダ `C:/Program Files/AMD/ROCm/7.1/include/hip/amd_detail/math_fwd.h` が
`__attribute__((pure))` を多用しているが、 MSVC はこの GCC 拡張属性を認識しない。
- `__attribute__` 自体は MSVC も知っている (空マクロ扱い)
- `pure` 識別子が見つからず構文エラー
- `__attribute__` が 2 度初期化された扱いになる (C2374)

## 影響範囲
- MSVC で `ggml-cuda` をビルドする全構成
- main HEAD (76adf21e4) / feat/b-only (a414acc7f) ともに同じヘッダを読むので影響
- **q* drop (a414acc7f) とは無関係** — ヘッダ起因の既存問題

## 回避策 (2026-08-26 時点で未確認)

### 1. clang-cl を試す
ROCm 7.1 は clang を正式サポート。 ビルドツールを clang-cl に切り替えれば
`__attribute__((pure))` を正しく解釈する可能性。
```
set CC=clang-cl
set CXX=clang-cl
cmake -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1101 ..
```
未検証 (この fork では MSVC 固定運用)。

### 2. PCH / include ガードで `pure` をマクロ化
ビルド前に `pure` を `__attribute__((pure))` に展開するよう PCH 設定。
ただし AMD 公式 HIP ヘッダを改変しない方が安全なので、PCH 側で吸収する方が現実的。

### 3. math_fwd.h のローカルパッチ
`#define __attribute__(x)` を空マクロ化 (純粋に MSVC 向け) する wrapper を
`third_party/rocm-patches/math_fwd_no_attribute.h` として用意し、
`-include` で先頭に注入。
副作用大きいため要検証。

### 4. ビルドターゲットを CPU のみにする
HIP を無効化すれば math_fwd.h は include されないが、 RX 7800 XT で
MoE GPU expert slot cache の動作確認ができなくなる。

## 推奨アクション
- 短期: 既存 `build-hip/bin/llama-server.exe` (8fa0959bc ビルド) で運用継続
- 中期: clang-cl ビルドを再試行 (B / A 環境いずれか)
- 長期: AMD に issue 報告 (`__attribute__((pure))` が MSVC で壊れる件)

## 関連コミット
- main 76adf21e4 (q* drop 込み merge)
- feat/b-only a414acc7f
- a414acc7f 自体は本問題と無関係 (q* body 削除のみ)

## 関連ファイル
- `C:/Program Files/AMD/ROCm/7.1/include/hip/amd_detail/math_fwd.h:88-136` (cl.exe 失敗位置)
- `C:/Program/VC/Auxiliary/Build/vcvars64.bat` (BuildTools vcvars、 通例のパス)
