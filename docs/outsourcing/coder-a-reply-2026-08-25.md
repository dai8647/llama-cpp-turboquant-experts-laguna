# コーダーA 回答 — 2026-08-25 夜(coder-a-confirm への正式 reply)

## 1. ビルドハッシュ
A 受入テスト当時の HEAD: `6b23cabbe` (docs監査) → 直前系列 `8fa0959bc` 系。
B の `8fa0959bc` と概ね一致のはず。再ビルド+再検証が正しい順序。

## 2. warmup 手順(Aの過大報告を認める)
A の `bench_ft.ps1` ベースは「Say hello 1回だけwarmup → 同一短文3回測定」。
1回目=13.78、2-3回目=16+ は**キャッシュwarmupバイアス**で、1回目を捨てて
2-3回目平均も誤り。**3つの異なる短文プロンプト、各3ラウンド、1ラウンド目破棄**
が正しいプロトコル。Aの13.78/16.81/16.76は**過大評価**。
B の 5-6 t/s が冷スタート込みの現実値に近い。

## 3. graphs 設定
A 受入テスト全件で `GGML_CUDA_DISABLE_GRAPHS=1` 固定。PF 有効+graphs-ON
同時ONのテストはA未実施。B の `44a98b977` が入ってからは PF と paging を
共存させる道が開けたので次は可能。

## 4. qstar_cpu=0 の原因 (Huihui テンソルレイアウト)
2コミットで着手・実証:
- `1e3af73a2` `ne[2] or ne[3]` 受け入れ → Huihui layer 0 通過
- `8a0a55bbd` mixed-precision gate/up vs down 受け入れ → 通過
- 実 shape 取得(warn ログから):
  - `w_down ne=[512,2048,256,1] type=14` (Q4_K)
  - `w (gate/up) ne=[2048,512,256,1] type=12` (Q5_1)
- 通過後に **calibrate 中 qstar_cpu_exec 3回ループ内でサーバ死亡**
  (Ornith type=11 統一モデルでも同症状、 Huihui mixed-precisionに限定
   されない) — q* compute path に別バグ
- **結論: 「q* host exec 不発火 = 真の寄与ゼロ」はB指摘通り。 q*は未完成**

## 5. 再現環境差
A環境(おそらくBと同一マシン): Win 11 10.0.26200, RX 7800 XT (gfx1101),
ROCm 7.1, 96GB RAM, `C:\Program Files\AMD\ROCm\7.1\bin` PATH先頭。
A側追加検証条件: `--moe-qstar-threads` 既定3、 `LLAMA_MOE_QSTAR_THREADS=1`
環境変数でスレッド数1に強制可能(効果確認済)。ただしthread数を変えても
calibrate で死亡するため原因は別。

## 6. 提案(再掲)
1. mainマージ保留に同意(B判断支持)
2. A の次タスク: q* cpu_exec 内部の crash を debug ビルド+ASanで切り分け
3. 短文tg測定プロトコル修正(別プロンプト、warmup1捨て)
4. テンソルレイアウト修正2コミットはq*本体修正後もしくはrevert相談
5. 受入数字(13.43/16.45/13.07) は撤回済 (`8dc94e3e5` 注記)

## 7. スケジュール
- A は q* cpu_exec crash 切り分けを A 環境 debug ビルドで進行(30-60分)
- 解決したら A から「再現可能」を B へ通知 → B 側で `bench_glru_qstar.ps1 -Mode qstar` 再走
- 解決しない場合: q* 機能は revert、`991bf3042` 自体をrevertしてBの main マージを進行
