# SO-101：6軸＋グリッパー

追加モーターは既存と同じ Feetech STS3215 を前提とします。`enable_wrist_yaw=true` を指定すると次の構成になります。省略時は従来構成です。

| ID | 関節 |
|---|---|
| 1 | shoulder_pan |
| 2 | shoulder_lift |
| 3 | elbow_flex |
| 4 | wrist_flex |
| 5 | wrist_yaw |
| 6 | wrist_roll |
| 7 | gripper |

## 環境

```bash
cd /Users/hiroaki.ishikawa/Documents/iot_tech/6dof_so101/lerobot
uv sync --extra feetech
uv run lerobot-find-port
```

以下の `/dev/tty.usbmodemXXXX` を確認したポートに置き換えます。`robot.id` はキャリブレーションを区別する名前であり、モーターの数値IDとは異なります。

## IDを設定する

アームを支え、配線変更の前にはモーター電源を切ってください。コントローラーには設定対象のモーターを **1台だけ** 接続します。下流に他のモーターをつなげないでください。特に既存gripperと追加wrist_yawが同じIDでも、単独接続なら順に設定できます。

```bash
uv run lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX \
  --robot.id=so101_6dof_follower \
  --robot.enable_wrist_yaw=true
```

画面に表示される順序は **gripper → wrist_roll → wrist_yaw → wrist_flex → elbow_flex → shoulder_lift → shoulder_pan** です。それぞれ対象の1台だけを接続し、電源を入れてからEnterを押します。IDは順に7、6、5、4、3、2、1になります。コマンドはIDと通信速度を書き換えます。

## 全台接続してキャリブレーション

設定完了後、電源を切って7台を接続し直し、電源を入れます。旧構成とは異なる `robot.id` を使ってください。

```bash
uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX \
  --robot.id=so101_6dof_follower \
  --robot.enable_wrist_yaw=true
```

中央姿勢を設定後、画面の案内に従い関節を動かします。追加したwrist_yawも実際に使う可動範囲を手で動かして記録します。wrist_rollのみ従来実装どおり全周範囲として扱います。機械的な干渉やケーブルのねじれでwrist_rollの範囲が制限される構造なら、実機運用前にそのキャリブレーション処理も変更してください。

## テレオペレーション・記録

以後のrobot設定にも毎回 `--robot.enable_wrist_yaw=true` と同じ `--robot.id` を指定します。観測・アクションには `wrist_yaw.pos` が追加され、gripperを含む7要素になります。旧6要素のデータセットやポリシーはそのまま流用できません。

リーダーも同じ6軸＋gripper構成に改造した場合は、セットアップとキャリブレーションを次の指定で別途実行します。

```bash
uv run lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemYYYY \
  --teleop.id=so101_6dof_leader \
  --teleop.enable_wrist_yaw=true

uv run lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemYYYY \
  --teleop.id=so101_6dof_leader \
  --teleop.enable_wrist_yaw=true

uv run lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX \
  --robot.id=so101_6dof_follower \
  --robot.enable_wrist_yaw=true \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodemYYYY \
  --teleop.id=so101_6dof_leader \
  --teleop.enable_wrist_yaw=true
```

標準の5軸リーダーではwrist_yawの指令を生成できません。追加軸を操作する入力方法または固定角の方針が別途必要です。今回の変更は関節単位の制御対応です。改造後の形状に合わせたURDF・逆運動学は含みません。

## 検証

```bash
uv run python -m unittest discover -s tests -p test_so_six_dof_layout.py
```

テストは実際のデバイスクラスを読み込み、通信と基底クラスをスタブに差し替えます。従来／追加軸構成、正規化、特徴量、設定順序、キャリブレーションIDの一致判定を検証します。依存パッケージを含むCLI統合および実機通信・動作は別途確認が必要です。

## 実機配置に合わせた修正（2026-09-19）

ユーザー確認の先端からの配置：gripper → wrist_roll → wrist_yaw → wrist_flex → elbow_flex → shoulder_lift → shoulder_pan。根元から順にID 1〜7を割り当てる方針で、wrist_yawを5、wrist_rollを6に修正しました。物理配置だけで通信IDが決まるわけではありませんが、このプロジェクトでは上記の対応に統一します。

以前の手順でwrist_roll=5、wrist_yaw=6に設定済みなら、それぞれ1台ずつ接続して新しいIDへ設定し直してください。全台接続のままID 5と6を入れ替えるとIDが衝突します。キャリブレーションも再実施し、必要なら新しいrobot.id／teleop.id（例：so101_6dof_v2_follower／so101_6dof_v2_leader）を使用してください。旧順序で収録したデータや学習済みポリシーは、特徴量順序と関節対応の確認が必要です。

ソフトウェアと手順のみ修正済みです。実機IDの変更と動作確認は未実施です。
