"""Set selected SO-101 6-DOF motor IDs, one physically isolated motor at a time."""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


MOTOR_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_yaw": 5,
    "wrist_roll": 6,
    "gripper": 7,
}


def find_single_motor(bus, retries=3):
    """Read-only discovery. Retry malformed SDK responses before any EEPROM writes."""
    expected = bus.model_number_table["sts3215"]
    for baudrate in bus.model_baudrate_table["sts3215"]:
        bus.set_baudrate(baudrate)
        for attempt in range(retries):
            bus.port_handler.clearPort()
            time.sleep(0.1)
            try:
                found = bus.broadcast_ping()
            except IndexError as exc:
                print(f"応答データ不足: baudrate={baudrate}, 再試行 {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise RuntimeError(
                        "短い応答が続いたため、追加の書き込みをせず停止しました。"
                        "電源を切り、対象1台だけの接続・ケーブル・電源を確認してください。"
                    ) from exc
                continue
            if not found:
                continue
            if len(found) != 1:
                raise RuntimeError(f"複数のIDを検出しました: {found}。対象の1台だけ接続してください。")
            old_id, model = next(iter(found.items()))
            if model != expected:
                raise RuntimeError(f"STS3215と異なるモデル番号です: {model} (expected {expected})")
            return baudrate, old_id
    raise RuntimeError("モーターが見つかりません。対象1台の配線・電源・ポートを確認してください。")


def setup_one(port, name):
    target_id = MOTOR_IDS[name]
    mode = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES
    bus = FeetechMotorsBus(port=port, motors={name: Motor(target_id, "sts3215", mode)})
    input(f"電源を切って '{name}' だけを接続し、電源を入れてからEnter (設定ID={target_id}): ")
    try:
        bus.connect(handshake=False)
        baudrate, old_id = find_single_motor(bus)
        print(f"検出: {name}, 現在ID={old_id}, baudrate={baudrate}; ID {target_id} に設定します。")
        bus.setup_motor(name, initial_baudrate=baudrate, initial_id=old_id)
        # Verify by read-only discovery; do not repeat writes if verification fails.
        verified_baudrate, verified_id = find_single_motor(bus)
        if (verified_baudrate, verified_id) != (bus.default_baudrate, target_id):
            raise RuntimeError(f"設定後の確認に失敗: baudrate={verified_baudrate}, ID={verified_id}")
        print(f"確認完了: {name} = ID {target_id}")
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--motors", nargs="+", choices=list(MOTOR_IDS), required=True)
    args = parser.parse_args()
    for name in args.motors:
        setup_one(args.port, name)


if __name__ == "__main__":
    main()
