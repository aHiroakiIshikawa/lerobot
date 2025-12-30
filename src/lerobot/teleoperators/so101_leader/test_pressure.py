#!/usr/bin/env python

"""
SO101 Leaderアームと圧力センサのテストスクリプト
"""

import sys
import time
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

# ========================================
# ポート設定（環境に合わせて変更）
# ========================================
LEADER_ARM_PORT = "/dev/tty.usbmodem5A7A0161151"
PRESSURE_SENSOR_PORT = "/dev/cu.usbmodem21101"
PRESSURE_THRESHOLD = 2.0  # ニュートン


def test_pressure_sensor_only():
    """圧力センサのみをテスト"""
    print("=" * 50)
    print("圧力センサ単体テスト")
    print("=" * 50)
    
    from lerobot.teleoperators.so101_leader.pressure_sensor import PressureSensor
    
    sensor = PressureSensor(
        port=PRESSURE_SENSOR_PORT,
        baudrate=9600
    )
    
    try:
        sensor.connect()
        print("圧力センサに接続しました")
        
        print("\n10秒間、圧力値を表示します...")
        print("グリッパを握ってみてください")
        
        for i in range(100):
            force = sensor.get_force()
            print(f"圧力: {force:.3f} N", end="\r")
            time.sleep(0.1)
        
        print("\n\n圧力センサテスト完了")
        
    finally:
        sensor.disconnect()


def test_leader_with_pressure():
    """Leaderアームと圧力センサを統合してテスト"""
    print("\n" + "=" * 50)
    print("Leaderアーム + 圧力センサ統合テスト")
    print("=" * 50)
    
    print(f"Leader ARM ポート: {LEADER_ARM_PORT}")
    print(f"圧力センサ ポート: {PRESSURE_SENSOR_PORT}")
    print(f"圧力閾値: {PRESSURE_THRESHOLD}N")
    print()
    
    config = SO101LeaderConfig(
        port=LEADER_ARM_PORT,
        id="hishikawa_leader_arm",
        pressure_sensor_enabled=True,
        pressure_sensor_port=PRESSURE_SENSOR_PORT,
        pressure_threshold=PRESSURE_THRESHOLD,
    )
    
    leader = SO101Leader(config)
    
    try:
        print("Leaderアームに接続中...")
        leader.connect()
        print("接続完了")
        
        print("\nテスト開始:")
        print("1. グリッパを握って圧力を加えてください")
        print("2. 閾値を超えると制限がかかります")
        print("3. 開く方向は常に動きます")
        print("4. Ctrl+C で終了")
        print()
        
        while True:
            # アクションを取得（内部で圧力制限が適用される）
            action = leader.get_action()
            
            # 圧力値を表示
            pressure = leader.read_pressure_sensor()
            gripper_pos = action["gripper.pos"]
            is_limited = leader.gripper_limited_position is not None
            
            status = "⚠️ 制限中" if is_limited else "✓ 正常"
            print(
                f"圧力: {pressure:.3f}N | "
                f"グリッパ位置: {gripper_pos:.2f} | "
                f"{status}",
                end="\r"
            )
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nテスト終了")
    finally:
        if leader.is_connected:
            leader.disconnect()
        print("切断完了")


def test_threshold_tuning():
    """適切な閾値を見つけるためのテスト"""
    print("\n" + "=" * 50)
    print("圧力閾値チューニングツール")
    print("=" * 50)
    
    from lerobot.teleoperators.so101_leader.pressure_sensor import PressureSensor
    
    sensor = PressureSensor(
        port=PRESSURE_SENSOR_PORT,
        baudrate=9600
    )
    
    try:
        sensor.connect()
        print("圧力センサに接続しました\n")
        
        max_force = 0.0
        min_force = float('inf')
        
        print("グリッパで物体を掴んでください...")
        print("最大圧力と最小圧力を記録します")
        print("Ctrl+C で終了\n")
        
        while True:
            force = sensor.get_force()
            
            if force > max_force:
                max_force = force
            if force < min_force and force > 0:
                min_force = force
            
            print(
                f"現在: {force:.3f}N | "
                f"最大: {max_force:.3f}N | "
                f"最小: {min_force:.3f}N",
                end="\r"
            )
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 50)
        print("測定結果:")
        print(f"  最大圧力: {max_force:.3f}N")
        print(f"  最小圧力: {min_force:.3f}N")
        print(f"\n推奨閾値: {max_force * 0.7:.3f}N - {max_force * 0.9:.3f}N")
        print("  (最大値の70-90%)")
        print("=" * 50)
    finally:
        sensor.disconnect()


def main():
    print("SO101 圧力センサテストツール\n")
    print("テストモードを選択してください:")
    print("1: 圧力センサ単体テスト")
    print("2: Leaderアーム + 圧力センサ統合テスト")
    print("3: 圧力閾値チューニング")
    
    choice = input("\n選択 (1/2/3): ").strip()
    
    if choice == "1":
        test_pressure_sensor_only()
    elif choice == "2":
        test_leader_with_pressure()
    elif choice == "3":
        test_threshold_tuning()
    else:
        print("無効な選択です")


if __name__ == "__main__":
    main()