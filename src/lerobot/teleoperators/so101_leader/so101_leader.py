#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_so101_leader import SO101LeaderConfig
from .pressure_sensor import PressureSensor

logger = logging.getLogger(__name__)


class SO101Leader(Teleoperator):
    """
    SO-101 Leader Arm designed by TheRobotStudio and Hugging Face.
    """
    
    config_class = SO101LeaderConfig
    name = "so101_leader"
    
    def __init__(self, config: SO101LeaderConfig):
        super().__init__(config)
        self.config = config
        self.gripper_limited_position = None
        self._gripper_torque_enabled = False
        
        # use_degreesに基づいてnorm_modeを選択
        if config.use_degrees:
            norm_mode_body = MotorNormMode.DEGREES
        else:
            norm_mode_body = MotorNormMode.RANGE_M100_100
        
        # 圧力センサの初期化
        self.pressure_sensor: PressureSensor | None = None
        if self.config.pressure_sensor_enabled:
            self.pressure_sensor = PressureSensor(
                port=self.config.pressure_sensor_port,
                baudrate=self.config.pressure_sensor_baudrate
            )
        
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        
        # 圧力センサを接続
        if self.pressure_sensor:
            try:
                self.pressure_sensor.connect(max_retries=3, retry_delay=1.0)
                logger.info("Pressure sensor connected successfully")
            except Exception as e:
                logger.warning(f"Could not connect pressure sensor: {e}")
                logger.warning("Continuing without pressure limiting feature")
                self.pressure_sensor = None
        
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move all joints to the middle of their range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        # 全モーターのトルクを無効化（手で動かせるように）
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        
        # グリッパのPID設定（制限時用）
        self.bus.write("P_Coefficient", "gripper", 32)
        self.bus.write("I_Coefficient", "gripper", 0)
        self.bus.write("D_Coefficient", "gripper", 32)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def _enable_gripper_torque_soft(self) -> bool:
        """グリッパのトルクをソフトスタートで有効化"""
        try:
            # 現在位置を取得
            current_pos = self.bus.sync_read("Present_Position")["gripper"]
            
            # まずGoal_Positionを現在位置に設定（急激な動きを防ぐ）
            self.bus.write("Goal_Position", "gripper", current_pos)
            
            # 少し待機
            time.sleep(0.01)
            
            # トルクを有効化
            self.bus.enable_torque("gripper")
            self._gripper_torque_enabled = True
            
            return True
        except RuntimeError as e:
            logger.warning(f"Failed to enable gripper torque: {e}")
            return False

    def _disable_gripper_torque_safe(self) -> bool:
        """グリッパのトルクを安全に無効化"""
        try:
            if self._gripper_torque_enabled:
                self.bus.disable_torque("gripper")
                self._gripper_torque_enabled = False
            return True
        except RuntimeError as e:
            logger.warning(f"Failed to disable gripper torque: {e}")
            self._gripper_torque_enabled = False
            return False

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        
        # 圧力制限を適用
        if self.pressure_sensor:
            pressure = self.pressure_sensor.get_force()
            current_gripper_pos = action["gripper"]
            
            if pressure > self.config.pressure_threshold:
                # 圧力が閾値を超えた場合
                if self.gripper_limited_position is None:
                    self.gripper_limited_position = current_gripper_pos
                    logger.info(f"Gripper limited at {current_gripper_pos:.2f} (pressure: {pressure:.3f}N)")
                
                # 閉じる方向を制限（現在位置が制限位置より小さい＝閉じようとしている）
                if current_gripper_pos < self.gripper_limited_position:
                    # 閉じようとしている場合：トルクを有効にして制限
                    if not self._gripper_torque_enabled:
                        self._enable_gripper_torque_soft()
                    
                    # トルクが有効な場合のみGoal_Positionを書き込む
                    if self._gripper_torque_enabled:
                        try:
                            self.bus.write("Goal_Position", "gripper", self.gripper_limited_position)
                        except RuntimeError as e:
                            logger.warning(f"Failed to write Goal_Position: {e}")
                            # エラー時はトルクを無効化してリセット
                            self._disable_gripper_torque_safe()
                    
                    action["gripper"] = self.gripper_limited_position
                else:
                    # 開く方向：まずトルクを無効化してから制限位置を更新
                    self._disable_gripper_torque_safe()
                    self.gripper_limited_position = current_gripper_pos
            else:
                # 圧力が閾値以下：制限解除
                if self.gripper_limited_position is not None:
                    logger.info(f"Gripper limit released (pressure: {pressure:.3f}N)")
                    self.gripper_limited_position = None
                
                self._disable_gripper_torque_safe()
        
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def read_pressure_sensor(self) -> float:
        """圧力値を取得 [N]"""
        if self.pressure_sensor and self.pressure_sensor.is_connected:
            return self.pressure_sensor.get_force()
        return 0.0

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # グリッパのトルクを安全に無効化
        self._disable_gripper_torque_safe()
        
        if self.pressure_sensor:
            self.pressure_sensor.disconnect()
        
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
