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
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_openarm_mini import OpenArmMiniConfig

logger = logging.getLogger(__name__)

# Per-side motor direction flips applied during readout.
SIDE_MOTORS_TO_FLIP: dict[str, list[str]] = {
    "left": ["joint_1", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
    "right": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_7"],
}

# Leader joint 6 ↔ follower joint 7 (symmetric — its own inverse).
JOINT_REMAP = {"joint_6": "joint_7", "joint_7": "joint_6"}

GRIPPER_TELEOP_TO_DEGREES = -0.65
GRIPPER_MOTOR_ID = 8
YAM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_yaw",
)
YAM_JOINT_MOTOR_IDS = (1, 2, 4, 6, 7, 5)


class OpenArmMini(Teleoperator):
    """OpenArm Mini single-arm teleoperator (Feetech STS3215, 7DOF + gripper).

    For the bimanual setup, see :class:`BiOpenArmMini` which composes two of these.
    """

    config_class = OpenArmMiniConfig
    name = "openarm_mini"

    def __init__(self, config: OpenArmMiniConfig):
        super().__init__(config)
        self.config = config

        if config.side is not None and config.side not in SIDE_MOTORS_TO_FLIP:
            raise ValueError(f"Invalid side '{config.side}'; expected 'left', 'right', or None.")

        if config.yam_6dof:
            logical_motor_names = YAM_JOINT_NAMES
            active_joint_ids = YAM_JOINT_MOTOR_IDS
            norm_mode_body = MotorNormMode.RANGE_M100_100
        else:
            logical_motor_names = tuple(f"joint_{motor_id}" for motor_id in range(1, 8))
            active_joint_ids = tuple(range(1, 8))
            norm_mode_body = MotorNormMode.DEGREES

        physical_motor_names = {
            logical_name: f"joint_{motor_id}"
            for logical_name, motor_id in zip(logical_motor_names, active_joint_ids, strict=True)
        }
        motors = {
            logical_name: Motor(motor_id, "sts3215", norm_mode_body)
            for logical_name, motor_id in zip(physical_motor_names, active_joint_ids, strict=True)
        }
        motors["gripper"] = Motor(GRIPPER_MOTOR_ID, "sts3215", MotorNormMode.RANGE_0_100)

        physical_motors_to_flip = set(SIDE_MOTORS_TO_FLIP.get(config.side, []))
        self._motors_to_flip = {
            logical_name
            for logical_name, physical_name in physical_motor_names.items()
            if physical_name in physical_motors_to_flip
        }
        self._joint_remap = {} if config.yam_6dof else JOINT_REMAP

        if self.calibration:
            expected_calibration_ids = {name: motor.id for name, motor in motors.items()}
            loaded_calibration_ids = {name: calibration.id for name, calibration in self.calibration.items()}
            if loaded_calibration_ids != expected_calibration_ids:
                raise ValueError(
                    f"Calibration for teleoperator id '{self.id}' does not match "
                    f"yam_6dof={config.yam_6dof}. Use a different --teleop.id and calibrate this layout."
                )

        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        if calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for a single OpenArm Mini arm.

        1. Disable torque
        2. Ask user to position arm in hanging position with gripper closed
        3. Set this as zero position via half-turn homing
        4. Interactive gripper calibration (open/close positions)
        5. Save calibration
        """
        if self.calibration:
            user_input = input(
                f"Press ENTER to use existing calibration for {self.id}, "
                f"or type 'c' and press ENTER to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration for {self.id}")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")

        self.bus.disable_torque()

        logger.info("Setting Phase to 12 for all motors...")
        for motor in self.bus.motors:
            self.bus.write("Phase", motor, 12)

        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(
            "\nCalibration: Zero Position\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )

        homing_offsets = self.bus.set_half_turn_homings()
        logger.info("Arm zero position set.")

        print("\nSetting motor ranges\n")

        if self.calibration is None:
            self.calibration = {}

        range_mins: dict[str, int | float] = {}
        range_maxes: dict[str, int | float] = {}
        if self.config.yam_6dof:
            body_motors = [motor for motor in self.bus.motors if motor != "gripper"]
            print(
                "Move all six arm joints through their full usable ranges. "
                "Recording positions; press ENTER to stop..."
            )
            range_mins, range_maxes = self.bus.record_ranges_of_motion(body_motors)

        motor_resolution = self.bus.model_resolution_table[list(self.bus.motors.values())[0].model]
        max_res = motor_resolution - 1

        for motor_name, motor in self.bus.motors.items():
            if motor_name == "gripper":
                input(
                    "\nGripper Calibration\n"
                    "Step 1: CLOSE the gripper fully\n"
                    "Press ENTER when gripper is closed..."
                )
                closed_pos = self.bus.read("Present_Position", motor_name, normalize=False)
                logger.info(f"  Gripper closed position recorded: {closed_pos}")

                input("\nStep 2: OPEN the gripper fully\nPress ENTER when gripper is fully open...")
                open_pos = self.bus.read("Present_Position", motor_name, normalize=False)
                logger.info(f"  Gripper open position recorded: {open_pos}")

                if closed_pos < open_pos:
                    range_min = int(closed_pos)
                    range_max = int(open_pos)
                    drive_mode = 0
                else:
                    range_min = int(open_pos)
                    range_max = int(closed_pos)
                    drive_mode = 1

                logger.info(
                    f"  {motor_name}: range set to [{range_min}, {range_max}] "
                    f"(0=closed, 100=open, drive_mode={drive_mode})"
                )
            elif self.config.yam_6dof:
                range_min = int(range_mins[motor_name])
                range_max = int(range_maxes[motor_name])
                drive_mode = 0
                logger.info(f"  {motor_name}: range set to [{range_min}, {range_max}]")
            else:
                range_min = 0
                range_max = max_res
                drive_mode = 0
                logger.info(f"  {motor_name}: range set to [0, {max_res}] (full motor range)")

            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=drive_mode,
                homing_offset=homing_offsets[motor_name],
                range_min=range_min,
                range_max=range_max,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"\nCalibration complete and saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Get current action (read positions from all motors)."""
        start = time.perf_counter()

        positions = self.bus.sync_read("Present_Position")

        # Joint 6↔7 remap: leader joint_6 → follower joint_7 and vice versa.
        # Per-side direction flip is applied based on the configured `side`.
        action: dict[str, Any] = {}
        for motor, val in positions.items():
            target = self._joint_remap.get(motor, motor)
            if motor == "gripper":
                action[f"{target}.pos"] = val if self.config.yam_6dof else val * GRIPPER_TELEOP_TO_DEGREES
            else:
                action[f"{target}.pos"] = -val if motor in self._motors_to_flip else val

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def enable_torque(self) -> None:
        self.bus.enable_torque()

    def disable_torque(self) -> None:
        self.bus.disable_torque()

    def write_goal_positions(self, positions: dict[str, float]) -> None:
        """Write goal positions to motors (inverse of get_action flip/gripper/remap logic)."""
        goals: dict[str, float] = {}
        for key, val in positions.items():
            if not key.endswith(".pos"):
                continue
            base = key.removesuffix(".pos")
            # The configured joint remap is symmetric (its own inverse).
            target = self._joint_remap.get(base, base)
            if base == "gripper":
                goals[target] = val if self.config.yam_6dof else val / GRIPPER_TELEOP_TO_DEGREES
            else:
                # Un-flip using the ORIGINAL motor name (target = leader motor)
                goals[target] = -val if target in self._motors_to_flip else val

        if goals:
            self.bus.sync_write("Goal_Position", goals)

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        self.write_goal_positions(feedback)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
