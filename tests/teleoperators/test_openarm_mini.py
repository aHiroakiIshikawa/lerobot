#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import MagicMock, patch

from lerobot.motors import MotorNormMode
from lerobot.teleoperators.openarm_mini import OpenArmMini, OpenArmMiniConfig


def _make_teleop(tmp_path, *, yam_6dof: bool = False, side: str | None = None):
    bus = MagicMock(name="FeetechMotorsBus")
    bus.is_connected = True

    def make_bus(*_args, **kwargs):
        bus.motors = kwargs["motors"]
        return bus

    with patch(
        "lerobot.teleoperators.openarm_mini.openarm_mini.FeetechMotorsBus",
        side_effect=make_bus,
    ):
        teleop = OpenArmMini(
            OpenArmMiniConfig(
                id="openarm_mini_test",
                calibration_dir=tmp_path,
                port="/dev/null",
                side=side,
                yam_6dof=yam_6dof,
            )
        )

    return teleop, bus


def test_yam_6dof_skips_motor_id_3_and_uses_yam_features(tmp_path):
    teleop, bus = _make_teleop(tmp_path, yam_6dof=True)

    expected_motor_ids = {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_flex": 4,
        "wrist_flex": 6,
        "wrist_roll": 7,
        "wrist_yaw": 5,
        "gripper": 8,
    }
    assert {name: motor.id for name, motor in bus.motors.items()} == expected_motor_ids
    assert 3 not in expected_motor_ids.values()
    assert all(
        motor.norm_mode is MotorNormMode.RANGE_M100_100
        for name, motor in bus.motors.items()
        if name != "gripper"
    )
    assert bus.motors["gripper"].norm_mode is MotorNormMode.RANGE_0_100
    assert set(teleop.action_features) == {f"{name}.pos" for name in expected_motor_ids}

    positions = {name: float(index) for index, name in enumerate(expected_motor_ids, start=1)}
    bus.sync_read.return_value = positions

    action = teleop.get_action()

    assert action == {f"{name}.pos": value for name, value in positions.items()}
    teleop.send_feedback(action)
    bus.sync_write.assert_called_once_with("Goal_Position", positions)


def test_yam_6dof_applies_side_flips_by_physical_motor(tmp_path):
    teleop, bus = _make_teleop(tmp_path, yam_6dof=True, side="right")
    positions = {name: float(index) for index, name in enumerate(bus.motors, start=1)}
    bus.sync_read.return_value = positions

    action = teleop.get_action()

    assert action == {
        "shoulder_pan.pos": -1.0,
        "shoulder_lift.pos": -2.0,
        "elbow_flex.pos": -3.0,
        "wrist_flex.pos": 4.0,
        "wrist_roll.pos": -5.0,
        "wrist_yaw.pos": -6.0,
        "gripper.pos": 7.0,
    }
    teleop.send_feedback(action)
    bus.sync_write.assert_called_once_with("Goal_Position", positions)


def test_yam_6dof_applies_left_side_flips(tmp_path):
    teleop, bus = _make_teleop(tmp_path, yam_6dof=True, side="left")
    positions = {name: float(index) for index, name in enumerate(bus.motors, start=1)}
    bus.sync_read.return_value = positions

    action = teleop.get_action()

    assert action == {
        "shoulder_pan.pos": -1.0,
        "shoulder_lift.pos": 2.0,
        "elbow_flex.pos": -3.0,
        "wrist_flex.pos": -4.0,
        "wrist_roll.pos": -5.0,
        "wrist_yaw.pos": -6.0,
        "gripper.pos": 7.0,
    }
    teleop.send_feedback(action)
    bus.sync_write.assert_called_once_with("Goal_Position", positions)


def test_default_mode_keeps_seven_dof_remap_and_gripper_conversion(tmp_path):
    teleop, bus = _make_teleop(tmp_path)
    assert [motor.id for motor in bus.motors.values()] == list(range(1, 9))
    assert set(teleop.action_features) == {
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "joint_4.pos",
        "joint_5.pos",
        "joint_6.pos",
        "joint_7.pos",
        "gripper.pos",
    }

    bus.sync_read.return_value = {
        "joint_1": 1.0,
        "joint_2": 2.0,
        "joint_3": 3.0,
        "joint_4": 4.0,
        "joint_5": 5.0,
        "joint_6": 6.0,
        "joint_7": 7.0,
        "gripper": 100.0,
    }

    action = teleop.get_action()

    assert action["joint_6.pos"] == 7.0
    assert action["joint_7.pos"] == 6.0
    assert action["gripper.pos"] == -65.0
