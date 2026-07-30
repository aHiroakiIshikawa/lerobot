"""Numerical contract tests for the Windows-compatible IKPy backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ikpy", reason="IKPy is installed by the kinematics extra on Windows")

from examples.so101_middle_position_handoff.ikpy_kinematics import (  # noqa: E402
    IKPyRobotKinematics,
)

SO101_POSITION_CHAIN_URDF = """<?xml version="1.0"?>
<robot name="so101_position_test">
  <link name="base_link"/>
  <link name="shoulder_link"/>
  <link name="upper_arm_link"/>
  <link name="lower_arm_link"/>
  <link name="wrist_link"/>
  <link name="gripper_link"/>
  <joint name="shoulder_pan" type="revolute">
    <parent link="base_link"/><child link="shoulder_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="shoulder_lift" type="revolute">
    <parent link="shoulder_link"/><child link="upper_arm_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="elbow_flex" type="revolute">
    <parent link="upper_arm_link"/><child link="lower_arm_link"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="wrist_flex" type="revolute">
    <parent link="lower_arm_link"/><child link="wrist_link"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="wrist_roll" type="revolute">
    <parent link="wrist_link"/><child link="gripper_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>
"""


@pytest.fixture
def kinematics(tmp_path: Path) -> IKPyRobotKinematics:
    urdf_path = tmp_path / "so101.urdf"
    urdf_path.write_text(SO101_POSITION_CHAIN_URDF)
    return IKPyRobotKinematics(
        str(urdf_path),
        target_frame_name="wrist_link",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex"],
    )


def test_forward_kinematics_targets_wrist_origin(kinematics: IKPyRobotKinematics) -> None:
    pose = kinematics.forward_kinematics(np.zeros(3))

    np.testing.assert_allclose(pose[:3, 3], [0.2, 0.0, 0.1], atol=1e-9)


def test_inverse_kinematics_tracks_position_and_preserves_other_joints(
    kinematics: IKPyRobotKinematics,
) -> None:
    current = np.array([0.0, 0.0, 0.0, 12.0, -8.0, 60.0])
    known_solution = np.array([20.0, -35.0, 65.0])
    target = kinematics.forward_kinematics(known_solution)

    solved = kinematics.inverse_kinematics(current, target, orientation_weight=0.0)

    solved_pose = kinematics.forward_kinematics(solved)
    np.testing.assert_allclose(solved_pose[:3, 3], target[:3, 3], atol=1e-5)
    np.testing.assert_array_equal(solved[3:], current[3:])


def test_orientation_constraint_is_rejected(kinematics: IKPyRobotKinematics) -> None:
    with pytest.raises(ValueError, match="position-only"):
        kinematics.inverse_kinematics(
            np.zeros(3),
            np.eye(4),
            orientation_weight=0.01,
        )
