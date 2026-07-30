"""IKPy adapter matching LeRobot's position-only kinematics interface."""

from __future__ import annotations

import warnings

import numpy as np
from defusedxml import ElementTree


class IKPyRobotKinematics:
    """Pure-Python FK/IK backend for platforms without Placo wheels.

    The chain is truncated at ``target_frame_name`` and only ``joint_names``
    are active. Other joints on the path remain fixed at zero, which is valid
    for the SO-101 ``wrist_link`` position target because ``wrist_flex`` rotates
    the wrist frame around its own origin without changing that origin's XYZ.
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ) -> None:
        from ikpy.chain import Chain

        if not joint_names:
            raise ValueError("IKPyRobotKinematics requires explicit joint_names")

        target_joint_name = self._find_parent_joint(urdf_path, target_frame_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            full_chain = Chain.from_urdf_file(urdf_path, base_elements=["base_link"])

        target_index = next(
            (index for index, link in enumerate(full_chain.links) if link.name == target_joint_name),
            None,
        )
        if target_index is None:
            raise ValueError(
                f"Could not find parent joint '{target_joint_name}' for frame "
                f"'{target_frame_name}' in IKPy chain"
            )

        links = full_chain.links[: target_index + 1]
        link_names = [link.name for link in links]
        missing = [name for name in joint_names if name not in link_names]
        if missing:
            raise ValueError(f"IK joints are not on the chain to {target_frame_name}: {missing}")

        active_links_mask = [link.name in joint_names for link in links]
        self.chain = Chain(
            links=links,
            active_links_mask=active_links_mask,
            name=f"{target_frame_name}_position_chain",
        )
        self.joint_names = list(joint_names)
        self._joint_indices = [link_names.index(name) for name in self.joint_names]

    @staticmethod
    def _find_parent_joint(urdf_path: str, target_frame_name: str) -> str:
        root = ElementTree.parse(urdf_path).getroot()
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None and child.attrib.get("link") == target_frame_name:
                return joint.attrib["name"]
        raise ValueError(f"Frame '{target_frame_name}' is not the child of any URDF joint")

    def _chain_position(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        if len(joint_pos_deg) < len(self.joint_names):
            raise ValueError(f"Expected at least {len(self.joint_names)} joints, got {len(joint_pos_deg)}")
        position = np.zeros(len(self.chain.links), dtype=float)
        for source_index, chain_index in enumerate(self._joint_indices):
            position[chain_index] = np.deg2rad(float(joint_pos_deg[source_index]))
        return position

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Return the target frame transform for joint positions in degrees."""
        return np.asarray(self.chain.forward_kinematics(self._chain_position(joint_pos_deg)))

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """Solve position-only IK and preserve joints outside ``joint_names``."""
        del position_weight
        if orientation_weight != 0.0:
            raise ValueError(
                "IKPyRobotKinematics currently supports position-only IK (orientation_weight must be 0.0)"
            )

        initial_position = self._chain_position(current_joint_pos)
        solution = self.chain.inverse_kinematics(
            target_position=np.asarray(desired_ee_pose, dtype=float)[:3, 3],
            initial_position=initial_position,
        )
        solved_deg = np.rad2deg(np.asarray(solution)[self._joint_indices])

        result = np.asarray(current_joint_pos, dtype=float).copy()
        result[: len(self.joint_names)] = solved_deg
        return result
