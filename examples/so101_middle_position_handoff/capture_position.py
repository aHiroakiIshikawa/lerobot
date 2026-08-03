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

"""Capture the SO-101 joint angles needed for ``rollout.py``'s ``--middle_positions``.

Connects to the follower arm, disables torque so it can be moved freely by
hand, then prints the current joint degrees — plus a ready-to-paste
``--middle_positions`` JSON snippet — every time you press ENTER. This is the
"small capture option" the handoff plan calls for, so the physical middle
pose is measured rather than guessed.

Usage::

    uv run python -m examples.so101_middle_position_handoff.capture_position \\
        --robot.type=so101_follower \\
        --robot.port=/dev/tty.usbmodemXXXX \\
        --robot.id=my_follower \\
        --robot.use_degrees=true

Move the arm by hand into the desired middle position, press ENTER to read
it, repeat for as many candidate poses as you like, then Ctrl+C to exit.
``gripper.pos`` is excluded from the printed snippet since the handoff
strategy warns against including it in ``middle_positions``.
"""

import logging
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

# Gripper motion is unrelated to arm pose and would interfere with dwell detection.
_EXCLUDED_KEYS = {"gripper.pos"}


@dataclass
class CapturePositionConfig:
    robot: RobotConfig


@parser.wrap()
def capture_position(cfg: CapturePositionConfig) -> None:
    """Connect, free the arm, and print joint angles on every ENTER press."""
    init_logging()
    robot = make_robot_from_config(cfg.robot)
    robot.connect(calibrate=True)
    robot.bus.disable_torque()
    print(
        "\nTorque disabled — move the arm by hand into the target middle position.\n"
        "Press ENTER to read it (repeatable), Ctrl+C to exit.\n"
    )

    try:
        while True:
            input("Press ENTER to read position...")
            obs = robot.get_observation()
            positions = {
                k: round(v, 1) for k, v in obs.items() if k.endswith(".pos") and k not in _EXCLUDED_KEYS
            }
            print("Current joint positions (deg):")
            for k, v in positions.items():
                print(f"  {k}: {v}")
            snippet = ", ".join(f'"{k}": {v}' for k, v in positions.items())
            print(f"\n--middle_positions='{{{snippet}}}'\n")
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        robot.disconnect()


def main():
    capture_position()


if __name__ == "__main__":
    main()
