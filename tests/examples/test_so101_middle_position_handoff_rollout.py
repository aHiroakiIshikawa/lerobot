"""Integration-oriented tests for the SO-101 handoff rollout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("datasets", reason="rollout requires the dataset extra")

from examples.so101_middle_position_handoff import rollout as handoff  # noqa: E402

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _settings(**overrides):
    values = {
        "middle_positions": {"shoulder_pan.pos": 0.0},
        "entry_tolerance_deg": 10.0,
        "exit_tolerance_deg": 15.0,
        "dwell_time_s": 1.0,
        "urdf_path": "so101.urdf",
        "eef_step_m": 0.002,
        "max_ee_step_m": 0.03,
        "max_joint_step_deg": 15.0,
        "gripper_step_per_tick": 2.0,
        "resume_blend_s": 0.5,
        "fps": 30.0,
        "return_to_initial_position": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _setup_context():
    inner = SimpleNamespace(bus=SimpleNamespace(motors=dict.fromkeys(MOTOR_NAMES)))
    return SimpleNamespace(
        hardware=SimpleNamespace(robot_wrapper=SimpleNamespace(inner=inner)),
    )


def test_cli_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "examples.so101_middle_position_handoff.rollout", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--middle_positions" in result.stdout
    assert "--max_joint_step_deg" in result.stdout


def test_config_rejects_unavailable_requested_accelerator_before_policy_load() -> None:
    cfg = SimpleNamespace(device="cuda")

    with (
        patch.object(handoff, "is_torch_device_available", return_value=False),
        patch.object(handoff.RolloutConfig, "__post_init__") as parent_post_init,
        pytest.raises(RuntimeError, match="Refusing to run.*CPU fallback"),
    ):
        handoff.MiddlePositionHandoffConfig.__post_init__(cfg)

    parent_post_init.assert_not_called()


def test_cli_policy_path_loads_before_hardware(monkeypatch) -> None:
    fake_policy = SimpleNamespace(device="cpu", pretrained_path=None)
    fake_context = object()
    fake_strategy = MagicMock()
    signal_handler = SimpleNamespace(shutdown_event=Event())
    keyboard = MagicMock()
    keyboard.is_available = True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "so101-handoff",
            "--robot.type=so101_follower",
            "--robot.port=/dev/null",
            "--robot.cameras={top: {type: opencv, index_or_path: 0, width: 640, "
            "height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, "
            "index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}}",
            "--policy.path=test/policy",
            '--middle_positions={"shoulder_pan.pos": 0.0}',
        ],
    )

    with (
        patch(
            "lerobot.rollout.configs.PreTrainedConfig.from_pretrained",
            return_value=fake_policy,
        ) as load_policy,
        patch.object(handoff, "init_logging"),
        patch.object(handoff, "KeyboardEEFController", return_value=keyboard),
        patch.object(handoff, "ProcessSignalHandler", return_value=signal_handler),
        patch.object(handoff, "build_rollout_context", return_value=fake_context) as build_context,
        patch.object(
            handoff,
            "MiddlePositionHandoffStrategy",
            return_value=fake_strategy,
        ),
    ):
        handoff.main()

    load_policy.assert_called_once()
    cfg = build_context.call_args.args[0]
    assert isinstance(cfg, handoff.MiddlePositionHandoffConfig)
    assert cfg.policy is fake_policy
    assert cfg.policy.pretrained_path == "test/policy"
    assert cfg.middle_positions == {"shoulder_pan.pos": 0.0}
    assert set(cfg.robot.cameras) == {"top", "wrist"}
    assert cfg.robot.cameras["top"].index_or_path == 0
    assert cfg.robot.cameras["wrist"].index_or_path == 1
    assert cfg.robot.cameras["top"].fourcc == "MJPG"
    assert cfg.robot.max_relative_target == cfg.max_joint_step_deg
    fake_strategy.setup.assert_called_once_with(fake_context)
    fake_strategy.run.assert_called_once_with(fake_context)
    fake_strategy.teardown.assert_called_once_with(fake_context)


def test_main_rejects_missing_keyboard_before_building_context() -> None:
    keyboard = MagicMock()
    keyboard.is_available = False

    with (
        patch.object(handoff, "init_logging"),
        patch.object(handoff, "KeyboardEEFController", return_value=keyboard),
        patch.object(handoff, "build_rollout_context") as build_context,
        pytest.raises(RuntimeError, match="key-release capture is unavailable"),
    ):
        handoff.main.__wrapped__(SimpleNamespace())

    build_context.assert_not_called()
    keyboard.stop.assert_called_once_with()


def test_setup_fails_before_engine_when_keyboard_cannot_capture() -> None:
    keyboard = MagicMock()
    keyboard.is_available = False
    strategy = handoff.MiddlePositionHandoffStrategy(
        _settings(),
        keyboard_controller=keyboard,
    )

    with (
        patch.object(strategy, "_init_engine") as init_engine,
        pytest.raises(
            RuntimeError,
            match="working pynput listener",
        ),
    ):
        strategy.setup(SimpleNamespace())

    keyboard.start.assert_called_once_with()
    init_engine.assert_not_called()


def test_setup_uses_three_joint_wrist_kinematics() -> None:
    keyboard = MagicMock()
    keyboard.is_available = True
    strategy = handoff.MiddlePositionHandoffStrategy(
        _settings(),
        keyboard_controller=keyboard,
    )

    with (
        patch.object(strategy, "_init_engine"),
        patch.object(handoff, "make_handoff_kinematics") as make_kinematics,
    ):
        strategy.setup(_setup_context())

    make_kinematics.assert_called_once_with("so101.urdf")
    assert strategy._eef_pipeline.steps[1].motor_names == list(handoff._IK_JOINT_NAMES)
    assert strategy._eef_pipeline.steps[-1].motor_names == list(MOTOR_NAMES)


def test_make_handoff_kinematics_uses_ikpy_on_windows() -> None:
    with (
        patch.object(handoff.platform, "system", return_value="Windows"),
        patch(
            "examples.so101_middle_position_handoff.ikpy_kinematics.IKPyRobotKinematics"
        ) as ikpy_kinematics,
    ):
        handoff.make_handoff_kinematics("so101.urdf")

    assert ikpy_kinematics.call_args.kwargs == {
        "urdf_path": "so101.urdf",
        "target_frame_name": "wrist_link",
        "joint_names": list(handoff._IK_JOINT_NAMES),
    }


def test_make_handoff_kinematics_uses_placo_off_windows() -> None:
    with (
        patch.object(handoff.platform, "system", return_value="Linux"),
        patch.object(handoff, "RobotKinematics") as placo_kinematics,
    ):
        handoff.make_handoff_kinematics("so101.urdf")

    assert placo_kinematics.call_args.kwargs == {
        "urdf_path": "so101.urdf",
        "target_frame_name": "wrist_link",
        "joint_names": list(handoff._IK_JOINT_NAMES),
    }


def test_limit_joint_step_clamps_without_mutating_input() -> None:
    action = {
        "shoulder_pan.pos": 100.0,
        "wrist_flex.pos": -50.0,
        "gripper.pos": 22.0,
        "metadata": 7.0,
    }
    observation = {
        "shoulder_pan.pos": 0.0,
        "wrist_flex.pos": -10.0,
        "gripper.pos": 20.0,
    }

    limited = handoff.limit_joint_step(action, observation, max_step_deg=15.0)

    assert limited == {
        "shoulder_pan.pos": 15.0,
        "wrist_flex.pos": -25.0,
        "gripper.pos": 22.0,
        "metadata": 7.0,
    }
    assert action["shoulder_pan.pos"] == 100.0


def test_manual_mode_pauses_engine_before_waiting_for_resume() -> None:
    resume_requested = MagicMock()
    resume_requested.wait.return_value = True
    keyboard = SimpleNamespace(
        is_available=True,
        resume_requested=resume_requested,
        shutdown_requested=Event(),
    )
    strategy = handoff.MiddlePositionHandoffStrategy(
        _settings(),
        keyboard_controller=keyboard,
    )
    strategy._engine = MagicMock()
    strategy._eef_pipeline = SimpleNamespace(steps=[])
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(shutdown_event=Event()),
        hardware=SimpleNamespace(robot_wrapper=MagicMock()),
    )

    strategy._hold_mode(ctx)

    strategy._engine.pause.assert_called_once_with()
    assert resume_requested.clear.call_count == 2


def test_resume_resets_then_resumes_before_blend() -> None:
    events = []
    strategy = handoff.MiddlePositionHandoffStrategy(_settings())
    strategy._engine = MagicMock()
    strategy._engine.reset.side_effect = lambda: events.append("engine.reset")
    strategy._engine.resume.side_effect = lambda: events.append("engine.resume")
    strategy._interpolator = MagicMock()
    strategy._interpolator.reset.side_effect = lambda: events.append("interpolator.reset")
    strategy._resume_with_blend = MagicMock(side_effect=lambda *_args: events.append("blend"))

    strategy._resume_autonomous(SimpleNamespace(), {"shoulder_pan.pos": 0.0})

    assert events == ["engine.reset", "interpolator.reset", "engine.resume", "blend"]
