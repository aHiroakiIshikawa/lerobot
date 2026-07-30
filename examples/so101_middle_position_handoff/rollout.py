"""SO-101 middle-position keyboard EEF handoff rollout.

Autonomous policy execution with automatic handoff to keyboard EEF control:

1. **AUTONOMOUS** — policy inference drives the robot normally.
2. **MANUAL_EEF** — when all monitored joints dwell within ±entry_tolerance_deg
   of ``middle_positions`` for ``dwell_time_s`` seconds, policy inference is
   suspended and the operator takes over via arrow keys (end-effector Cartesian
   control).
3. **RESUMING** — when the operator presses Enter, a linear joint blend
   transitions from the manual end-pose to the first new policy action over
   ``resume_blend_s`` seconds, then autonomous inference resumes.

Usage::

    uv run python -m examples.so101_middle_position_handoff.rollout \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=my_follower \\
        --robot.cameras="..." \\
        --policy.path=<HF_repo_or_local_path> \\
        --task="<task description>" \\
        --middle_positions='{"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -30.0}' \\
        --urdf_path=./SO101/so101_new_calib.urdf

Keyboard controls (active only in MANUAL_EEF mode):
    Left / Right arrow   → EEF X (±)
    Up / Down arrow      → EEF Y (±)
    Shift_R              → EEF Z +
    Left Shift           → EEF Z -
    Ctrl_R               → gripper open
    Ctrl_L               → gripper close
    Enter                → resume policy inference
    ESC                  → stop rollout

Notes:
- Only ``SyncInferenceConfig`` (``--inference.type=sync``) is supported.
  RTC requires additional queue/thread handling not implemented here.
- Requires the ``hardware`` and ``kinematics`` extras (for example,
    ``uv sync --extra hardware --extra feetech --extra kinematics --extra dataset``).
    On macOS, grant *Accessibility* / *Input Monitoring*
  permission to the Python process.
- ``middle_positions`` is required and has no default.
- ``urdf_path`` must point to the SO-101 URDF for inverse kinematics.
"""

# Keep annotations eager in this module: parser.wrap() reads the raw cfg
# annotation and requires the dataclass type rather than a postponed string.
import logging
import platform
import time
from dataclasses import dataclass, field

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.processor.delta_action_processor import MapDeltaActionToRobotActionStep
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.rollout import RolloutConfig, build_rollout_context
from lerobot.rollout.configs import BaseStrategyConfig
from lerobot.rollout.context import RolloutContext
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.rollout.strategies.core import RolloutStrategy, send_next_action
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .dwell_detector import DwellDetector, DwellDetectorConfig
from .keyboard_eef import KeyboardEEFController

logger = logging.getLogger(__name__)

# EEF workspace bounds (wrist_link frame, metres)
_EEF_X_MIN, _EEF_X_MAX = 0.02, 0.25
_EEF_Y_MIN, _EEF_Y_MAX = -0.19, 0.17
_EEF_Z_MIN, _EEF_Z_MAX = 0.00, 0.35
_IK_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex")


def make_handoff_kinematics(urdf_path: str):
    """Select the native Placo backend or the pure-Python Windows backend."""
    if platform.system() == "Windows":
        from .ikpy_kinematics import IKPyRobotKinematics

        return IKPyRobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="wrist_link",
            joint_names=list(_IK_JOINT_NAMES),
        )
    return RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="wrist_link",
        joint_names=list(_IK_JOINT_NAMES),
    )


def limit_joint_step(
    action: RobotAction,
    observation: RobotObservation,
    max_step_deg: float,
) -> RobotAction:
    """Clamp each joint target around its latest measured position."""
    limited = action.copy()
    clipped = []
    for key, target in action.items():
        if not key.endswith(".pos") or key not in observation:
            continue
        current = float(observation[key])
        target = float(target)
        delta = max(-max_step_deg, min(max_step_deg, target - current))
        limited[key] = current + delta
        if limited[key] != target:
            clipped.append(key)
    if clipped:
        logger.warning(
            "Clamped joint targets to %.1f deg/tick: %s",
            max_step_deg,
            ", ".join(clipped),
        )
    return limited


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MiddlePositionHandoffConfig(RolloutConfig):
    """Configuration for the SO-101 middle-position keyboard EEF handoff rollout.

    Inheriting :class:`RolloutConfig` preserves standard ``--policy.path``
    loading, device resolution, and parser path-field handling.
    """

    # --- Dwell detection ---
    #: Joint-name → target angle (degrees).  All listed joints must stay
    #: within ``entry_tolerance_deg`` of these values for ``dwell_time_s``
    #: to trigger the handoff.  Required — no default.
    middle_positions: dict[str, float] = field(default_factory=dict)
    entry_tolerance_deg: float = 10.0
    exit_tolerance_deg: float = 15.0
    dwell_time_s: float = 1.0

    # --- IK / EEF control ---
    urdf_path: str = "./SO101/so101_new_calib.urdf"
    #: Distance (metres) the EEF moves per control tick while a key is held.
    eef_step_m: float = 0.002
    #: Maximum allowed EEF jump per tick (safety clamp, metres).
    max_ee_step_m: float = 0.03
    #: Maximum commanded change for any joint in one control tick.
    max_joint_step_deg: float = 15.0
    #: Gripper position change (degrees) per tick in discrete open/close mode.
    gripper_step_per_tick: float = 2.0

    # --- Blend on resume ---
    #: Duration (seconds) of the joint blend from end-of-manual to first
    #: policy action.
    resume_blend_s: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.inference, SyncInferenceConfig):
            raise ValueError(
                "MiddlePositionHandoffConfig only supports SyncInferenceConfig. "
                "Use SyncInferenceConfig (--inference.type=sync, the default)."
            )
        if not isinstance(self.strategy, BaseStrategyConfig):
            raise ValueError("Middle-position handoff requires --strategy.type=base")

        DwellDetectorConfig(
            middle_positions=self.middle_positions,
            entry_tolerance_deg=self.entry_tolerance_deg,
            exit_tolerance_deg=self.exit_tolerance_deg,
            dwell_time_s=self.dwell_time_s,
        )
        if self.eef_step_m <= 0:
            raise ValueError(f"eef_step_m must be positive, got {self.eef_step_m}")
        if self.max_ee_step_m <= 0:
            raise ValueError(f"max_ee_step_m must be positive, got {self.max_ee_step_m}")
        if self.max_joint_step_deg <= 0:
            raise ValueError(f"max_joint_step_deg must be positive, got {self.max_joint_step_deg}")
        if self.gripper_step_per_tick <= 0:
            raise ValueError(f"gripper_step_per_tick must be positive, got {self.gripper_step_per_tick}")
        if self.resume_blend_s < 0:
            raise ValueError(f"resume_blend_s must be non-negative, got {self.resume_blend_s}")
        if self.robot is not None and hasattr(self.robot, "use_degrees") and not self.robot.use_degrees:
            raise ValueError("Middle-position handoff requires --robot.use_degrees=true")
        if (
            self.robot is not None
            and hasattr(self.robot, "max_relative_target")
            and self.robot.max_relative_target is None
        ):
            self.robot.max_relative_target = self.max_joint_step_deg

        super().__post_init__()


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class MiddlePositionHandoffStrategy(RolloutStrategy):
    """Autonomous policy rollout with keyboard EEF manual-control handoff.

    Pass an instance of :class:`MiddlePositionHandoffConfig` to the
    constructor; the rollout ``RolloutContext`` is provided via ``setup()``
    as usual.
    """

    def __init__(
        self,
        handoff_cfg: MiddlePositionHandoffConfig,
        keyboard_controller: KeyboardEEFController | None = None,
    ) -> None:
        super().__init__(BaseStrategyConfig())
        self._handoff_cfg = handoff_cfg

        # Populated in setup()
        self._dwell: DwellDetector | None = None
        self._keyboard = keyboard_controller or KeyboardEEFController()
        self._eef_pipeline = None  # RobotProcessorPipeline

    # ------------------------------------------------------------------
    # RolloutStrategy interface
    # ------------------------------------------------------------------

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise inference engine, kinematics, EEF pipeline, and I/O devices."""
        if not self._keyboard.is_available:
            self._keyboard.start()
        if not self._keyboard.is_available:
            raise RuntimeError(
                "Keyboard EEF handoff requires a working pynput listener with key-release "
                "capture. Use X11/macOS/Windows, install the hardware extra, and grant "
                "desktop input permissions before starting the rollout."
            )

        self._init_engine(ctx)

        cfg = self._handoff_cfg

        if not cfg.middle_positions:
            raise ValueError(
                "MiddlePositionHandoffStrategy: 'middle_positions' is required but empty. "
                "Pass e.g. --middle_positions='{\"shoulder_pan.pos\": 0.0}'"
            )

        # --- Kinematics ---
        # Get motor names from the connected robot
        robot_inner = ctx.hardware.robot_wrapper.inner
        motor_names: list[str] = list(robot_inner.bus.motors.keys())
        missing_ik_joints = [name for name in _IK_JOINT_NAMES if name not in motor_names]
        if missing_ik_joints:
            raise ValueError(f"Robot is missing required IK joints: {missing_ik_joints}")

        available_position_keys = {f"{name}.pos" for name in motor_names}
        unknown_middle_keys = sorted(set(cfg.middle_positions) - available_position_keys)
        if unknown_middle_keys:
            raise ValueError(
                "middle_positions contains keys that are not robot joints: "
                f"{unknown_middle_keys}. Available keys: {sorted(available_position_keys)}"
            )

        kinematics = make_handoff_kinematics(cfg.urdf_path)

        # --- EEF pipeline (keyboard → joint actions) ---
        self._eef_pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
            steps=[
                MapDeltaActionToRobotActionStep(position_scale=cfg.eef_step_m),
                EEReferenceAndDelta(
                    kinematics=kinematics,
                    end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
                    motor_names=list(_IK_JOINT_NAMES),
                    use_latched_reference=False,
                ),
                EEBoundsAndSafety(
                    end_effector_bounds={
                        "min": [_EEF_X_MIN, _EEF_Y_MIN, _EEF_Z_MIN],
                        "max": [_EEF_X_MAX, _EEF_Y_MAX, _EEF_Z_MAX],
                    },
                    max_ee_step_m=cfg.max_ee_step_m,
                    raise_on_jump=False,
                ),
                GripperVelocityToJoint(
                    speed_factor=cfg.gripper_step_per_tick / 100.0,
                    discrete_gripper=True,
                ),
                InverseKinematicsEEToJoints(
                    kinematics=kinematics,
                    motor_names=motor_names,
                    orientation_weight=0.0,
                    initial_guess_current_joints=True,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )

        # --- Dwell detector ---
        self._dwell = DwellDetector(
            DwellDetectorConfig(
                middle_positions=cfg.middle_positions,
                entry_tolerance_deg=cfg.entry_tolerance_deg,
                exit_tolerance_deg=cfg.exit_tolerance_deg,
                dwell_time_s=cfg.dwell_time_s,
            )
        )

        logger.info("MiddlePositionHandoffStrategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Autonomous control loop with mid-episode EEF handoff."""
        engine = self._engine
        cfg = self._handoff_cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("MiddlePositionHandoffStrategy control loop started")

        while not ctx.runtime.shutdown_event.is_set():
            # Check keyboard shutdown
            if self._keyboard.shutdown_requested.is_set():
                logger.info("ESC pressed — stopping rollout")
                ctx.runtime.shutdown_event.set()
                break

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            loop_start = time.perf_counter()

            obs_raw = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs_raw)

            if self._handle_warmup(False, loop_start, control_interval):
                continue

            # --- Dwell check ---
            if self._dwell.update(obs_raw):
                logger.info("Dwell condition met — entering MANUAL EEF mode")
                self._hold_mode(ctx)

                if ctx.runtime.shutdown_event.is_set():
                    break

                # Capture final manual position, then reset + blend.
                final_obs = robot.get_observation()
                self._resume_autonomous(ctx, final_obs)

                # Continue to top of loop (fresh obs for next autonomous tick)
                continue

            # --- Autonomous step ---
            action_dict = send_next_action(obs_processed, obs_raw, ctx, interpolator)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    "Control loop running slower (%0.1f Hz) than target FPS (%0.1f Hz)",
                    1.0 / max(dt, 1e-6),
                    cfg.fps,
                )

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop keyboard listener and disconnect hardware."""
        if self._keyboard is not None:
            self._keyboard.stop()
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=self._handoff_cfg.return_to_initial_position,
        )
        logger.info("MiddlePositionHandoffStrategy teardown complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hold_mode(self, ctx: RolloutContext) -> None:
        """Run the manual EEF control loop until Enter or ESC is pressed."""
        cfg = self._handoff_cfg
        robot = ctx.hardware.robot_wrapper
        keyboard = self._keyboard
        control_interval = 1.0 / cfg.fps

        self._engine.pause()

        # Reset EEF pipeline state for a clean entry (no stale last-pos / IK guess)
        for step in self._eef_pipeline.steps:
            if hasattr(step, "reset"):
                step.reset()

        # Clear any stale resume_requested from a previous handoff
        keyboard.resume_requested.clear()

        print(
            "\n[HANDOFF] *** MANUAL EEF MODE ***\n"
            "  Arrow keys → EEF X/Y  |  Shift_R=Z+  Shift=Z-\n"
            "  Ctrl_R=gripper open   |  Ctrl_L=gripper close\n"
            "  Enter → resume policy | ESC → stop\n"
        )

        while not ctx.runtime.shutdown_event.is_set():
            # Non-blocking check for Enter
            if keyboard.resume_requested.wait(0.0):
                keyboard.resume_requested.clear()
                print("[HANDOFF] Resuming policy inference...\n")
                break

            if keyboard.shutdown_requested.is_set():
                ctx.runtime.shutdown_event.set()
                break

            if not keyboard.is_available:
                raise RuntimeError("Keyboard listener stopped during manual EEF control")

            t0 = time.perf_counter()

            key_action = keyboard.read_action()
            obs_raw = robot.get_observation()

            try:
                joint_action = self._eef_pipeline((key_action, obs_raw))
                joint_action = limit_joint_step(
                    joint_action,
                    obs_raw,
                    cfg.max_joint_step_deg,
                )
                robot.send_action(joint_action)
            except Exception as exc:
                logger.warning("EEF pipeline error (skipping frame): %s", exc)

            dt = time.perf_counter() - t0
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)

    def _resume_autonomous(self, ctx: RolloutContext, final_obs: dict) -> None:
        """Reset policy state and resume through a bounded joint-space blend."""
        self._engine.reset()
        self._interpolator.reset()
        self._cached_obs_processed = None
        self._engine.resume()
        self._resume_with_blend(ctx, final_obs)

    def _resume_with_blend(
        self,
        ctx: RolloutContext,
        final_obs: dict,
    ) -> None:
        """Blend from the last manual joint positions to the first policy action.

        Linearly interpolates from ``final_obs`` joint positions to the first
        action produced by the (freshly reset) policy over ``resume_blend_s``
        seconds, then primes the interpolator with that action so the
        autonomous loop picks up cleanly.

        If the policy returns ``None`` (shouldn't happen for SyncInferenceEngine),
        the blend is skipped and the autonomous loop will recover on its own.
        """
        cfg = self._handoff_cfg
        robot = ctx.hardware.robot_wrapper
        features = ctx.data.dataset_features
        ordered_keys = ctx.data.ordered_action_keys

        # Fresh obs for policy input
        obs_raw = robot.get_observation()
        obs_processed = ctx.processors.robot_observation_processor(obs_raw)
        self._cached_obs_processed = obs_processed
        self._engine.notify_observation(obs_processed)

        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
        action_tensor = self._engine.get_action(obs_frame)

        if action_tensor is None:
            logger.warning("Engine returned None on first action after resume — skipping blend")
            return

        # Prime the interpolator
        self._interpolator.add(action_tensor.cpu())

        first_action: dict[str, float] = {k: float(action_tensor[i]) for i, k in enumerate(ordered_keys)}

        # Latched manual end-positions (joint space).
        # Keys in final_obs are like "shoulder_pan.pos"; ordered_keys are the same
        # for direct joint-space policies.
        latched: dict[str, float] = {
            k: float(final_obs.get(k, first_action.get(k, 0.0))) for k in ordered_keys
        }

        n_steps = max(int(cfg.resume_blend_s * cfg.fps), 1)
        control_interval = self._interpolator.get_control_interval(cfg.fps)

        logger.info("Blending from manual pose to policy action over %d steps", n_steps)
        for step in range(n_steps):
            if ctx.runtime.shutdown_event.is_set():
                break
            t = (step + 1) / n_steps
            blended = {k: latched[k] * (1.0 - t) + first_action[k] * t for k in ordered_keys}
            obs_for_send = robot.get_observation()
            processed_action = ctx.processors.robot_action_processor((blended, obs_for_send))
            safe_action = limit_joint_step(
                processed_action,
                obs_for_send,
                cfg.max_joint_step_deg,
            )
            robot.send_action(safe_action)
            precise_sleep(control_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@parser.wrap()
def main(cfg: MiddlePositionHandoffConfig) -> None:
    """CLI entry point for the SO-101 middle-position keyboard EEF handoff rollout."""
    init_logging()

    keyboard = KeyboardEEFController()
    keyboard.start()
    if not keyboard.is_available:
        keyboard.stop()
        raise RuntimeError(
            "Keyboard EEF handoff cannot start because pynput key-release capture is unavailable."
        )

    signal_handler = ProcessSignalHandler(use_threads=True)
    strategy = MiddlePositionHandoffStrategy(cfg, keyboard_controller=keyboard)
    ctx = None
    try:
        ctx = build_rollout_context(cfg, signal_handler.shutdown_event)
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        if ctx is not None:
            strategy.teardown(ctx)
        else:
            keyboard.stop()


if __name__ == "__main__":
    main()
