"""SO-101 6-DOF leader gripper pressure feedback, with optional follower teleoperation."""

import argparse
import logging
import math
import re
import threading
import time
from dataclasses import dataclass

import serial

from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader


LOGGER = logging.getLogger(__name__)
FORCE_PATTERN = re.compile(r"\bF\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\[N\]")


def parse_force(line: str) -> float | None:
    match = FORCE_PATTERN.search(line)
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) and value >= 0 else None


class PressureSensor:
    def __init__(self, port: str, baudrate: int, max_age: float):
        self.port, self.baudrate, self.max_age = port, baudrate, max_age
        self.serial = None
        self.thread = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.sample = None
        self.error = None

    def connect(self):
        self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
        time.sleep(2)  # Arduino resets when opening the serial port.
        self.serial.reset_input_buffer()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        pending = bytearray()
        try:
            while not self.stop.is_set():
                chunk = self.serial.read_until(b"\n", 256)
                pending.extend(chunk)
                if len(pending) > 1024:
                    raise RuntimeError("圧力センサーの行が長すぎます。通信速度と出力形式を確認してください。")
                if not pending.endswith(b"\n"):
                    continue
                force = parse_force(pending.decode("ascii", errors="replace"))
                pending.clear()
                if force is not None:
                    with self.lock:
                        self.sample = (force, time.monotonic())
        except Exception as exc:
            with self.lock:
                self.error = exc

    def get_force(self):
        with self.lock:
            sample, error = self.sample, self.error
        if error is not None:
            raise RuntimeError(f"圧力センサー通信エラー: {error}") from error
        if sample is None or time.monotonic() - sample[1] > self.max_age:
            raise RuntimeError("新しい圧力値がありません。制御を停止します。センサーの接続・出力を確認してください。")
        return sample[0]

    def wait_ready(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                ready = self.sample is not None or self.error is not None
            if ready:
                return self.get_force()
            time.sleep(0.05)
        return self.get_force()

    def disconnect(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.serial is not None:
            self.serial.close()


@dataclass
class GripperLimit:
    threshold: float = 1.0
    release_threshold: float = 0.8
    deadband: float = 0.5  # normalized gripper position, 0..100
    close_direction: str = "decreasing"
    boundary: float | None = None
    holding: bool = False

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.threshold, self.release_threshold, self.deadband)):
            raise ValueError("閾値とdeadbandは有限値にしてください。")
        if not 0 <= self.release_threshold < self.threshold or not 0 <= self.deadband <= 100:
            raise ValueError("0 <= release-threshold < threshold、0 <= deadband <= 100 が必要です。")
        if self.close_direction not in ("decreasing", "increasing"):
            raise ValueError("Unknown closing direction")

    def update(self, position: float, force: float) -> tuple[float, float | None]:
        if not math.isfinite(position) or not math.isfinite(force) or force < 0:
            raise ValueError("関節位置または圧力値が不正です。")
        # Internally a larger value always means a more open gripper.
        sign = 1 if self.close_direction == "decreasing" else -1
        opened = sign * position
        if self.boundary is None and force >= self.threshold:
            self.boundary = opened
            self.holding = True  # Hold immediately on threshold crossing, before further closing.
        elif self.boundary is not None and force <= self.release_threshold:
            self.boundary = None
            self.holding = False
        if self.boundary is None:
            return position, None
        if opened > self.boundary + self.deadband:
            self.holding = False
        elif opened < self.boundary - self.deadband:
            self.holding = True
        # Ratchet the limit in the opening direction while pressure remains high.
        self.boundary = max(self.boundary, opened)
        command = sign * max(opened, self.boundary)
        goal = sign * self.boundary if self.holding else None
        return command, goal


class GripperFeedback:
    def __init__(self, bus):
        self.bus = bus
        self.enabled = False
        self.saved = {}
        self.last_goal = None

    def configure(self, torque_limit):
        self.bus.disable_torque("gripper")
        # Only the gripper holding torque limit is pressure-specific; preserve PID.
        for register, value in (("Torque_Limit", torque_limit),):
            self.saved[register] = self.bus.read(register, "gripper", normalize=False)
            self.bus.write(register, "gripper", value, normalize=False)

    def apply(self, current, goal):
        if goal is None:
            if self.enabled:
                self.bus.disable_torque("gripper")
                self.enabled = False
                self.last_goal = None
            return
        if not self.enabled:
            # Set the current position first to avoid enabling torque with an old goal.
            self.bus.write("Goal_Position", "gripper", current)
            self.bus.enable_torque("gripper")
            self.enabled = True
            self.last_goal = current
        if goal != self.last_goal:
            self.bus.write("Goal_Position", "gripper", goal)
            self.last_goal = goal

    def close(self):
        self.bus.disable_torque("gripper")
        self.enabled = False
        for register, value in self.saved.items():
            self.bus.write(register, "gripper", value, normalize=False)



def apply_pressure_action(action, force, limit, feedback):
    """Change only gripper.pos; preserve all six arm targets and the input mapping."""
    result = dict(action)
    current = action["gripper.pos"]
    previous = (limit.boundary is not None, limit.holding)
    result["gripper.pos"], goal = limit.update(current, force)
    feedback.apply(current, goal)
    if previous != (limit.boundary is not None, limit.holding):
        LOGGER.info("圧力保持切替: force=%.3f N gripper=%.2f goal=%s limited=%s holding=%s",
                    force, current, goal, limit.boundary is not None, limit.holding)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", default="/dev/tty.usbmodem5A7A0161151")
    parser.add_argument("--leader-id", default="so101_6dof_leader")
    parser.add_argument("--sensor-port", required=True)
    parser.add_argument("--sensor-baudrate", type=int, default=9600)
    parser.add_argument("--threshold", type=float, default=1.0, help="Force threshold in N")
    parser.add_argument("--release-threshold", type=float, default=0.8, help="Release threshold in N")
    parser.add_argument("--deadband", type=float, default=0.5, help="0..100 gripper position units")
    parser.add_argument("--close-direction", choices=("decreasing", "increasing"), default="decreasing")
    parser.add_argument("--torque-limit", type=int, default=100, help="STS3215 Torque_Limit, 1..1000")
    parser.add_argument("--sensor-timeout", type=float, default=0.5)
    parser.add_argument("--max-relative-target", type=float, default=None,
                        help="Optional follower goal error cap; omitted means standard teleop behavior")
    parser.add_argument("--feedback-mode", choices=("hold", "monitor"), default="hold",
                        help="hold: pressure resistance; monitor: log force without pressure motor control")
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--sensor-only", action="store_true", help="Print force; do not connect to either arm")
    parser.add_argument("--follower-port", help="Optional: also teleoperate the follower")
    parser.add_argument("--follower-id", default="so101_6dof_follower")
    args = parser.parse_args()
    if not 1 <= args.torque_limit <= 1000:
        parser.error("torque-limit must be in 1..1000")
    if not all(math.isfinite(v) and v > 0 for v in (args.sensor_timeout, args.fps)):
        parser.error("sensor-timeout and fps must be finite and positive")
    if args.max_relative_target is not None and (not math.isfinite(args.max_relative_target) or args.max_relative_target <= 0):
        parser.error("max-relative-target must be finite and positive when supplied")
    if args.sensor_baudrate <= 0:
        parser.error("sensor-baudrate must be positive")
    ports = [args.sensor_port] if args.sensor_only else [args.sensor_port, args.leader_port]
    if args.follower_port and not args.sensor_only:
        ports.append(args.follower_port)
    if len(set(ports)) != len(ports):
        parser.error("Sensor, leader and follower must use different ports")
    return args


def run(args):
    limit = GripperLimit(args.threshold, args.release_threshold, args.deadband, args.close_direction)
    sensor = PressureSensor(args.sensor_port, args.sensor_baudrate, args.sensor_timeout)
    leader = follower = feedback = None
    try:
        LOGGER.info("圧力センサー接続開始: %s (%d bps)。Arduinoのリセットとデータを待ちます。",
                    args.sensor_port, args.sensor_baudrate)
        sensor.connect()
        initial_force = sensor.wait_ready()
        LOGGER.info("圧力センサー受信開始: %.3f N", initial_force)
        if not args.sensor_only:
            leader = SOLeader(SOLeaderTeleopConfig(
                port=args.leader_port, id=args.leader_id, enable_wrist_yaw=True, use_degrees=True,
            ))
            if not leader.calibration:
                raise RuntimeError("リーダーのキャリブレーションファイルがありません。先にlerobot-calibrateを実行してください。")
            leader.connect(calibrate=False)
            if not leader.is_calibrated:
                raise RuntimeError("リーダーの保存済みキャリブレーションと実機が一致しません。先に既存値を適用してください。")
            LOGGER.info("Leader motor IDs: %s", {name: motor.id for name, motor in leader.bus.motors.items()})
            if args.feedback_mode == "hold":
                feedback = GripperFeedback(leader.bus)
                feedback.configure(args.torque_limit)
            if args.follower_port:
                from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
                from lerobot.robots.so_follower.so_follower import SOFollower

                follower = SOFollower(SOFollowerRobotConfig(
                    port=args.follower_port, id=args.follower_id, enable_wrist_yaw=True, use_degrees=True,
                    max_relative_target=args.max_relative_target,
                ))
                LOGGER.info("Follower motor IDs: %s", {name: motor.id for name, motor in follower.bus.motors.items()})
                if not follower.calibration:
                    raise RuntimeError("フォロワーのキャリブレーションファイルがありません。")
                # Check calibration before enabling follower torque.
                follower.bus.connect()
                if not follower.is_calibrated:
                    raise RuntimeError("フォロワーの保存済みキャリブレーションと実機が一致しません。")
                follower.bus.disable_torque()
                follower.bus.sync_write("Goal_Position", follower.bus.sync_read("Present_Position"))
                input("両アームを対応する姿勢に合わせ、周囲を空けてEnterで追従開始: ")
                sensor.get_force()
                follower.bus.sync_write("Goal_Position", follower.bus.sync_read("Present_Position"))
                follower.configure()
            LOGGER.info("開始: mode=%s fps=%.1f max_relative_target=%s。Ctrl+Cで終了。",
                        args.feedback_mode, args.fps, args.max_relative_target)
        last_log = 0.0
        rate_started = time.monotonic()
        rate_cycles = 0
        while True:
            started = time.monotonic()
            force = sensor.get_force()
            position = None
            if leader is not None:
                # Match standard teleop: read follower state before leader/action send.
                if follower is not None:
                    follower.get_observation()
                action = leader.get_action()
                position = action["gripper.pos"]
                # Recheck freshness after the motor read before applying any goals.
                force = sensor.get_force()
                if feedback is not None:
                    action = apply_pressure_action(action, force, limit, feedback)
                if follower is not None:
                    follower.send_action(action)
            rate_cycles += 1
            elapsed = time.monotonic() - rate_started
            if started - last_log >= 0.5:
                LOGGER.info("force=%.3f N gripper=%s limited=%s holding=%s", force,
                            "-" if position is None else f"{position:.2f}", limit.boundary is not None, limit.holding)
                LOGGER.info("loop_hz=%.1f", rate_cycles / elapsed if elapsed > 0 else 0)
                rate_started = time.monotonic()
                rate_cycles = 0
                last_log = started
            time.sleep(max(0, 1 / args.fps - (time.monotonic() - started)))
    finally:
        if leader is not None or follower is not None:
            LOGGER.warning("制御終了: 両アームのトルク解除を試みます。アームを支えてください。")
        # Attempt every cleanup even if one serial device has been unplugged.
        cleanup = []
        if feedback is not None:
            cleanup.append(feedback.close)
        if leader is not None and leader.is_connected:
            cleanup.append(leader.disconnect)
        if follower is not None and follower.is_connected:
            cleanup.append(follower.disconnect)
        cleanup.append(sensor.disconnect)
        for close in cleanup:
            try:
                close()
            except Exception:
                LOGGER.exception("終了処理で通信エラー。モーター電源を切ってください。")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        LOGGER.info("終了しました。")


if __name__ == "__main__":
    main()
