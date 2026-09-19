import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

path = Path(__file__).resolve().parents[1] / "examples/so101_6dof/pressure_teleoperate.py"
if not path.exists():
    path = Path(__file__).with_name("pressure_teleoperate.py")
spec = importlib.util.spec_from_file_location("pressure_teleoperate", path)
module = importlib.util.module_from_spec(spec)
# dataclasses inspects the defining module on some Python versions.
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class TestPressure(unittest.TestCase):
    def test_threshold_immediately_enables_hold_without_extra_movement(self):
        for direction in ('decreasing', 'increasing'):
            with self.subTest(direction=direction):
                bus = Mock()
                feedback = module.GripperFeedback(bus)
                limit = module.GripperLimit(close_direction=direction)
                module.apply_pressure_action({'gripper.pos': 40}, 0.9, limit, feedback)
                bus.enable_torque.assert_not_called()
                module.apply_pressure_action({'gripper.pos': 40}, 1.0, limit, feedback)
                bus.enable_torque.assert_called_once_with('gripper')
                bus.write.assert_called_once_with('Goal_Position', 'gripper', 40)
                self.assertTrue(limit.holding)
                module.apply_pressure_action({'gripper.pos': 40}, 0.9, limit, feedback)
                self.assertTrue(limit.holding)
                module.apply_pressure_action({'gripper.pos': 40}, 0.8, limit, feedback)
                bus.disable_torque.assert_called_once_with('gripper')

    def test_defaults_match_standard_teleop(self):
        with patch.object(sys, 'argv', ['pressure_teleoperate.py', '--sensor-port', 'sensor']):
            args = module.parse_args()
        self.assertEqual(args.fps, 60)
        self.assertIsNone(args.max_relative_target)

    def test_pressure_changes_only_gripper_for_all_pressure_states(self):
        limiter = module.GripperLimit()
        feedback = Mock()
        arm = {'shoulder_pan.pos': 10, 'shoulder_lift.pos': -100, 'elbow_flex.pos': 30,
               'wrist_flex.pos': -40, 'wrist_yaw.pos': 50, 'wrist_roll.pos': 60}
        for force, position, expected in ((0, 50, 50), (2, 50, 50), (2, 40, 50), (2, 60, 60), (0, 30, 30)):
            action = {**arm, 'gripper.pos': position}
            result = module.apply_pressure_action(action, force, limiter, feedback)
            self.assertEqual({k: result[k] for k in arm}, arm)
            self.assertEqual(result['gripper.pos'], expected)
            self.assertEqual(action['gripper.pos'], position)

    def test_same_holding_goal_is_not_rewritten(self):
        bus = Mock()
        feedback = module.GripperFeedback(bus)
        feedback.apply(48, 50)
        bus.reset_mock()
        feedback.apply(47, 50)
        bus.write.assert_not_called()
        bus.enable_torque.assert_not_called()

    def test_monitor_loop_preserves_seven_targets_and_no_pressure_writes(self):
        from lerobot.robots.so_follower import so_follower
        args = SimpleNamespace(threshold=1, release_threshold=0.8, deadband=0.5, close_direction='decreasing',
                               sensor_port='sensor', sensor_baudrate=9600, sensor_timeout=0.5,
                               sensor_only=False, leader_port='leader', leader_id='test', torque_limit=100,
                               follower_port='follower', follower_id='test', fps=60,
                               feedback_mode='monitor', max_relative_target=None)
        sensor, leader, follower = Mock(), Mock(), Mock()
        leader.bus.motors = follower.bus.motors = {}
        action = {f'{name}.pos': i for i, name in enumerate(
            ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper'))}
        leader.get_action.return_value = action
        sensor.get_force.return_value = 9  # Exceeds threshold; monitor must not clamp.
        follower.send_action.side_effect = KeyboardInterrupt
        with patch.object(module, 'PressureSensor', return_value=sensor), patch.object(module, 'SOLeader', return_value=leader), patch.object(so_follower, 'SOFollower', return_value=follower) as factory, patch('builtins.input', return_value=''), self.assertRaises(KeyboardInterrupt):
            module.run(args)
        self.assertIsNone(factory.call_args.args[0].max_relative_target)
        follower.get_observation.assert_called_once()
        follower.send_action.assert_called_once_with(action)
        for method in ('write', 'enable_torque', 'disable_torque'):
            getattr(leader.bus, method).assert_not_called()
        leader.disconnect.assert_called_once()
        follower.disconnect.assert_called_once()

    def test_main_overrides_existing_warning_logging(self):
        import io
        import logging
        from contextlib import redirect_stderr

        root_logger = logging.getLogger()
        old_handlers, old_level = root_logger.handlers[:], root_logger.level
        output = io.StringIO()
        try:
            root_logger.handlers = [logging.StreamHandler(output)]
            root_logger.setLevel(logging.WARNING)
            with redirect_stderr(output), patch.object(module, "parse_args"), patch.object(
                module, "run", side_effect=lambda args: module.LOGGER.info("force=0.123 N")
            ):
                module.main()
            self.assertIn("force=0.123 N", output.getvalue())
        finally:
            root_logger.handlers = old_handlers
            root_logger.setLevel(old_level)

    def test_arduino_parser(self):
        self.assertEqual(module.parse_force("g = 20.00 [g] F = 0.196 [N]\r\n"), 0.196)
        self.assertEqual(module.parse_force("F=1.2e+1 [N]"), 12)
        for text in ("F = -1 [N]", "F = nan [N]", "F = 1e999 [N]", "F = 1. [g]", "booting"):
            self.assertIsNone(module.parse_force(text))

    def test_close_hold_open_and_release(self):
        limiter = module.GripperLimit()
        self.assertEqual(limiter.update(50, 0.1), (50, None))
        self.assertEqual(limiter.update(50, 1.1), (50, 50))
        self.assertEqual(limiter.update(48, 1.1), (50, 50))
        # Pressure between thresholds keeps the limit active.
        self.assertEqual(limiter.update(49, 0.9), (50, 50))
        self.assertEqual(limiter.update(55, 1.1), (55, None))
        self.assertEqual(limiter.update(52, 1.1), (55, 55))
        self.assertEqual(limiter.update(48, 0.8), (48, None))

    def test_reverse_direction(self):
        limiter = module.GripperLimit(close_direction="increasing")
        limiter.update(50, 1.1)
        self.assertEqual(limiter.update(52, 1.1), (50, 50))
        self.assertEqual(limiter.update(45, 1.1), (45, None))
        self.assertEqual(limiter.update(48, 1.1), (45, 45))

    def test_deadband_does_not_chatter(self):
        limiter = module.GripperLimit()
        limiter.update(50, 1)
        self.assertEqual(limiter.update(49.8, 1), (50, 50))
        limiter.update(49, 1)
        self.assertEqual(limiter.update(50.1, 1)[1], 50.1)
        self.assertIsNone(limiter.update(51, 1)[1])

    def test_invalid_settings_and_samples(self):
        for kwargs in ({"threshold": 0.5}, {"release_threshold": -1}, {"threshold": float("nan")}, {"deadband": -1}):
            with self.assertRaises(ValueError):
                module.GripperLimit(**kwargs)
        with self.assertRaises(ValueError):
            module.GripperLimit().update(float("nan"), 1)

    def test_stale_missing_and_error_samples_stop(self):
        sensor = module.PressureSensor("unused", 9600, 0.5)
        with self.assertRaises(RuntimeError):
            sensor.get_force()
        sensor.sample = (0.5, 10)
        with patch.object(module.time, "monotonic", return_value=10.4):
            self.assertEqual(sensor.get_force(), 0.5)
        with patch.object(module.time, "monotonic", return_value=10.6), self.assertRaises(RuntimeError):
            sensor.get_force()
        sensor.error = OSError("unplugged")
        with self.assertRaisesRegex(RuntimeError, "通信エラー"):
            sensor.get_force()

    def test_fragmented_serial_lines(self):
        sensor = module.PressureSensor("unused", 9600, 0.5)
        sensor.serial = Mock()
        sensor.serial.read_until.side_effect = [b"g = 1 [g] F = 1.", b"25 [N]\n", OSError("end test")]
        sensor._read_loop()
        self.assertEqual(sensor.sample[0], 1.25)
        self.assertIsInstance(sensor.error, OSError)

    def test_feedback_goal_precedes_enable_and_release(self):
        bus = Mock()
        feedback = module.GripperFeedback(bus)
        feedback.apply(48, 50)
        self.assertEqual(bus.mock_calls, [call.write("Goal_Position", "gripper", 48),
                                        call.enable_torque("gripper"), call.write("Goal_Position", "gripper", 50)])
        feedback.apply(55, None)
        bus.disable_torque.assert_called_once_with("gripper")

    def test_restore_feedback_registers(self):
        bus = Mock()
        bus.read.return_value = 1000
        feedback = module.GripperFeedback(bus)
        feedback.configure(100)
        feedback.close()
        self.assertIn(call.write("Torque_Limit", "gripper", 1000, normalize=False), bus.mock_calls)
        self.assertTrue(all(c.args[0] == "Torque_Limit" and c.args[1] == "gripper"
                            for c in bus.write.call_args_list))
        self.assertFalse(feedback.enabled)

    def test_loop_sensor_failure_cleans_up_without_sending_goal(self):
        args = SimpleNamespace(threshold=1, release_threshold=0.8, deadband=0.5, close_direction="decreasing",
                               sensor_port="sensor", sensor_baudrate=9600, sensor_timeout=0.5,
                               sensor_only=False, leader_port="leader", leader_id="test", torque_limit=100,
                               follower_port=None, fps=60, feedback_mode="hold", max_relative_target=None)
        sensor, leader = Mock(), Mock()
        sensor.get_force.side_effect = [0.1, RuntimeError("stale")]
        leader.get_action.return_value = {"gripper.pos": 50}
        leader.bus.motors = {}
        with patch.object(module, "PressureSensor", return_value=sensor), patch.object(module, "SOLeader", return_value=leader), self.assertRaisesRegex(RuntimeError, "stale"):
            module.run(args)
        leader.disconnect.assert_called_once()
        sensor.disconnect.assert_called_once()
        self.assertFalse(any(c.args and c.args[0] == "Goal_Position" for c in leader.bus.write.call_args_list))


if __name__ == "__main__":
    unittest.main()
