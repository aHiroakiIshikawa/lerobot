"""Hardware-free regression tests; run with uv run python -m unittest discover -s tests -p test_so_six_dof_layout.py."""

import ast
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_class(kind, calibration=None):
    """Execute actual device methods with transport/base-class dependencies stubbed."""
    role = 'follower' if kind == 'robots' else 'leader'
    path = ROOT / f'src/lerobot/{kind}/so_{role}/so_{role}.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    class Base:
        def __init__(self, config):
            self.calibration = calibration or {}
            self.id = config.id

    def motor(id, model, norm_mode):
        return SimpleNamespace(id=id, model=model, norm_mode=norm_mode)

    def bus(**kwargs):
        return SimpleNamespace(**kwargs, setup_motor=Mock())

    namespace = dict(
        Robot=Base, Teleoperator=Base, SOFollowerRobotConfig=object, SOLeaderTeleopConfig=object,
        Motor=motor, MotorNormMode=SimpleNamespace(DEGREES='degrees', RANGE_M100_100='range', RANGE_0_100='gripper'),
        FeetechMotorsBus=bus, make_cameras_from_configs=lambda config: {}, cached_property=cached_property,
        check_if_already_connected=lambda f: f, check_if_not_connected=lambda f: f,
    )
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[cls.name]


class TestSixDofLayout(unittest.TestCase):
    def test_layout_features_normalization_and_setup(self):
        for kind in ('robots', 'teleoperators'):
            for enabled in (False, True):
                for degrees in (False, True):
                    with self.subTest(kind=kind, enabled=enabled, degrees=degrees):
                        config = SimpleNamespace(id='test', port='unused', cameras={}, use_degrees=degrees, enable_wrist_yaw=enabled)
                        device = load_class(kind)(config)
                        names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
                        if enabled:
                            names.insert(4, 'wrist_yaw')
                        names.append('gripper')
                        self.assertEqual(list(device.bus.motors), names)
                        self.assertEqual([m.id for m in device.bus.motors.values()], list(range(1, len(names) + 1)))
                        self.assertEqual(list(device.action_features), [f'{n}.pos' for n in names])
                        if kind == 'robots':
                            self.assertEqual(device.observation_features, device.action_features)
                        for name, motor in device.bus.motors.items():
                            self.assertEqual(motor.norm_mode, 'gripper' if name == 'gripper' else 'degrees' if degrees else 'range')
                        with patch('builtins.input', return_value=''), patch('builtins.print'):
                            device.setup_motors()
                        self.assertEqual([c.args[0] for c in device.bus.setup_motor.call_args_list], list(reversed(names)))

    def test_rejects_old_or_mismatched_calibration(self):
        for kind in ('robots', 'teleoperators'):
            config = SimpleNamespace(id='test', port='unused', cameras={}, use_degrees=True, enable_wrist_yaw=True)
            for calibration in ({'gripper': SimpleNamespace(id=6)}, {'wrist_yaw': SimpleNamespace(id=7)}):
                with self.subTest(kind=kind, calibration=calibration), self.assertRaisesRegex(ValueError, 'Calibration'):
                    load_class(kind, calibration)(config)

    def test_rejects_previous_seven_motor_layout(self):
        names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'wrist_yaw', 'gripper']
        calibration = {name: SimpleNamespace(id=i) for i, name in enumerate(names, 1)}
        for kind in ('robots', 'teleoperators'):
            config = SimpleNamespace(id='test', port='unused', cameras={}, use_degrees=True, enable_wrist_yaw=True)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'Calibration'):
                load_class(kind, calibration)(config)

    def test_accepts_matching_calibration(self):
        names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper']
        calibration = {name: SimpleNamespace(id=i) for i, name in enumerate(names, 1)}
        for kind in ('robots', 'teleoperators'):
            config = SimpleNamespace(id='test', port='unused', cameras={}, use_degrees=True, enable_wrist_yaw=True)
            self.assertEqual(load_class(kind, calibration)(config).bus.calibration, calibration)


if __name__ == '__main__':
    unittest.main()
