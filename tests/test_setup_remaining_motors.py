import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

path = Path(__file__).resolve().parents[1] / 'examples/so101_6dof/setup_remaining_motors.py'
if not path.exists():
    path = Path(__file__).with_name('setup_remaining_motors.py')
spec = importlib.util.spec_from_file_location('setup_remaining', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestRecovery(unittest.TestCase):
    def bus(self, responses):
        bus = Mock()
        bus.model_number_table = {'sts3215': 777}
        bus.model_baudrate_table = {'sts3215': {1000000: 0}}
        bus.default_baudrate = 1000000
        bus.broadcast_ping.side_effect = responses
        return bus

    @patch.object(module.time, 'sleep')
    def test_short_response_then_recovery(self, _):
        bus = self.bus([IndexError(), {9: 777}])
        self.assertEqual(module.find_single_motor(bus), (1000000, 9))
        self.assertEqual(bus.port_handler.clearPort.call_count, 2)
        bus.setup_motor.assert_not_called()

    @patch.object(module.time, 'sleep')
    def test_discovery_failures_do_not_write(self, _):
        for responses in ([IndexError()] * 3, [{2: 777, 3: 777}], [{3: 999}], [None] * 3):
            bus = self.bus(responses)
            with self.subTest(responses=responses), patch.object(module, 'FeetechMotorsBus', return_value=bus), patch('builtins.input', return_value=''), self.assertRaises(RuntimeError):
                module.setup_one('unused', 'elbow_flex')
            bus.setup_motor.assert_not_called()
            bus.disconnect.assert_called_once_with(disable_torque=False)

    @patch.object(module.time, 'sleep')
    def test_write_and_verify(self, _):
        bus = self.bus([{9: 777}, {3: 777}])
        with patch.object(module, 'FeetechMotorsBus', return_value=bus), patch('builtins.input', return_value=''):
            module.setup_one('unused', 'elbow_flex')
        bus.connect.assert_called_once_with(handshake=False)
        bus.setup_motor.assert_called_once_with('elbow_flex', initial_baudrate=1000000, initial_id=9)
        bus.disconnect.assert_called_once_with(disable_torque=False)

    @patch.object(module.time, 'sleep')
    def test_failed_verification_does_not_repeat_write(self, _):
        bus = self.bus([{9: 777}, {9: 777}])
        with patch.object(module, 'FeetechMotorsBus', return_value=bus), patch('builtins.input', return_value=''), self.assertRaisesRegex(RuntimeError, '設定後'):
            module.setup_one('unused', 'elbow_flex')
        self.assertEqual(bus.setup_motor.call_count, 1)
        bus.disconnect.assert_called_once_with(disable_torque=False)


if __name__ == '__main__':
    unittest.main()
