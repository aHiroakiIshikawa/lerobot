import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

path = Path(__file__).resolve().parents[1] / 'examples/so101_6dof/diagnose_motors.py'
if not path.exists():
    path = Path(__file__).with_name('diagnose_motors.py')
spec = importlib.util.spec_from_file_location('diagnose_motors', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestDiagnostics(unittest.TestCase):
    def test_read_only_with_status_errors(self):
        bus = Mock()
        bus.model_ctrl_table = {'sts3215': {name: (i, 1) for i, name in enumerate(module.REGISTERS)}}
        bus._read.return_value = (70, 0, 1)
        bus._is_comm_success.return_value = True
        bus.packet_handler.getRxPacketError.return_value = 'Input voltage error'
        with patch.object(module, 'FeetechMotorsBus', return_value=bus), patch('builtins.print') as output:
            module.diagnose('unused', [7])
        self.assertEqual(bus._read.call_count, len(module.REGISTERS))
        bus.connect.assert_called_once_with(handshake=False)
        bus.disconnect.assert_called_once_with(disable_torque=False)
        self.assertTrue(any('Input voltage error' in str(c) for c in output.call_args_list))
        for method in ('write', '_write', 'sync_write', 'enable_torque', 'disable_torque', 'write_calibration'):
            getattr(bus, method).assert_not_called()


if __name__ == '__main__':
    unittest.main()
