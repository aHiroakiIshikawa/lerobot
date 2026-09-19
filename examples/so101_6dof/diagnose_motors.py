"""Read SO-101 motor registers without changing goals, torque or calibration."""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


REGISTERS = (
    "Present_Voltage", "Min_Voltage_Limit", "Max_Voltage_Limit",
    "Present_Temperature", "Present_Load", "Present_Current", "Status",
    "Torque_Enable", "Torque_Limit", "Max_Torque_Limit",
)


def diagnose(port, ids):
    bus = FeetechMotorsBus(port=port, motors={
        f"motor_{id_}": Motor(id_, "sts3215", MotorNormMode.RANGE_0_100) for id_ in ids
    })
    try:
        bus.connect(handshake=False)
        print(f"Port: {port}. Values are raw register values (not converted units).", flush=True)
        for id_ in ids:
            print(f"\nID {id_}", flush=True)
            for register in REGISTERS:
                address, length = bus.model_ctrl_table["sts3215"][register]
                try:
                    # A status error should be reported alongside the data, not hide it.
                    value, comm, error = bus._read(address, length, id_, raise_on_error=False)
                    if not bus._is_comm_success(comm):
                        print(f"  {register}: communication failure ({bus.packet_handler.getTxRxResult(comm)})", flush=True)
                        continue
                    status = bus.packet_handler.getRxPacketError(error) if error else "OK"
                    print(f"  {register}: {value}  [{status}]", flush=True)
                except (RuntimeError, ConnectionError, IndexError) as exc:
                    print(f"  {register}: read failed: {exc}", flush=True)
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--ids", nargs="+", type=int, default=[7], choices=range(1, 8))
    args = parser.parse_args()
    diagnose(args.port, list(dict.fromkeys(args.ids)))


if __name__ == "__main__":
    main()
