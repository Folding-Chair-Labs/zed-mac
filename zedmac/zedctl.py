"""ZED camera vendor controls over the OV580 UVC extension unit (protocol from stereolabs/zed-open-capture)."""
from __future__ import annotations
import time
from .uvc import UvcControl, GET_LEN

VID, PID_ZEDM_VIDEO = 0x2B03, 0xF682
XU_UNIT, XU_SEL = 0x04, 0x02
XU_TASK_SET, XU_TASK_GET = 0x50, 0x51
I2C_LEFT, I2C_RIGHT = 0xA3, 0xA5   # ASIC_INT_NULL_I2C (side 0), ASIC_INT_I2C (side 1)
REG_SYNC = 0x3002                  # OV sensor: bit7 = VSYNC output enable


class ZedVendor:
    def __init__(self, pid: int = PID_ZEDM_VIDEO):
        self.uvc = UvcControl(VID, pid, 0)
        self.xlen = int.from_bytes(self.uvc.get(XU_UNIT, XU_SEL, 2, req=GET_LEN), "little")  # 384 on USB3

    def _sensor_buf(self, task, side, address, value=0):
        b = bytearray(self.xlen)
        b[0] = task
        b[1] = I2C_LEFT if side == 0 else I2C_RIGHT
        b[2] = 0x6C
        b[3] = 2            # address width (sscb_id 1 + 1)
        b[4] = 1            # data width
        b[5:9] = address.to_bytes(4, "big")
        b[9], b[10], b[11], b[12] = 0x90, 0x01, 0x00, 0x01   # limit=1 with page-addr flags (0x10|0x80)
        b[16] = value
        return b

    def read_sensor_reg(self, side: int, address: int) -> int:
        self.uvc.set(XU_UNIT, XU_SEL, self._sensor_buf(XU_TASK_GET, side, address))
        time.sleep(0.002)
        return self.uvc.get(XU_UNIT, XU_SEL, self.xlen)[17]

    def write_sensor_reg(self, side: int, address: int, value: int):
        self.uvc.set(XU_UNIT, XU_SEL, self._sensor_buf(XU_TASK_SET, side, address, value))
        time.sleep(0.002)

    def enable_imu_sync(self) -> tuple[int, int]:
        """Turn on VSYNC output of both sensors so the IMU MCU flags samples at each exposure."""
        out = []
        for side in (0, 1):
            v = self.read_sensor_reg(side, REG_SYNC)
            self.write_sensor_reg(side, REG_SYNC, v | 0x80)
            out.append(self.read_sensor_reg(side, REG_SYNC))
        return tuple(out)

    def close(self):
        self.uvc.close()
