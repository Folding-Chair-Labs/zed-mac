"""ZED Mini IMU over USB HID (VID 0x2B03, PID 0xF681). Packet format from stereolabs/zed-open-capture.

Each sample: device timestamp (s, MCU clock), gyro (rad/s), accel (m/s^2), frame_sync flag + counter
(the MCU latches a sample at each camera exposure -> camera timestamps on the IMU clock).
"""
from __future__ import annotations
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
import numpy as np
import hid

VID, PID_ZEDM = 0x2B03, 0xF681
REP_SENSOR_DATA, REP_REQUEST_SET, REP_STREAM_STATUS = 0x01, 0x21, 0x32
RQ_CMD_PING = 0xF2
G = 9.8189
ACC_SCALE = G * 8.0 / 32768.0
GYRO_SCALE_DPS = 1000.0 / 32768.0
TS_SCALE_S = 39062.5e-9
# packed RawData (pragma pack 1)
FMT = "<BBQhhhhhhBBIhBhhhBIBIBhIIhh"
SIZE = struct.calcsize(FMT)


@dataclass
class ImuSample:
    t_dev: float          # MCU time (s)
    t_host: float         # time.monotonic() at receipt
    gyro: np.ndarray      # rad/s, sensor frame
    acc: np.ndarray       # m/s^2, sensor frame
    frame_sync: bool
    sync_count: int
    temp_c: float


def parse(buf: bytes, t_host: float) -> ImuSample | None:
    if len(buf) < SIZE or buf[0] not in (0x00, REP_SENSOR_DATA):  # ZED-M uses 0x00
        return None
    f = struct.unpack_from(FMT, bytes(buf))
    if f[1] == 1:  # imu_not_valid
        return None
    gyro = np.radians(np.array(f[3:6], np.float64) * GYRO_SCALE_DPS)
    acc = np.array(f[6:9], np.float64) * ACC_SCALE
    return ImuSample(f[2] * TS_SCALE_S, t_host, gyro, acc, bool(f[9]), f[11], f[12] * 0.01)


class ZedImu:
    def __init__(self, maxlen: int = 400 * 600, out_queue=None):
        info = [d for d in hid.enumerate(VID) if d["product_id"] == PID_ZEDM]
        if not info:
            raise RuntimeError("ZED-M HID device (2b03:f681) not found")
        self.serial = info[0]["serial_number"]
        self.dev = hid.device()
        self.dev.open_path(info[0]["path"])
        self.dev.send_feature_report([REP_STREAM_STATUS, 1])
        self.samples: deque[ImuSample] = deque(maxlen=maxlen)
        self.out_queue = out_queue  # optional queue.Queue receiving every sample (for live streaming)
        self.lock = threading.Lock()
        self._stop = False
        self._wrap = 0
        self._last_raw = None
        self.ping_log: list[tuple[float, float, int]] = []  # (t, duration_s, result)
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._pinger, daemon=True).start()

    def _pinger(self):
        # Separate thread: without pings the MCU stops streaming after ~11 s. A few % of SET_REPORTs block for
        # IOKit's 5 s timeout and fail (-1); on the read thread that dropped 5 s of IMU data. (64-byte ping: no effect.)
        while not self._stop:
            t = time.monotonic()
            try:
                r = self.dev.send_feature_report([REP_REQUEST_SET, RQ_CMD_PING])
            except OSError:
                r = -2
            self.ping_log.append((t, time.monotonic() - t, r))
            time.sleep(0.5)

    def _loop(self):
        while not self._stop:
            buf = self.dev.read(64, timeout_ms=2000)
            th = time.monotonic()
            if not buf:
                continue
            s = parse(bytes(buf), th)
            if s is not None:
                with self.lock:
                    self.samples.append(s)
                if self.out_queue is not None:
                    self.out_queue.put(s)

    def snapshot(self) -> list[ImuSample]:
        with self.lock:
            return list(self.samples)

    def close(self):
        self._stop = True
        time.sleep(0.05)
        try:
            self.dev.send_feature_report([REP_STREAM_STATUS, 0])
        finally:
            self.dev.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
