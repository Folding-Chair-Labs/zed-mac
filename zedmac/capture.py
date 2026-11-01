"""ZED side-by-side capture through macOS AVFoundation (via ffmpeg), latest-frame-wins."""
from __future__ import annotations
import re
import subprocess
import threading
import time
import numpy as np

FPS = {1344: [100, 60, 30, 15], 2560: [60, 30, 15], 3840: [30, 15], 4416: [15]}
EXACT = {100: "100.000000", 60: "60.000240", 30: "30.000030", 15: "15.000015"}


def find_device(pattern: str = r"ZED") -> str:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "audio devices" in line:
            break
        m = re.search(r"\]\s+\[(\d+)\]\s+(.*)$", line)
        if m and re.search(pattern, m.group(2)):
            return m.group(2).strip()
    raise RuntimeError("No ZED camera found. Use a USB 3 port; if only 'ZED-M Hid Device' enumerates, flip the USB-C plug.")


class ZedCapture:
    def __init__(self, width: int = 2560, fps: int = 30, device: str | None = None):
        self.w, self.h = width, {1344: 376, 2560: 720, 3840: 1080, 4416: 1242}[width]
        assert fps in FPS[width], f"{width} supports {FPS[width]} fps"
        self.device = device or find_device()
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-pixel_format", "yuyv422",
               "-video_size", f"{self.w}x{self.h}", "-framerate", EXACT[fps], "-i", self.device,
               "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._frame, self._t, self._n = None, 0.0, 0
        self._cv = threading.Condition()
        self._stop = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        nbytes = self.w * self.h * 3
        buf = bytearray(nbytes)
        view = memoryview(buf)
        while not self._stop:
            got = 0
            while got < nbytes:
                k = self.proc.stdout.readinto(view[got:])
                if not k:
                    self._stop = True
                    with self._cv:
                        self._cv.notify_all()
                    return
                got += k
            frame = np.frombuffer(bytes(buf), np.uint8).reshape(self.h, self.w, 3)
            with self._cv:
                self._frame, self._t, self._n = frame, time.monotonic(), self._n + 1
                self._cv.notify_all()

    def read(self, timeout: float = 5.0, newer_than: int = 0):
        """Return (frame_index, timestamp, sbs_bgr). Blocks until a frame newer than `newer_than` arrives."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._n > newer_than or self._stop, timeout):
                raise TimeoutError("no frame from ZED within timeout")
            if self._frame is None:
                err = self.proc.stderr.read().decode(errors="replace") if self.proc.poll() is not None else ""
                raise RuntimeError(f"capture ended: {err[-300:]}")
            return self._n, self._t, self._frame

    def close(self):
        self._stop = True
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
