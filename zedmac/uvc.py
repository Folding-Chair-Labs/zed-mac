"""ctypes wrapper for native/libuvcctl.dylib: UVC class requests via IOKit (works alongside AVFoundation, no root)."""
from __future__ import annotations
import ctypes
import subprocess
from pathlib import Path

_NATIVE = Path(__file__).resolve().parent / "native"
_LIB = _NATIVE / "libuvcctl.dylib"
if not _LIB.exists() or _LIB.stat().st_mtime < (_NATIVE / "uvcctl.c").stat().st_mtime:
    subprocess.run(["clang", "-O2", "-dynamiclib", "-o", str(_LIB), str(_NATIVE / "uvcctl.c"),
                    "-framework", "IOKit", "-framework", "CoreFoundation"], check=True)
_lib = ctypes.CDLL(str(_LIB))
_lib.uvc_open.restype = ctypes.c_void_p
_lib.uvc_open.argtypes = [ctypes.c_int] * 3
_lib.uvc_request.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_lib.uvc_close.argtypes = [ctypes.c_void_p]

SET_CUR, GET_CUR, GET_LEN = 0x01, 0x81, 0x85


class UvcControl:
    def __init__(self, vid: int, pid: int, interface: int = 0):
        self.h = _lib.uvc_open(vid, pid, interface)
        if not self.h:
            raise RuntimeError(f"VideoControl interface {vid:04x}:{pid:04x}#{interface} not found")
        self.interface = interface

    def _req(self, rtype, req, unit, sel, buf: bytearray):
        cbuf = (ctypes.c_ubyte * len(buf)).from_buffer(buf)
        done = ctypes.c_int(0)
        r = _lib.uvc_request(self.h, rtype, req, sel << 8, (unit << 8) | self.interface, cbuf, len(buf), ctypes.byref(done))
        if r:
            raise OSError(f"UVC request 0x{req:02x} unit {unit} sel {sel} failed: IOReturn 0x{r & 0xffffffff:08x}")
        return done.value

    def get(self, unit, sel, length, req=GET_CUR) -> bytes:
        b = bytearray(length)
        n = self._req(0xA1, req, unit, sel, b)
        return bytes(b[:n])

    def set(self, unit, sel, data: bytes):
        self._req(0x21, SET_CUR, unit, sel, bytearray(data))

    def close(self):
        if self.h:
            _lib.uvc_close(self.h)
            self.h = None
