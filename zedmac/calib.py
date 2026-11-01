"""Load a Stereolabs SN<serial>.conf and build fisheye stereo rectification maps.

Verified on a ZED Mini: k1..k4 are OpenCV *fisheye* (Kannala-Brandt) coefficients,
R = Rodrigues([RX, CV, RZ]), T = [-Baseline, TY, TZ] (mm). Median rectified row error 0.57 px.
"""
from __future__ import annotations
import configparser
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

# side-by-side capture width -> (calibration section suffix, per-eye size)
MODES = {
    1344: ("VGA", (672, 376)),
    2560: ("HD", (1280, 720)),
    3840: ("FHD", (1920, 1080)),
    4416: ("2K", (2208, 1242)),
}


@dataclass
class Rectifier:
    size: tuple[int, int]          # per-eye (w, h)
    maps_l: tuple[np.ndarray, np.ndarray]
    maps_r: tuple[np.ndarray, np.ndarray]
    P1: np.ndarray                 # rectified left projection
    Q: np.ndarray                  # disparity->depth reprojection
    baseline_m: float

    @property
    def f(self) -> float:
        return float(self.P1[0, 0])

    def rectify(self, left: np.ndarray, right: np.ndarray, interp=cv2.INTER_LINEAR):
        return (cv2.remap(left, *self.maps_l, interp), cv2.remap(right, *self.maps_r, interp))

    def split_rectify(self, sbs: np.ndarray):
        w = sbs.shape[1] // 2
        return self.rectify(sbs[:, :w], sbs[:, w:])

    def depth_from_disparity(self, disp: np.ndarray, min_disp: float = 0.5) -> np.ndarray:
        """Metric depth (m) from left-image disparity (px); invalid -> nan."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(disp > min_disp, self.f * self.baseline_m / disp, np.nan).astype(np.float32)


def find_conf(serial: str | int | None = None, root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parent.parent / "calib"
    if serial is not None:
        return root / f"SN{serial}.conf"
    confs = sorted(root.glob("SN*.conf"))
    if not confs:
        raise FileNotFoundError(f"no SN*.conf in {root}; fetch https://calib.stereolabs.com/?SN=<serial>")
    return confs[0]


def load_rectifier(sbs_width: int, conf: Path | None = None, balance: float = 0.0) -> Rectifier:
    suffix, (w, h) = MODES[sbs_width]
    cfg = configparser.ConfigParser()
    cfg.read(conf or find_conf())

    def K(sec):
        c = cfg[sec]
        return np.array([[float(c["fx"]), 0, float(c["cx"])], [0, float(c["fy"]), float(c["cy"])], [0, 0, 1]])

    def D(sec):
        c = cfg[sec]
        return np.array([float(c[k]) for k in ("k1", "k2", "k3", "k4")]).reshape(4, 1)

    st = cfg["STEREO"]
    R, _ = cv2.Rodrigues(np.array([float(st[f"RX_{suffix}"]), float(st[f"CV_{suffix}"]), float(st[f"RZ_{suffix}"])]))
    B = float(st["Baseline"])
    T = np.array([-B, float(st["TY"]), float(st["TZ"])]).reshape(3, 1)
    KL, DL, KR, DR = K(f"LEFT_CAM_{suffix}"), D(f"LEFT_CAM_{suffix}"), K(f"RIGHT_CAM_{suffix}"), D(f"RIGHT_CAM_{suffix}")
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(KL, DL, KR, DR, (w, h), R, T,
                                                 flags=cv2.CALIB_ZERO_DISPARITY, balance=balance)
    ml = cv2.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, (w, h), cv2.CV_16SC2)
    mr = cv2.fisheye.initUndistortRectifyMap(KR, DR, R2, P2, (w, h), cv2.CV_16SC2)
    return Rectifier((w, h), ml, mr, P1, Q, B / 1000.0)
