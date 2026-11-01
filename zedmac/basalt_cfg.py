"""Build Basalt calibration/config JSON for the rectified ZED Mini stream used by this project.

Images given to Basalt: rectified pinhole pair, downscaled to `eye_width` (default 640x360), with the right image
shifted RIGHT_SHIFT px (cam1 cx += RIGHT_SHIFT). Basalt's stereo optical flow starts every match at zero disparity
and fails beyond ~15 px; close-range scenes (0.4-1.5 m) have 25-40 px, so the shift recenters them.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

RIGHT_SHIFT = 30
# Measured on a ZED Mini (two independent recordings agreed within 1.8 deg): IMU x = cam y, IMU y = cam x,
# IMU z = -cam z, plus ~1 deg unit-specific tilt. Refine per unit with scripts/calib_cam_imu.py.
DEFAULT_R_CAM_IMU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# Tuned for close-range handheld / desk scenes at 640x360 (see scripts/basalt_sweep.py, scripts/vio_sweep.py).
CONFIG_OVERRIDES = {
    "config.optical_flow_detection_grid_size": 20,
    "config.optical_flow_levels": 3,
    "config.vio_min_triangulation_dist": 0.02,
    "config.vio_new_kf_keypoints_thresh": 0.9,
    "config.vio_min_frames_after_kf": 2,
}


def rectified_intrinsics(rect, eye_width: int = 640) -> dict:
    s = eye_width / rect.size[0]
    return {"fx": rect.P1[0, 0] * s, "fy": rect.P1[1, 1] * s, "cx": rect.P1[0, 2] * s, "cy": rect.P1[1, 2] * s,
            "eye_size": [eye_width, round(rect.size[1] * s)], "baseline_m": rect.baseline_m}


def build_calib(intr: dict, R_cam_imu=None, gyro_bias=(0.0, 0.0, 0.0), right_shift: int = RIGHT_SHIFT) -> dict:
    R_cam_imu = DEFAULT_R_CAM_IMU if R_cam_imu is None else np.asarray(R_cam_imu)
    R_imu_cam = R_cam_imu.T
    B = intr["baseline_m"]
    # Lever arm is not in the factory calibration; assume the IMU sits midway between the lenses
    # (VIO tolerates cm-level error here; moving it to cam0 or 2 cm back changed drift by <2 cm).
    p_imu_in_cam0 = np.array([B / 2, 0.0, 0.0])
    p0 = -R_imu_cam @ p_imu_in_cam0
    p1 = p0 + R_imu_cam @ np.array([B, 0.0, 0.0])  # rectified: cam1 = cam0 + B along x
    q = Rotation.from_matrix(R_imu_cam).as_quat()

    def pose(p):
        return {"px": p[0], "py": p[1], "pz": p[2], "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3]}

    k0 = {"camera_type": "pinhole", "intrinsics": {k: intr[k] for k in ("fx", "fy", "cx", "cy")}}
    k1 = json.loads(json.dumps(k0))
    k1["intrinsics"]["cx"] += right_shift
    gyro_nd = 5 * np.radians(0.007)   # BMI160-class datasheet density (deg/s/rtHz), x5 for real-world use
    acc_nd = 5 * 180e-6 * 9.81        # 180 ug/rtHz, x5
    return {"value0": {
        "T_imu_cam": [pose(p0), pose(p1)],
        "intrinsics": [k0, k1],
        "resolution": [intr["eye_size"], intr["eye_size"]],
        "calib_accel_bias": [0.0] * 9,
        "calib_gyro_bias": list(map(float, gyro_bias)) + [0.0] * 9,
        "imu_update_rate": 800.0,
        "accel_noise_std": [acc_nd] * 3,
        "gyro_noise_std": [gyro_nd] * 3,
        "accel_bias_std": [0.002] * 3,
        "gyro_bias_std": [0.0002] * 3,
        "cam_time_offset_ns": 0,
        "vignette": [],
    }}


def build_config(base_config: Path) -> dict:
    cfg = json.loads(Path(base_config).read_text())
    cfg["value0"].update(CONFIG_OVERRIDES)
    return cfg
