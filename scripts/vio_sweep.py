"""Sweep calibration/config variants, score each Basalt run against loop_truth.npz (from loop_check.py).
usage: vio_sweep.py <euroc_dir>"""
import json, subprocess, sys, tempfile, shutil
from pathlib import Path
import numpy as np, ubjson
from scipy.spatial.transform import Rotation as Rot
E = Path(sys.argv[1]).resolve()
BIN = Path(__file__).resolve().parent.parent / "third_party/basalt/build/release/basalt_vio"
truth = np.load(E / "loop_truth.npz"); T_true = truth["T_true"]
cal0 = json.loads((E / "basalt_calib.json").read_text()); cfg0 = json.loads((E / "basalt_config.json").read_text())
R2 = np.array(json.loads((E / "cam_imu_own.json").read_text())["R_cam_imu"]) if (E / "cam_imu_own.json").exists() else None

def mat(c):
    M = np.eye(4); M[:3, :3] = Rot.from_quat([c["qx"], c["qy"], c["qz"], c["qw"]]).as_matrix(); M[:3, 3] = [c["px"], c["py"], c["pz"]]; return M
def put(c, M):
    q = Rot.from_matrix(M[:3, :3]).as_quat(); c.update(px=M[0, 3], py=M[1, 3], pz=M[2, 3], qx=q[0], qy=q[1], qz=q[2], qw=q[3])

def run(name, cal_fn=None, cfg_over=None):
    d = Path(tempfile.mkdtemp()); cal = json.loads(json.dumps(cal0)); cfg = json.loads(json.dumps(cfg0))
    if cal_fn: cal_fn(cal["value0"])
    if cfg_over: cfg["value0"].update(cfg_over)
    (d / "c.json").write_text(json.dumps(cal)); (d / "g.json").write_text(json.dumps(cfg))
    subprocess.run([str(BIN), "--dataset-path", str(E), "--dataset-type", "euroc", "--cam-calib", str(d / "c.json"),
                    "--config-path", str(d / "g.json"), "--show-gui", "0", "--save-trajectory", "tum"], cwd=d, capture_output=True)
    T = np.loadtxt(d / "trajectory.txt"); ts = T[:, 0]; T_ic = mat(cal["value0"]["T_imu_cam"][0])
    def T_wc(ns):
        i = np.argmin(np.abs(ts - ns * 1e-9)); M = np.eye(4); M[:3, :3] = Rot.from_quat(T[i, 4:8]).as_matrix(); M[:3, 3] = T[i, 1:4]; return M @ T_ic
    Err = np.linalg.inv(T_true) @ np.linalg.inv(T_wc(int(truth["n0"]))) @ T_wc(int(truth["n1"]))
    path = np.linalg.norm(np.diff(T[:, 1:4], axis=0), axis=1).sum()
    lms = np.array(ubjson.loadb((d / "stats_all.ubjson").read_bytes())["num_lms"], float).mean()
    print(f"{name:34s} drift {np.linalg.norm(Err[:3, 3]) * 100:5.1f} cm ({100 * np.linalg.norm(Err[:3, 3]) / path:4.2f}%)  "
          f"rot {np.degrees(Rot.from_matrix(Err[:3, :3]).magnitude()):5.2f} deg  lms {lms:5.1f}", flush=True)
    shutil.rmtree(d)

def scale_noise(k):
    def f(c):
        c["accel_noise_std"] = [x * k for x in c["accel_noise_std"]]; c["gyro_noise_std"] = [x * k for x in c["gyro_noise_std"]]
    return f
def offset(ms):
    def f(c): c["cam_time_offset_ns"] = int(ms * 1e6)
    return f
def lever(p_imu_in_cam0):
    def f(c):
        for i, dx in enumerate((0.0, None)):
            M = mat(c["T_imu_cam"][i]); R_ic = M[:3, :3]; B = 0.0626 * i
            M[:3, 3] = -R_ic @ (np.array(p_imu_in_cam0) - np.array([B, 0, 0])); put(c["T_imu_cam"][i], M)
    return f
def rot_from(R_cam_imu):
    def f(c):
        for i in range(2):
            M = mat(c["T_imu_cam"][i]); p_c = -M[:3, :3].T @ M[:3, 3]  # imu position in cam_i
            M[:3, :3] = R_cam_imu.T; M[:3, 3] = -R_cam_imu.T @ p_c; put(c["T_imu_cam"][i], M)
    return f

def gyro_scale(diag):  # imu-axis diagonal of the 3x3 part (params 3..11 = column-major M, corrected uses I+M)
    def f(c):
        g = list(c["calib_gyro_bias"]); g[3], g[7], g[11] = diag; c["calib_gyro_bias"] = g
    return f
def accel_scale(k):  # params 3..8 lower-triangular, diag at 3, 6, 8
    def f(c):
        a = list(c["calib_accel_bias"]); a[3] = a[6] = a[8] = k; c["calib_accel_bias"] = a
    return f
N = scale_noise(0.2)
run("current")
run("noise x0.2", N)
D = (-0.0022, -0.0021, -0.0129)  # IMU x<->cam y, IMU y<->cam x, IMU z<->cam z
run("noise x0.2 + gyro scale (I+M), M=-d", lambda c: (N(c), gyro_scale(D)(c)))
run("noise x0.2 + gyro scale (I+M), M=+d", lambda c: (N(c), gyro_scale(tuple(-x for x in D))(c)))
run("noise x0.2 + gyro -d + accel -0.7%", lambda c: (N(c), gyro_scale(D)(c), accel_scale(-0.007)(c)))
run("noise x0.2 + gyro -d + accel +0.7%", lambda c: (N(c), gyro_scale(D)(c), accel_scale(0.007)(c)))
