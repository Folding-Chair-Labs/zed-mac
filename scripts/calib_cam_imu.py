"""Estimate camera<-IMU rotation, camera-IMU time offset, and gyro bias from a EuRoC recording with motion.

usage: .venv/bin/python scripts/calib_cam_imu.py recordings/<stamp>/euroc [--install]
  --install: also save to calib/cam_imu.json (used by setup_calib.py / make_basalt_calib.py)

Method: track features in cam0 (rectified pinhole), get frame-to-frame rotation from the essential matrix,
convert to camera angular velocity; average gyro over the same intervals (shifted by a candidate offset);
solve Kabsch for R_cam_imu at each offset and keep the offset with the lowest residual.
Needs the first ~2 s static (gyro bias) and then vigorous rotation about all three axes.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

root = Path(sys.argv[1])
install = "--install" in sys.argv
meta = json.loads((root / "meta.json").read_text())
K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]])
cams = np.loadtxt(root / "mav0/cam0/data.csv", delimiter=",", dtype=np.int64, comments="#", usecols=0)
imu = np.loadtxt(root / "mav0/imu0/data.csv", delimiter=",", comments="#")
ti, gyro, acc = imu[:, 0] * 1e-9, imu[:, 1:4], imu[:, 4:7]
tc = cams * 1e-9

# gyro bias from the static start (lowest-motion 1.5 s window in the first 5 s)
win = int(1.5 * 800)
first = gyro[ti < ti[0] + 5]
energy = [np.linalg.norm(first[i:i + win] - first[i:i + win].mean(0), axis=1).mean() for i in range(0, len(first) - win, 40)]
i0 = int(np.argmin(energy)) * 40
bias = first[i0:i0 + win].mean(0)
g_static = acc[(ti >= ti[i0]) & (ti < ti[i0] + 1.5)].mean(0)
print(f"gyro bias {np.degrees(bias).round(3)} deg/s (static motion level {energy[i0 // 40] * 57.3:.2f} deg/s); "
      f"static accel {g_static.round(3)} |g|={np.linalg.norm(g_static):.3f}")

# visual rotations

w_cam, t_mid, dts = [], [], []
lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
B = meta["baseline_m"]
f = meta["fx"]


def track(a, b, p):
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **lk)
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **lk)
    ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm(p - pb, axis=2)[:, 0] < 0.7)
    return q, ok


prev_l = prev_r = None
for j, ns in enumerate(cams):
    img = cv2.imread(str(root / "mav0/cam0/data" / f"{ns}.png"), cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(str(root / "mav0/cam1/data" / f"{ns}.png"), cv2.IMREAD_GRAYSCALE)
    if prev_l is not None:
        p0 = cv2.goodFeaturesToTrack(prev_l, 500, 0.01, 8)
        if p0 is not None and len(p0) > 30:
            # stereo: triangulate in previous frame (rectified -> same row, positive disparity)
            pr, okr = track(prev_l, prev_r, p0)
            d = p0[:, 0, 0] - pr[:, 0, 0] + meta.get("right_shift_px", 0)  # undo cam1 shift from to_euroc
            okr &= (np.abs(p0[:, 0, 1] - pr[:, 0, 1]) < 1.0) & (d > 1.0)
            # temporal: previous left -> current left
            p1, okt = track(prev_l, img, p0)
            good = okr & okt
            if good.sum() > 20:
                Z = f * B / d[good]
                X = np.stack([(p0[good, 0, 0] - K[0, 2]) * Z / f, (p0[good, 0, 1] - K[1, 2]) * Z / meta["fy"], Z], 1)
                okp, rvec, tvec, inl = cv2.solvePnPRansac(X, p1[good], K, None, reprojectionError=1.5,
                                                          iterationsCount=200, flags=cv2.SOLVEPNP_ITERATIVE)
                if okp and inl is not None and len(inl) > 15:
                    R, _ = cv2.Rodrigues(rvec)
                    dt = tc[j] - tc[j - 1]
                    # R maps frame-(j-1) coords to frame-j coords; camera rotation = R^T
                    w_cam.append(Rotation.from_matrix(R.T).as_rotvec() / dt)
                    t_mid.append(0.5 * (tc[j] + tc[j - 1]))
                    dts.append(dt)
    prev_l, prev_r = img, img_r
w_cam, t_mid, dts = np.array(w_cam), np.array(t_mid), np.array(dts)
moving = np.linalg.norm(w_cam, axis=1) > np.radians(15)
print(f"visual rotations: {len(w_cam)} intervals, {moving.sum()} with >15 deg/s rotation")
if moving.sum() < 30:
    sys.exit("not enough rotation; record again rotating the camera about all three axes")

cg = np.concatenate([np.zeros((1, 3)), np.cumsum((gyro[1:] - bias) * np.diff(ti)[:, None], axis=0)])  # integrated gyro


def gyro_avg(shift):
    a, b = t_mid - dts / 2 + shift, t_mid + dts / 2 + shift
    ia = np.stack([np.interp(a, ti, cg[:, k]) for k in range(3)], 1)
    ib = np.stack([np.interp(b, ti, cg[:, k]) for k in range(3)], 1)
    return (ib - ia) / dts[:, None]


def kabsch(A, B):  # R minimizing |A - R B|
    U, _, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def fit(shift, mask):
    wi = gyro_avg(shift)
    R = kabsch(w_cam[mask], wi[mask])
    r = np.linalg.norm(w_cam[mask] - wi[mask] @ R.T, axis=1)
    return np.median(r), R


shifts = np.arange(-0.1, 0.1, 0.001)
res = [fit(s, moving)[0] for s in shifts]
s0 = shifts[int(np.argmin(res))]
fine = np.arange(s0 - 0.002, s0 + 0.002, 0.0001)
resf = [fit(s, moving)[0] for s in fine]
shift = fine[int(np.argmin(resf))]
# robust refit excluding outliers
med, R = fit(shift, moving)
wi = gyro_avg(shift)
r = np.linalg.norm(w_cam - wi @ R.T, axis=1)
inl = moving & (r < 3 * med)
_, R = fit(shift, inl)
med2, _ = fit(shift, inl)
mean_speed = np.degrees(np.median(np.linalg.norm(w_cam[moving], axis=1)))
print(f"time offset: t_imu = t_cam + {shift * 1e3:+.1f} ms   (residual {np.degrees(med2):.2f} deg/s at median speed {mean_speed:.0f} deg/s)")
print("residual vs shift (ms: deg/s):", "  ".join(f"{s * 1e3:+.0f}:{np.degrees(v):.1f}" for s, v in zip(shifts[::10], res[::10])))
print("R_cam_imu =\n", R.round(4))
print("as nearest axis permutation:\n", np.round(R).astype(int))
g_cam = R @ g_static
print(f"static gravity reaction in cam frame: {g_cam.round(2)} (camera y is down, so a level camera reads ~[0,-9.8,0])")
out = {"R_cam_imu": R.tolist(), "t_imu_minus_t_cam_s": float(shift), "gyro_bias_rad_s": bias.tolist(),
       "residual_deg_s": float(np.degrees(med2)), "n_intervals": int(inl.sum())}
(root / "cam_imu.json").write_text(json.dumps(out, indent=2))
if install:
    dst = Path(__file__).resolve().parent.parent / "calib" / "cam_imu.json"
    dst.write_text(json.dumps(out, indent=2))
    print("installed", dst, "-- now rerun scripts/setup_calib.py")
print("wrote", root / "cam_imu.json")
