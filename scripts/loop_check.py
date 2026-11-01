"""Ground-truth the VIO start->end relative pose from images: stereo depth (RAFT) in the first frame, SIFT match to
the last frame, PnP -> true camera motion first->last. Compare with Basalt's trajectory (IMU poses -> cam0 via calib).
usage: loop_check.py <euroc_dir>"""
import json, sys, glob
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zedmac.neural import RaftStereo

E = Path(sys.argv[1]); meta = json.loads((E / "meta.json").read_text()); cal = json.loads((E / "basalt_calib.json").read_text())["value0"]
K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]]); B = meta["baseline_m"]; S = meta.get("right_shift_px", 0)
fs = sorted(glob.glob(str(E / "mav0/cam0/data/*.png")))
L0, R0, L1 = (cv2.imread(f, 0) for f in (fs[5], fs[5].replace("cam0", "cam1"), fs[-5]))
R0 = np.hstack([R0[:, S:], np.zeros((R0.shape[0], S), np.uint8)])  # undo the right-image shift for RAFT
disp = RaftStereo("middlebury", scale=1.0)(cv2.cvtColor(L0, cv2.COLOR_GRAY2RGB), cv2.cvtColor(R0, cv2.COLOR_GRAY2RGB))
sift = cv2.SIFT_create(4000); k0, d0 = sift.detectAndCompute(L0, None); k1, d1 = sift.detectAndCompute(L1, None)
m = [a for a, b in cv2.BFMatcher().knnMatch(d0, d1, k=2) if a.distance < 0.7 * b.distance]
p0 = np.float32([k0[a.queryIdx].pt for a in m]); p1 = np.float32([k1[a.trainIdx].pt for a in m])
dd = np.array([disp[int(y), int(x)] for x, y in p0]); ok = dd > 2
Z = meta["fx"] * B / dd[ok]
X = np.stack([(p0[ok, 0] - K[0, 2]) * Z / K[0, 0], (p0[ok, 1] - K[1, 2]) * Z / K[1, 1], Z], 1)
_, rv, tv, inl = cv2.solvePnPRansac(X, p1[ok], K, None, reprojectionError=1.0, iterationsCount=2000)
_, rv, tv = cv2.solvePnP(X[inl[:, 0]], p1[ok][inl[:, 0]], K, None, rv, tv, useExtrinsicGuess=True)
R_10 = cv2.Rodrigues(rv)[0]  # X_cam_last = R_10 X_cam_first + t
T_10 = np.eye(4); T_10[:3, :3] = R_10; T_10[:3, 3] = tv[:, 0]
T_true = np.linalg.inv(T_10)  # pose of last camera in first camera frame
print(f"image-based (PnP on {len(inl)} inliers / {len(m)} matches): camera moved {np.linalg.norm(T_true[:3, 3]) * 100:.1f} cm, "
      f"rotated {np.degrees(Rot.from_matrix(T_true[:3, :3]).magnitude()):.1f} deg")

np.savez(E / "loop_truth.npz", T_true=T_true, n0=int(Path(fs[5]).stem), n1=int(Path(fs[-5]).stem))
T = np.loadtxt(E / "basalt_out/trajectory.txt"); ts = T[:, 0]
c = cal["T_imu_cam"][0]; T_ic = np.eye(4); T_ic[:3, :3] = Rot.from_quat([c["qx"], c["qy"], c["qz"], c["qw"]]).as_matrix(); T_ic[:3, 3] = [c["px"], c["py"], c["pz"]]
def T_wc(ns):
    i = np.argmin(np.abs(ts - ns * 1e-9)); M = np.eye(4); M[:3, :3] = Rot.from_quat(T[i, 4:8]).as_matrix(); M[:3, 3] = T[i, 1:4]; return M @ T_ic
n0, n1 = (int(Path(f).stem) for f in (fs[5], fs[-5]))
T_vio = np.linalg.inv(T_wc(n0)) @ T_wc(n1)
print(f"VIO:                                camera moved {np.linalg.norm(T_vio[:3, 3]) * 100:.1f} cm, "
      f"rotated {np.degrees(Rot.from_matrix(T_vio[:3, :3]).magnitude()):.1f} deg")
Err = np.linalg.inv(T_true) @ T_vio
path = np.linalg.norm(np.diff(T[:, 1:4], axis=0), axis=1).sum()
print(f"VIO DRIFT over {path:.2f} m / {ts[-1] - ts[0]:.0f} s: translation {np.linalg.norm(Err[:3, 3]) * 100:.1f} cm "
      f"({100 * np.linalg.norm(Err[:3, 3]) / path:.2f}% of path), rotation {np.degrees(Rot.from_matrix(Err[:3, :3]).magnitude()):.2f} deg")
print("  translation error in first-camera frame (x right, y down, z fwd) cm:", np.round(Err[:3, 3] * 100, 1))
