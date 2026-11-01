"""Live visual-inertial odometry for the ZED Mini on macOS: capture + IMU sync here, Basalt in a C++ bridge.

usage: .venv/bin/python scripts/live_vio.py [--headless SECONDS] [--save out.txt]
  Keep the camera still for the first ~2 s (gyro bias), then move.
  keys: q quit | r recenter view | c clear trail

Pipeline: AVFoundation frames (ffmpeg) -> rectify -> 640x360 gray (right image shifted 30 px, matching the calib)
          IMU HID 800 Hz -> timestamps from MCU clock (x1.004629 tick correction)
          frame exposure time = IMU sample where sync_count increments (frame idx = sync_count + k, k locked online)
          -> third_party/basalt/build/release/basalt_zed_live_vio (stdin records) -> poses on stdout
"""
import argparse
import json
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zedmac.calib import load_rectifier  # noqa: E402
from zedmac.capture import ZedCapture  # noqa: E402
from zedmac.imu import ZedImu  # noqa: E402
from zedmac.zedctl import ZedVendor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--calib", default=str(ROOT / "calib/basalt_calib_zedm_640.json"))
ap.add_argument("--config", default=str(ROOT / "calib/basalt_config_640.json"))
ap.add_argument("--bin", default=str(ROOT / "third_party/basalt/build/release/basalt_zed_live_vio"))
ap.add_argument("--latency-ms", type=float, default=71.0, help="VSYNC->host frame latency prior (locks frame<->sync)")
ap.add_argument("--clock-scale", type=float, default=1.004629, help="MCU tick correction (measured +4629 ppm)")
ap.add_argument("--headless", type=float, default=0, help="run N seconds without a window")
ap.add_argument("--save", default=None, help="TUM trajectory output path")
a = ap.parse_args()

for f, hint in ((a.calib, "run: .venv/bin/python scripts/setup_calib.py"), (a.bin, "run: ./setup.sh --vio")):
    if not Path(f).exists():
        sys.exit(f"missing {f}\n{hint}")
EYE_W, EYE_H, SHIFT = 640, 360, 30
rect = load_rectifier(2560)

# ---------------- poses from the bridge ----------------
poses = deque(maxlen=100000)   # (t_ns, p(3), q(4), v(3))
pose_lock = threading.Lock()


def read_poses(stdout):
    for raw in stdout:
        f = raw.split()
        if not f or f[0] != b"P":
            continue
        v = np.array(f[2:], float)
        with pose_lock:
            poses.append((int(f[1]), v[0:3], v[3:7], v[7:10]))


# ---------------- start sensors ----------------
imu_q: queue.Queue = queue.Queue()
cap = ZedCapture(2560, 30)
cap.read(10)
vend = ZedVendor()
print("VSYNC ->", [hex(r) for r in vend.enable_imu_sync()])
vend.close()
imu = ZedImu(out_queue=imu_q)

print("hold the camera still (gyro bias)...", flush=True)
time.sleep(2.0)
warm = []
while not imu_q.empty():
    warm.append(imu_q.get())
g = np.array([s.gyro for s in warm[-1200:]])
bias = g.mean(0)
motion = np.degrees(np.linalg.norm(g - bias, axis=1).mean())
print(f"gyro bias {np.degrees(bias).round(3)} deg/s (motion level {motion:.2f} deg/s{' -- was it moving?' if motion > 1 else ''})")
calib = json.loads(Path(a.calib).read_text())
calib["value0"]["calib_gyro_bias"][:3] = bias.tolist()
tmp_calib = Path(tempfile.mkdtemp()) / "calib.json"
tmp_calib.write_text(json.dumps(calib))
T_ic = np.eye(4)
c0 = calib["value0"]["T_imu_cam"][0]
T_ic[:3, :3] = Rot.from_quat([c0["qx"], c0["qy"], c0["qz"], c0["qw"]]).as_matrix()
T_ic[:3, 3] = [c0["px"], c0["py"], c0["pz"]]

proc = subprocess.Popen([a.bin, str(tmp_calib), a.config], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
threading.Thread(target=read_poses, args=(proc.stdout,), daemon=True).start()
out = proc.stdin

# ---------------- sync state ----------------
events: dict[int, int] = {}      # sync_count -> exposure t_ns (scaled MCU clock)
ev_order: deque = deque(maxlen=600)
last_count = None
host_off = np.inf                # min(t_host - t_scaled): MCU->host offset (lower envelope)
first_imu_ns = None
k_locked, k_votes = None, deque(maxlen=15)
pending: deque = deque()         # (frame_idx, host_arrival, gray_pair)
stats = dict(imu=0, frames=0, dropped=0)


def scaled_ns(s):
    return int(s.t_dev * a.clock_scale * 1e9)


def pump_imu():
    global last_count, host_off, first_imu_ns
    while True:
        try:
            s = imu_q.get_nowait()
        except queue.Empty:
            return
        t = scaled_ns(s)
        host_off = min(host_off, s.t_host - t * 1e-9)
        if last_count is not None and s.sync_count > last_count:
            events[s.sync_count] = t
            ev_order.append(s.sync_count)
            while len(events) > 600:
                events.pop(ev_order.popleft(), None)
        last_count = s.sync_count
        if k_locked is None:
            continue  # don't stream IMU until frames can follow (avoids a long IMU-only backlog in Basalt)
        if first_imu_ns is None:
            first_imu_ns = t
        out.write(struct.pack("<Bq6d", 1, t, *s.gyro, *s.acc))
        stats["imu"] += 1


def prep(sbs):
    l, r = rect.split_rectify(sbs)
    gl = cv2.resize(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), (EYE_W, EYE_H), interpolation=cv2.INTER_AREA)
    gr = cv2.resize(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), (EYE_W, EYE_H), interpolation=cv2.INTER_AREA)
    gr = np.hstack([np.zeros((EYE_H, SHIFT), np.uint8), gr[:, :EYE_W - SHIFT]])
    return gl, gr


def flush_frames():
    """Send pending frames whose exposure event is known (in order)."""
    global k_locked
    while pending:
        idx, arr, (gl, gr) = pending[0]
        if k_locked is None:
            if not ev_order or not np.isfinite(host_off):
                return
            t_exp_est = (arr - a.latency_ms / 1e3 - host_off) * 1e9
            cs = np.array(ev_order)
            ts = np.array([events[c] for c in cs])
            k_votes.append(idx - int(cs[np.argmin(np.abs(ts - t_exp_est))]))
            if len(k_votes) == k_votes.maxlen and len(set(k_votes)) == 1:
                k_locked = k_votes[0]
                print(f"frame<->sync locked: idx = sync_count + {k_locked}", flush=True)
            pending.popleft()
            continue
        c = idx - k_locked
        if c in events:
            t = events[c]
            pending.popleft()
            if first_imu_ns is None or t <= first_imu_ns + 50_000_000:
                continue  # need IMU before the first frame
            out.write(struct.pack("<BqHH", 2, t, EYE_W, EYE_H) + gl.tobytes() + gr.tobytes())
            stats["frames"] += 1
        elif last_count is not None and c < last_count - 300:
            pending.popleft(); stats["dropped"] += 1
        else:
            return  # wait for the IMU to deliver this frame's sync event


# ---------------- visualization ----------------
view = dict(center=np.zeros(2), scale=150.0)   # px per meter
trail_start = [0]


def draw_traj(size=360):
    img = np.full((size, size, 3), 24, np.uint8)
    with pose_lock:
        P = np.array([p for _, p, _, _ in list(poses)[trail_start[0]:]]) if len(poses) > trail_start[0] else None
        last = poses[-1] if poses else None
    if P is not None and len(P):
        span = max(0.5, np.ptp(P[:, :2], axis=0).max() * 1.3)
        view["scale"] = 0.9 * view["scale"] + 0.1 * (size / span)
        view["center"] = 0.9 * view["center"] + 0.1 * (P[:, :2].min(0) + P[:, :2].max(0)) / 2
    s, cxy = view["scale"], view["center"]
    to_px = lambda xy: (int(size / 2 + (xy[0] - cxy[0]) * s), int(size / 2 - (xy[1] - cxy[1]) * s))
    step = 0.1 if s > 600 else 0.5 if s > 120 else 1.0
    for gx in np.arange(np.floor((cxy[0] - size / 2 / s) / step) * step, cxy[0] + size / 2 / s, step):
        x = to_px((gx, 0))[0]; cv2.line(img, (x, 0), (x, size), (45, 45, 45), 1)
    for gy in np.arange(np.floor((cxy[1] - size / 2 / s) / step) * step, cxy[1] + size / 2 / s, step):
        y = to_px((0, gy))[1]; cv2.line(img, (0, y), (size, y), (45, 45, 45), 1)
    if P is not None and len(P) > 1:
        pts = np.array([to_px(p) for p in P[:, :2]], np.int32)
        cv2.polylines(img, [pts], False, (0, 200, 255), 1, cv2.LINE_AA)
    if last is not None:
        T = np.eye(4); T[:3, :3] = Rot.from_quat(last[2]).as_matrix(); T[:3, 3] = last[1]
        Twc = T @ T_ic
        o = to_px(Twc[:2, 3]); fwd = Twc[:3, 2]
        tip = to_px(Twc[:2, 3] + 0.15 * fwd[:2] / max(1e-6, np.linalg.norm(fwd[:2])) * min(1, np.linalg.norm(fwd[:2]) * 1.5))
        cv2.arrowedLine(img, o, tip, (0, 255, 0), 2, tipLength=0.35)
        cv2.circle(img, o, 4, (0, 255, 0), -1)
    cv2.putText(img, f"top view, grid {step:g} m", (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return img


# ---------------- main loop ----------------
print("streaming to Basalt; move the camera", flush=True)
t0 = time.monotonic()
n = 0
try:
    while True:
        pump_imu()
        try:
            n, arr, sbs = cap.read(timeout=0.5, newer_than=n)
        except TimeoutError:
            continue
        pending.append((n, arr, prep(sbs)))
        pump_imu()
        flush_frames()
        if a.headless:
            if time.monotonic() - t0 > a.headless:
                break
            continue
        gl = pending[-1][2][0] if pending else prep(sbs)[0]
        left = cv2.cvtColor(gl, cv2.COLOR_GRAY2BGR)
        with pose_lock:
            last = poses[-1] if poses else None
            npos = len(poses)
        if last is not None:
            T = np.eye(4); T[:3, :3] = Rot.from_quat(last[2]).as_matrix(); T[:3, 3] = last[1]
            Twc = T @ T_ic
            ypr = Rot.from_matrix(Twc[:3, :3]).as_euler("zyx", degrees=True)
            lag = (time.monotonic() - host_off) * 1e3 - last[0] * 1e-6
            lines = [f"pos  x {Twc[0, 3]:+.3f}  y {Twc[1, 3]:+.3f}  z {Twc[2, 3]:+.3f} m",
                     f"speed {np.linalg.norm(last[3]):.2f} m/s   lag {lag:.0f} ms",
                     f"poses {npos}  frames {stats['frames']}  k {'locked' if k_locked is not None else '...'}"]
        else:
            lines = ["initializing..." if k_locked is None else "waiting for VIO...",
                     f"frames sent {stats['frames']}  imu {stats['imu']}"]
        for i, txt in enumerate(lines):
            cv2.putText(left, txt, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(left, txt, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("zed-mac live VIO", np.hstack([left, draw_traj(EYE_H)]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            with pose_lock:
                trail_start[0] = len(poses)
except KeyboardInterrupt:
    pass
finally:
    try:
        out.close()
    except BrokenPipeError:
        pass
    proc.wait(timeout=10)
    imu.close(); cap.close()
    cv2.destroyAllWindows()
    with pose_lock:
        P = list(poses)
    save = Path(a.save) if a.save else ROOT / "recordings" / f"live_{time.strftime('%Y%m%d-%H%M%S')}_traj.txt"
    save.parent.mkdir(exist_ok=True)
    with open(save, "w") as f:
        for t, p, q, _ in P:
            f.write(f"{t * 1e-9:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}\n")
    path = sum(np.linalg.norm(P[i][1] - P[i - 1][1]) for i in range(1, len(P))) if len(P) > 1 else 0
    print(f"done: {stats['frames']} frames, {stats['imu']} imu, {len(P)} poses, path {path:.2f} m, "
          f"dropped {stats['dropped']} -> {save}")
