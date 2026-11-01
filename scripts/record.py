"""Record a ZED Mini stereo + IMU sequence (raw logs), for VIO / calibration.

usage: .venv/bin/python scripts/record.py [--seconds 60] [--out recordings/<stamp>]

Writes:
  <out>/frames/<idx:06d>.png      rectified left|right grayscale, side by side (eye size = --eye-width)
  <out>/frames.csv                idx, host_arrival_s
  <out>/imu.csv                   t_dev_s, t_host_s, gx, gy, gz (rad/s), ax, ay, az (m/s^2), sync_flag, sync_count
  <out>/meta.json                 calibration of the rectified pair, settings
Then: scripts/to_euroc.py <out>   (associates frames <-> IMU sync pulses, writes EuRoC layout)
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zedmac.calib import load_rectifier, find_conf  # noqa: E402
from zedmac.capture import ZedCapture  # noqa: E402
from zedmac.imu import ZedImu  # noqa: E402
from zedmac.zedctl import ZedVendor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--out", default=None)
ap.add_argument("--width", type=int, default=2560)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--eye-width", type=int, default=640, help="saved per-eye width (rectified, downscaled)")
ap.add_argument("--preview", action="store_true")
a = ap.parse_args()

out = Path(a.out or ROOT / "recordings" / time.strftime("%Y%m%d-%H%M%S"))
(out / "frames").mkdir(parents=True, exist_ok=True)
rect = load_rectifier(a.width)
w, h = rect.size
s = a.eye_width / w
ew, eh = a.eye_width, round(h * s)

meta = {
    "camera": "ZED-M", "calib_file": str(find_conf()), "capture": f"{a.width}x{rect.size[1]}@{a.fps}",
    "eye_size": [ew, eh], "model": "pinhole (rectified, zero distortion)",
    "fx": rect.P1[0, 0] * s, "fy": rect.P1[1, 1] * s, "cx": rect.P1[0, 2] * s, "cy": rect.P1[1, 2] * s,
    "baseline_m": rect.baseline_m,
    "note": "cam1 = cam0 translated +baseline along x (rectified). imu axes != camera axes; estimate R_cam_imu.",
}
(out / "meta.json").write_text(json.dumps(meta, indent=2))

q: queue.Queue = queue.Queue(maxsize=256)


def writer():
    while True:
        item = q.get()
        if item is None:
            return
        idx, img = item
        cv2.imwrite(str(out / "frames" / f"{idx:06d}.png"), img)


wt = threading.Thread(target=writer, daemon=True)
wt.start()
frames_log = []
with ZedCapture(a.width, a.fps) as cap:
    cap.read(10)
    vend = ZedVendor()
    regs = vend.enable_imu_sync()
    vend.close()
    print(f"VSYNC regs -> {[hex(r) for r in regs]}; recording {a.seconds:.0f}s to {out}")
    with ZedImu() as imu:
        time.sleep(0.3)
        n, t0, dropped = 0, time.monotonic(), 0
        while time.monotonic() - t0 < a.seconds:
            prev = n
            n, th, sbs = cap.read(newer_than=n)
            dropped += n - prev - 1 if prev else 0
            l, r = rect.split_rectify(sbs)
            g = np.hstack([cv2.resize(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), (ew, eh), interpolation=cv2.INTER_AREA),
                           cv2.resize(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), (ew, eh), interpolation=cv2.INTER_AREA)])
            frames_log.append((n, th))
            try:
                q.put_nowait((n, g))
            except queue.Full:
                print("writer behind, frame dropped", file=sys.stderr)
            if a.preview:
                cv2.imshow("record", g)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if len(frames_log) % (a.fps * 5) == 0:
                print(f"  {time.monotonic() - t0:5.1f}s  frames {len(frames_log)}  skipped {dropped}  imu {len(imu.samples)}")
        time.sleep(0.2)
        samples = imu.snapshot()
q.put(None)
wt.join()

np.savetxt(out / "frames.csv", np.array(frames_log), delimiter=",", header="idx,host_arrival_s", fmt=["%d", "%.6f"])
arr = np.array([[x.t_dev, x.t_host, *x.gyro, *x.acc, x.frame_sync, x.sync_count] for x in samples])
np.savetxt(out / "imu.csv", arr, delimiter=",", header="t_dev_s,t_host_s,gx,gy,gz,ax,ay,az,sync_flag,sync_count",
           fmt=["%.9f", "%.6f"] + ["%.6f"] * 6 + ["%d", "%d"])
print(f"done: {len(frames_log)} frames ({dropped} skipped by capture), {len(samples)} imu samples -> {out}")
