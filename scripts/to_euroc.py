"""Associate recorded frames with IMU frame-sync pulses and write a EuRoC-layout dataset.

usage: .venv/bin/python scripts/to_euroc.py recordings/<stamp> [--out <dir>]

Timing model: the IMU MCU increments sync_count at each camera exposure; the IMU sample where it increments
gives the exposure time on the MCU clock (+-1.25 ms). Frame idx (ffmpeg order) = sync_count + k for one constant
k per recording; k is chosen so host arrival latency (arrival - exposure mapped to host clock) is most consistent.
Output timestamps: MCU clock rescaled to true seconds via the fitted MCU->host map (its tick is ~0.46% off nominal), in ns.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("rec")
ap.add_argument("--out", default=None)
ap.add_argument("--latency-ms", type=float, default=71.0,
                help="expected VSYNC->host latency; resolves the 1-frame ambiguity (71 ms measured for 2560x720@30 "
                     "by gyro/visual correlation; verify with calib_cam_imu.py, its offset should be ~0)")
ap.add_argument("--right-shift", type=int, default=30,
                help="shift cam1 image right by N px (cam1 cx += N) so close-range disparities start near zero; "
                     "Basalt's zero-initialized stereo flow fails beyond ~15 px (measured 7-33%% match rate at 25-40 px)")
a = ap.parse_args()
rec = Path(a.rec)
out = Path(a.out or rec / "euroc")

imu = np.loadtxt(rec / "imu.csv", delimiter=",")
frames = np.loadtxt(rec / "frames.csv", delimiter=",", ndmin=2)
t_dev, t_host, cnt = imu[:, 0], imu[:, 1], imu[:, 9].astype(np.int64)

# --- MCU -> host clock map: host receipt = a*t_dev + b + latency(>=0); fit the lower envelope
A = np.polyfit(t_dev, t_host, 1)
for _ in range(5):
    res = t_host - np.polyval(A, t_dev)
    keep = res < np.percentile(res, 5)
    A = np.polyfit(t_dev[keep], t_host[keep], 1)
res = t_host - np.polyval(A, t_dev)
print(f"MCU clock rate vs host: {(A[0] - 1) * 1e6:+.0f} ppm; IMU USB latency p50 {np.median(res - res.min()) * 1e3:.2f} ms")

gap = np.diff(t_dev)
if (gap > 0.01).any():
    for i in np.nonzero(gap > 0.01)[0]:
        print(f"WARNING: IMU gap {gap[i] * 1e3:.0f} ms at t={t_dev[i] - t_dev[0]:.1f}s -- VIO across this gap is unreliable")

# --- exposure events: first sample at which sync_count increments
inc = np.nonzero(np.diff(cnt) > 0)[0] + 1
ev_count, ev_tdev = cnt[inc], t_dev[inc]
gaps = np.diff(ev_count)
print(f"sync events: {len(inc)} (counts {ev_count[0]}..{ev_count[-1]}, multi-step gaps: {(gaps > 1).sum()})")
ev_host = np.polyval(A, ev_tdev)
count_to = dict(zip(ev_count.tolist(), zip(ev_host.tolist(), ev_host.tolist())))  # (exposure time, same on host clock)

fidx, farr = frames[:, 0].astype(np.int64), frames[:, 1]
# seed k from timing: nearest exposure event preceding each frame's arrival, then refine around the mode
prev_ev = np.searchsorted(ev_host, farr) - 1
ok = prev_ev >= 0
k_seed = int(np.bincount((fidx[ok] - ev_count[prev_ev[ok]]) - (fidx[ok] - ev_count[prev_ev[ok]]).min()).argmax()
             + (fidx[ok] - ev_count[prev_ev[ok]]).min())
best = None
for k in range(k_seed - 4, k_seed + 5):
    pairs = [(i, count_to[c][0], farr[j] - count_to[c][1]) for j, i in enumerate(fidx) if (c := i - k) in count_to]
    if len(pairs) < 0.8 * len(fidx):
        continue
    lat = np.array([p[2] for p in pairs])
    score = (abs(np.median(lat) - a.latency_ms / 1e3), k)  # std is ~identical for every k; use the latency prior
    if lat.min() > -0.002 and (best is None or score < best[0]):
        best = (score, k, pairs, lat)
if best is None:
    sys.exit("could not associate frames with sync pulses")
(_, k, pairs, lat) = best
print(f"frame idx = sync_count + {k}: matched {len(pairs)}/{len(fidx)} frames; "
      f"capture->host latency {np.median(lat) * 1e3:.1f} ms (std {lat.std() * 1e3:.2f} ms)")

# --- write EuRoC
keep = {n: (out / n).read_text() for n in ("cam_imu.json",) if (out / n).exists()}  # preserve calibration
if out.exists():
    shutil.rmtree(out)
for c in ("cam0", "cam1"):
    (out / "mav0" / c / "data").mkdir(parents=True)
(out / "mav0" / "imu0").mkdir(parents=True)
rows = []
for i, texp, _ in pairs:
    img = cv2.imread(str(rec / "frames" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    ns = int(round(texp * 1e9))
    w = img.shape[1] // 2
    cv2.imwrite(str(out / "mav0/cam0/data" / f"{ns}.png"), img[:, :w])
    right = img[:, w:]
    if a.right_shift:
        right = np.hstack([np.zeros((right.shape[0], a.right_shift), right.dtype), right[:, :w - a.right_shift]])
    cv2.imwrite(str(out / "mav0/cam1/data" / f"{ns}.png"), right)
    rows.append(ns)
for c in ("cam0", "cam1"):
    with open(out / "mav0" / c / "data.csv", "w") as f:
        f.write("#timestamp [ns],filename\n")
        f.writelines(f"{ns},{ns}.png\n" for ns in rows)
with open(out / "mav0/imu0/data.csv", "w") as f:
    f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
    for r, tt in zip(imu, np.polyval(A, t_dev)):
        f.write(f"{int(round(tt * 1e9))},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f},{r[7]:.6f}\n")
for n, txt in keep.items():
    (out / n).write_text(txt)
meta = json.loads((rec / "meta.json").read_text())
meta.update(right_shift_px=a.right_shift, frame_offset_k=k, latency_ms=float(np.median(lat) * 1e3), n_frames=len(rows))
(out / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {len(rows)} stereo frames + {len(imu)} imu samples -> {out}")
