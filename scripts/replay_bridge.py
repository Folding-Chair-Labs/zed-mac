"""Replay a EuRoC recording through basalt_zed_live_vio (same stdin protocol as live), write TUM trajectory.
usage: replay_bridge.py <euroc_dir> <out_traj.txt> [--realtime]"""
import struct, subprocess, sys, time, threading
from pathlib import Path
import cv2, numpy as np
E = Path(sys.argv[1]); out = Path(sys.argv[2]); realtime = "--realtime" in sys.argv
BIN = Path(__file__).resolve().parent.parent / "third_party/basalt/build/release/basalt_zed_live_vio"
imu = np.loadtxt(E / "mav0/imu0/data.csv", delimiter=",", comments="#")
cams = np.loadtxt(E / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
p = subprocess.Popen([str(BIN), str(E / "basalt_calib.json"), str(E / "basalt_config.json")], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
lines = []
th = threading.Thread(target=lambda: lines.extend(p.stdout.readlines())); th.start()
ev = sorted([(int(r[0]), 1, r) for r in imu] + [(int(t) + 71_000_000, 2, int(t)) for t in cams], key=lambda e: e[0])  # frames arrive 71 ms late, like live
t0w, t0d = time.monotonic(), ev[0][0]
for t, typ, r in ev:
    if realtime:
        dt = (t - t0d) * 1e-9 - (time.monotonic() - t0w)
        if dt > 0: time.sleep(dt)
    if typ == 1:
        p.stdin.write(struct.pack("<Bq6d", 1, int(r[0]), *r[1:7]))
    else:
        l = cv2.imread(str(E / f"mav0/cam0/data/{r}.png"), 0); rr = cv2.imread(str(E / f"mav0/cam1/data/{r}.png"), 0)
        p.stdin.write(struct.pack("<BqHH", 2, r, l.shape[1], l.shape[0]) + l.tobytes() + rr.tobytes())
p.stdin.close(); p.wait(); th.join()
with open(out, "w") as f:
    for ln in lines:
        v = ln.split()
        if v and v[0] == b"P":
            f.write(f"{int(v[1]) * 1e-9:.9f} " + " ".join(x.decode() for x in v[2:9]) + "\n")
print(f"{sum(1 for l in lines if l.startswith(b'P'))} poses -> {out} ({time.monotonic() - t0w:.1f} s)")
