"""Benchmark RAFT-Stereo presets on a saved ZED side-by-side frame; writes samples/neural_compare.png.

usage: .venv/bin/python scripts/bench_neural.py [samples/sbs_2560x720.png]
"""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zedmac.calib import load_rectifier  # noqa: E402
from zedmac.neural import RaftStereo  # noqa: E402

sbs = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "samples/sbs_2560x720.png")
rect = load_rectifier(sbs.shape[1])
L, R = rect.split_rectify(sbs)
Lr, Rr = cv2.cvtColor(L, cv2.COLOR_BGR2RGB), cv2.cvtColor(R, cv2.COLOR_BGR2RGB)

# probe points in the 1280x720 rectified left image (same spots SGBM gave watch~33, forearm~25, face~54 cm)
pts = {"watch": (410, 430), "forearm": (800, 400), "shirt": (800, 220), "face": (640, 120)}


def sample(Z):
    return "  ".join(f"{k}={np.nanmedian(Z[y - 8:y + 8, x - 8:x + 8]) * 100:.0f}cm" for k, (x, y) in pts.items())


runs = [("realtime", 0.5, True), ("realtime", 1.0, True), ("middlebury", 0.5, True),
        ("middlebury", 1.0, True), ("middlebury", 1.0, False)]
out = {}
for preset, scale, fp16 in runs:
    m = RaftStereo(preset, scale=scale, fp16=fp16)
    dt = m.benchmark(Lr, Rr, n=5)
    d = m(Lr, Rr)
    Z = rect.depth_from_disparity(d)
    print(f"{preset:10s} scale={scale:.1f} fp16={fp16!s:5s} {dt * 1000:7.1f} ms ({1 / dt:5.1f} Hz)  "
          f"disp[{d.min():.1f},{d.max():.1f}]  {sample(Z)}", flush=True)
    out[(preset, scale, fp16)] = d

vmax = np.percentile(out[("middlebury", 1.0, False)], 99)
tiles = [cv2.resize(L, (640, 360))]
for key in [("realtime", 1.0, True), ("middlebury", 1.0, True)]:
    c = cv2.applyColorMap(np.clip(out[key] / vmax * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c = cv2.resize(c, (640, 360))
    cv2.putText(c, f"{key[0]} x{key[1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    tiles.append(c)
cv2.imwrite("samples/neural_compare.png", np.hstack(tiles))
print("wrote samples/neural_compare.png")
