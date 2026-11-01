"""Live ZED Mini viewer: rectified stereo + depth (SGBM or neural), no Stereolabs SDK.

usage:
  .venv/bin/python scripts/live.py                      # SGBM depth, 1280x720/eye @30
  .venv/bin/python scripts/live.py --neural realtime    # RAFT-Stereo realtime on MPS
  .venv/bin/python scripts/live.py --neural middlebury --scale 0.5
  .venv/bin/python scripts/live.py --headless 30        # no window; save last frame to samples/

keys: q quit | m cycle view (depth / rectified+lines / raw) | s save snapshot | click: print depth at pixel
"""
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zedmac.calib import load_rectifier  # noqa: E402
from zedmac.capture import ZedCapture  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=2560, choices=[1344, 2560, 3840, 4416])
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--neural", default=None, help="RAFT-Stereo preset: realtime|middlebury|eth3d|rvc")
ap.add_argument("--scale", type=float, default=0.5, help="neural input scale (speed/quality)")
ap.add_argument("--iters", type=int, default=None)
ap.add_argument("--max-depth", type=float, default=2.0, help="colormap range (m)")
ap.add_argument("--headless", type=int, default=0, help="process N frames without a window")
a = ap.parse_args()

rect = load_rectifier(a.width)
w, h = rect.size

if a.neural:
    from zedmac.neural import RaftStereo
    model = RaftStereo(a.neural, scale=a.scale, iters=a.iters)
    def depth_fn(l, r):
        d = model(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        return rect.depth_from_disparity(d)
    label = f"RAFT {a.neural} x{a.scale}"
else:
    nd = 16 * max(4, int(w / 1280 * 16))  # 256 at HD
    sg = cv2.StereoSGBM_create(0, nd, 5, P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                               speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    def depth_fn(l, r):
        gl, gr = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        return rect.depth_from_disparity(sg.compute(gl, gr).astype(np.float32) / 16)
    label = "SGBM"


def colorize(Z):
    v = np.clip(np.nan_to_num(Z, nan=0) / a.max_depth, 0, 1)
    c = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # near = red
    c[~np.isfinite(Z)] = 0
    return c


state = {"Z": None, "mode": 0}


def on_mouse(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN and state["Z"] is not None:
        Z = state["Z"]
        xx, yy = int(x % (Z.shape[1] * disp_scale) / disp_scale), int(y / disp_scale)
        print(f"depth @({xx},{yy}) = {np.nanmedian(Z[max(0, yy - 3):yy + 4, max(0, xx - 3):xx + 4]):.3f} m")


disp_scale = min(1.0, 1280 / (2 * w))
win = "zed-mac"
if not a.headless:
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

with ZedCapture(a.width, a.fps) as cap:
    n, k, ema = 0, 0, None
    while True:
        n, ts, sbs = cap.read(newer_than=n)
        t0 = time.perf_counter()
        l, r = rect.split_rectify(sbs)
        Z = depth_fn(l, r)
        dt = time.perf_counter() - t0
        ema = dt if ema is None else 0.9 * ema + 0.1 * dt
        state["Z"] = Z
        if state["mode"] == 0:
            vis = np.hstack([l, colorize(Z)])
        elif state["mode"] == 1:
            vis = np.hstack([l, r]).copy()
            for y in range(0, h, h // 20):
                cv2.line(vis, (0, y), (2 * w, y), (0, 255, 0), 1)
        else:
            vis = sbs
        vis = cv2.resize(vis, None, fx=disp_scale, fy=disp_scale)
        cv2.putText(vis, f"{label}  {ema * 1000:.0f} ms/frame  valid {np.isfinite(Z).mean() * 100:.0f}%",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        k += 1
        if a.headless:
            if k >= a.headless:
                out = ROOT / "samples" / f"live_{label.split()[0].lower()}.png"
                cv2.imwrite(str(out), vis)
                np.save(ROOT / "samples" / "live_depth.npy", Z)
                print(f"{label}: {ema * 1000:.1f} ms/frame avg, valid {np.isfinite(Z).mean() * 100:.0f}%, wrote {out}")
                break
            continue
        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            state["mode"] = (state["mode"] + 1) % 3
        if key == ord("s"):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(str(ROOT / "samples" / f"snap_{stamp}_sbs.png"), sbs)
            np.save(ROOT / "samples" / f"snap_{stamp}_depth.npy", Z)
            print("saved", stamp)
cv2.destroyAllWindows()
