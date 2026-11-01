"""Run basalt_vio headless with config overrides and report landmark counts. usage: basalt_sweep.py <euroc_dir>"""
import json, subprocess, sys, tempfile, shutil
from pathlib import Path
import numpy as np, ubjson
E = Path(sys.argv[1]).resolve()
BIN = Path(__file__).resolve().parent.parent / "third_party/basalt/build/release/basalt_vio"
base = json.loads((E / "basalt_config.json").read_text())
variants = {
    "as configured": {},
    "levels 3": {"config.optical_flow_levels": 3},
    "levels 3, grid 30": {"config.optical_flow_levels": 3, "config.optical_flow_detection_grid_size": 30},
}
for name, over in variants.items():
    d = Path(tempfile.mkdtemp())
    cfg = json.loads(json.dumps(base)); cfg["value0"].update(over)
    (d / "cfg.json").write_text(json.dumps(cfg))
    r = subprocess.run([str(BIN), "--dataset-path", str(E), "--dataset-type", "euroc", "--cam-calib", str(E / "basalt_calib.json"),
                        "--config-path", str(d / "cfg.json"), "--show-gui", "0", "--save-trajectory", "tum"],
                       cwd=d, capture_output=True, text=True)
    st = ubjson.loadb((d / "stats_all.ubjson").read_bytes())
    lms = np.array(st["num_lms"], float)
    T = np.loadtxt(d / "trajectory.txt"); t = T[:, 0] - T[0, 0]; p = T[:, 1:4]
    path = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    static = np.linalg.norm(p[t < 3] - p[0], axis=1).max()
    print(f"{name:28s} lms mean {lms.mean():5.1f}  first5s {lms[:150].mean():5.1f}  zero-frames {(lms == 0).sum():4d}  "
          f"path {path:5.2f} m  static-drift {static * 1000:4.0f} mm  end {np.round(p[-1], 2)}", flush=True)
    shutil.rmtree(d)
