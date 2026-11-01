"""Write Basalt calibration + VIO config for a recorded EuRoC dataset (rectified pinhole stereo + IMU).

usage: .venv/bin/python scripts/make_basalt_calib.py recordings/<stamp>/euroc
uses <euroc>/meta.json (record.py/to_euroc.py) and <euroc>/cam_imu.json (calib_cam_imu.py) if present,
else calib/cam_imu.json, else the default ZED Mini rotation.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zedmac.basalt_cfg import build_calib, build_config, DEFAULT_R_CAM_IMU  # noqa: E402

root = Path(sys.argv[1])
meta = json.loads((root / "meta.json").read_text())
ci_path = next((p for p in (root / "cam_imu.json", ROOT / "calib/cam_imu.json") if p.exists()), None)
ci = json.loads(ci_path.read_text()) if ci_path else {}
calib = build_calib(meta, ci.get("R_cam_imu", DEFAULT_R_CAM_IMU), ci.get("gyro_bias_rad_s", (0, 0, 0)),
                    right_shift=meta.get("right_shift_px", 0))
(root / "basalt_calib.json").write_text(json.dumps(calib, indent=4))
base = ROOT / "native/basalt/euroc_config.json"
(root / "basalt_config.json").write_text(json.dumps(build_config(base), indent=4))
print(f"camera-IMU rotation from {ci_path or 'default'}; wrote {root / 'basalt_calib.json'} and basalt_config.json")
