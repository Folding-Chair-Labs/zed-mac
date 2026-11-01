"""One-time per camera: fetch the factory calibration for the connected ZED Mini and write the live VIO calibration.

usage: .venv/bin/python scripts/setup_calib.py
  -> calib/SN<serial>.conf            (Stereolabs factory calibration, fetched by serial)
  -> calib/basalt_calib_zedm_640.json (rectified 640x360 stereo + IMU; uses calib/cam_imu.json if you ran
                                       scripts/calib_cam_imu.py --install, else the default ZED Mini rotation)
  -> calib/basalt_config_640.json
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zedmac.basalt_cfg import build_calib, build_config, rectified_intrinsics, DEFAULT_R_CAM_IMU  # noqa: E402
from zedmac.calib import load_rectifier  # noqa: E402

CAL = ROOT / "calib"
CAL.mkdir(exist_ok=True)
import hid  # noqa: E402

devs = [d for d in hid.enumerate(0x2B03) if d["serial_number"]]
if not devs:
    sys.exit("No ZED HID device found (2b03:*). Plug the camera into a USB 3 port; if only one device appears, flip the USB-C plug.")
serial = devs[0]["serial_number"]
conf = CAL / f"SN{serial}.conf"
if not conf.exists():
    url = f"https://calib.stereolabs.com/?SN={serial}"
    print(f"fetching factory calibration for serial {serial} ...")
    data = urllib.request.urlopen(url, timeout=30).read().decode()
    if "[STEREO]" not in data:
        sys.exit(f"unexpected response from {url}")
    conf.write_text(data)
print(f"factory calibration: {conf}")

rect = load_rectifier(2560, conf)
ci_path = CAL / "cam_imu.json"
ci = json.loads(ci_path.read_text()) if ci_path.exists() else {}
calib = build_calib(rectified_intrinsics(rect, 640), ci.get("R_cam_imu", DEFAULT_R_CAM_IMU), ci.get("gyro_bias_rad_s", (0, 0, 0)))
(CAL / "basalt_calib_zedm_640.json").write_text(json.dumps(calib, indent=4))
(CAL / "basalt_config_640.json").write_text(json.dumps(build_config(ROOT / "native/basalt/euroc_config.json"), indent=4))
print(f"camera-IMU rotation: {'calibrated (' + str(ci_path) + ')' if ci else 'default ZED Mini (run scripts/calib_cam_imu.py --install to refine)'}")
print("wrote calib/basalt_calib_zedm_640.json and calib/basalt_config_640.json")
