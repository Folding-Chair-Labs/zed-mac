Per-camera files live here (gitignored). Generate with `.venv/bin/python scripts/setup_calib.py`:
- `SN<serial>.conf` — Stereolabs factory calibration, fetched by the camera's serial number
- `basalt_calib_zedm_640.json`, `basalt_config_640.json` — live VIO calibration/config
- `cam_imu.json` — optional refined camera-IMU rotation (`scripts/calib_cam_imu.py <euroc> --install`)
