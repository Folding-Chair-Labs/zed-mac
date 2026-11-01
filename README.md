# zed-mac

**Use a Stereolabs ZED Mini on macOS (Apple Silicon): stereo video, neural depth, IMU, and live visual-inertial odometry. No ZED SDK, no CUDA, no root.**

The official ZED SDK requires an NVIDIA GPU, so it does not run on Macs. This project talks to the camera directly:

| | What you get | How |
|---|---|---|
| Video | Side-by-side stereo, 1344x376@100 up to 4416x1242@15 | The camera is a plain UVC device: AVFoundation via ffmpeg |
| Calibration | Rectified pairs, 0.57 px median row error | Factory calibration fetched by serial; its `k1..k4` are OpenCV **fisheye** coefficients |
| Depth | Dense metric depth, ~28 fps on an M5 Max | RAFT-Stereo on PyTorch MPS (or OpenCV SGBM) |
| IMU | 800 Hz gyro + accel | USB HID, protocol from Stereolabs' open-source `zed-open-capture` |
| Sync | Camera exposure timestamps on the IMU clock, ±1.25 ms | Sensor VSYNC output enabled through the UVC extension unit, sent with IOKit while AVFoundation streams |
| VIO | Live 6-DoF pose at 30 Hz | Basalt (stereo-inertial), fed through a small C++ bridge |

> Not affiliated with or endorsed by Stereolabs. "ZED" is a trademark of Stereolabs.
> Tested on a ZED Mini, macOS 27, Apple M5 Max. ZED 2 / 2i / X are untested; their USB IDs and report formats differ.

## Quick start

```bash
git clone https://github.com/Folding-Chair-Labs/zed-mac.git && cd zed-mac
./setup.sh            # Python env + RAFT-Stereo weights (add --vio to also build Basalt, 30-60 min the first time)
```

Plug the camera into a **USB 3** port. If only `ZED-M Hid Device` shows up, the video side didn't enumerate. The ZED Mini only gets a USB 3 link in one USB-C orientation, so **flip the plug at the camera end**. USB 2 hubs won't work.

```bash
.venv/bin/python scripts/live.py                      # rectified stereo + SGBM depth
.venv/bin/python scripts/live.py --neural realtime    # RAFT-Stereo on the GPU (MPS)
.venv/bin/python scripts/setup_calib.py               # once per camera: fetch factory calibration, write VIO calib
.venv/bin/python scripts/live_vio.py                  # live VIO (hold still ~2 s at start)
```

Viewer keys: `q` quit, `m` cycle views, `s` snapshot, click to print depth. VIO keys: `q` quit (saves the trajectory), `c` clear trail.

## Results (measured)

- **Rectification:** median vertical disparity 0.57 px after rectification, against 10.7 px raw (SIFT matches). Treating the coefficients as radtan instead of fisheye gives 3.6 px.
- **Depth:** SGBM, RAFT-Stereo realtime, and RAFT-Stereo middlebury agree within about 1 cm at 0.4-0.55 m.

  | model | input | time/frame (M5 Max) |
  |---|---|---|
  | realtime | 640x360 | 35 ms |
  | realtime | 1280x720 | 77 ms |
  | middlebury | 640x360 | 212 ms |
  | middlebury | 1280x720 | 562 ms |

- **Time sync:** frame-to-gyro offset +0.3 ms after resolving the frame/sync pairing, estimated by correlating visual rotation with the gyro.
- **VIO drift:** 2.0% of distance (15.9 cm over 8.0 m) and 7.7° over a 60 s handheld loop. Ground truth is the first-to-last-frame relative pose from stereo PnP (`scripts/loop_check.py`).
- **Live VIO at rest:** 2.3 mm drift over 20 s.
- **Live bridge:** replaying a recording through the bridge in real time gives the same trajectory as offline `basalt_vio`.

Probable remaining error sources: the ZED Mini's rolling shutter, which Basalt doesn't model, and the IMU lever arm, which is an estimate.

## Offline pipeline (record → calibrate → VIO)

```bash
.venv/bin/python scripts/record.py --seconds 60            # frames + 800 Hz IMU + sync counters
.venv/bin/python scripts/to_euroc.py recordings/<stamp>    # pair frames with sync pulses -> EuRoC layout
.venv/bin/python scripts/calib_cam_imu.py recordings/<stamp>/euroc --install   # cam-IMU rotation, time offset, gyro bias
.venv/bin/python scripts/make_basalt_calib.py recordings/<stamp>/euroc
third_party/basalt/build/release/basalt_vio --dataset-path recordings/<stamp>/euroc --dataset-type euroc \
    --cam-calib recordings/<stamp>/euroc/basalt_calib.json --config-path recordings/<stamp>/euroc/basalt_config.json
.venv/bin/python scripts/plot_traj.py trajectory.txt
```

For the camera-IMU calibration recording: hold the camera still for 3 s, then rotate it firmly about all three axes for about 20 s while it points at a textured scene. To measure drift, start and end at rest on a marked spot, then run `scripts/loop_check.py`.

## How it works / gotchas found along the way

- **UVC controls without root.** libusb (and librealsense/libuvc) get "access denied" on macOS even as root, because UVCAssistant owns the VideoControl interface. Sending class requests through `IOUSBInterfaceInterface::ControlRequest` on interface 0 works as a normal user, even while AVFoundation is streaming (`zedmac/native/uvcctl.c`).
- **Frame-IMU sync.** Writing `0x3002 |= 0x80` on both image sensors (I2C through the OV580 extension unit, unit 4 / selector 2, 384-byte commands) makes the IMU MCU bump `sync_count` at every exposure. Use the counter, not the `frame_sync` flag, which is often missed. The registers read `0xFF` unless the camera is streaming.
- **IMU quirks on macOS:**
  - The ZED-M HID report ID is `0x00`.
  - The MCU tick is about +4630 ppm off its nominal 39.0625 µs.
  - The stream stops after about 11 s without pings, and the 64-byte form of the ping doesn't count.
  - A few percent of pings block for IOKit's 5 s SET_REPORT timeout, so they run on their own thread.
- **Latency ambiguity.** A static scene can't tell 38 ms of latency from 71 ms (one frame). Correlating gyro with visual rotation picks 71 ms at 2560x720@30, and that's now the default prior.
- **Basalt at close range.** Basalt's stereo flow starts at zero disparity and fails beyond about 15 px. Desk-range scenes have 25-40 px at 640 px width, so only 7-33% of matches came out correct. The right image is shifted 30 px with `cx` compensated, which raised average landmarks from 11 to 74.
- **Building Basalt with Apple clang 21.** Pangolin needs `-Wno-error=unused-result`, set through the overlay triplet in `native/basalt/`. RealSense is dropped because its libusb fails to build. Both are applied by `setup.sh --vio`.

## Layout

```
zedmac/      capture.py (UVC), calib.py (fisheye rectification), imu.py (HID), zedctl.py + uvc.py + native/uvcctl.c
             (UVC extension unit via IOKit), neural.py (RAFT-Stereo on MPS), basalt_cfg.py
scripts/     live.py, live_vio.py, record.py, to_euroc.py, calib_cam_imu.py, setup_calib.py, make_basalt_calib.py,
             plot_traj.py, loop_check.py, bench_neural.py, basalt_sweep.py, vio_sweep.py, replay_bridge.py
native/      zed_live_vio.cpp (Basalt bridge: binary frames/IMU on stdin -> poses on stdout), basalt/ (patch, triplet)
```

## Acknowledgements

- [zed-open-capture](https://github.com/stereolabs/zed-open-capture) (Stereolabs, MIT): HID report layout and OV580 extension-unit command format.
- [Basalt](https://gitlab.com/VladyslavUsenko/basalt) (Usenko, Demmel et al., BSD-3): VIO.
- [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) (Lipson, Teed, Deng, MIT): stereo depth.

None of them are vendored. `setup.sh` fetches them at pinned commits. See `THIRD_PARTY_NOTICES.md`.

## License

MIT, see `LICENSE`.
