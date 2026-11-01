#!/usr/bin/env bash
# zed-mac setup (macOS, Apple Silicon).
#   ./setup.sh          Python env + RAFT-Stereo (neural depth)
#   ./setup.sh --vio    also build Basalt from source with the live VIO bridge (~30-60 min first time)
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD

RAFT_COMMIT=6e93ed2169bd858dbb43033988563f3b0bb49506
RAFT_WEIGHTS_URL="https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1"
BASALT_COMMIT=0f3b2b52c807f70ff4e2973ce253c73329eea7bc   # basalt 0.1.7 + fixes (2026-03-22)

need() { command -v "$1" >/dev/null || { echo "missing '$1' -- $2"; exit 1; }; }
need brew "install Homebrew from https://brew.sh"
need uv   "brew install uv"
command -v ffmpeg >/dev/null || brew install ffmpeg

echo "== Python environment"
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

echo "== RAFT-Stereo (MIT, princeton-vl)"
mkdir -p third_party weights/raft-stereo
if [ ! -d third_party/RAFT-Stereo ]; then
  git clone https://github.com/princeton-vl/RAFT-Stereo.git third_party/RAFT-Stereo
  git -C third_party/RAFT-Stereo checkout -q "$RAFT_COMMIT"
fi
if [ ! -f weights/raft-stereo/raftstereo-realtime.pth ]; then
  curl -L --fail -o weights/raft-stereo/models.zip "$RAFT_WEIGHTS_URL"
  (cd weights/raft-stereo && unzip -o -q models.zip && rm -rf models.zip __MACOSX)
fi

if [ "${1:-}" = "--vio" ]; then
  echo "== Basalt (BSD-3) with live bridge"
  need cmake "brew install cmake"; need ninja "brew install ninja"
  if [ ! -d third_party/basalt ]; then
    git clone https://gitlab.com/VladyslavUsenko/basalt.git third_party/basalt
    git -C third_party/basalt checkout -q "$BASALT_COMMIT"
    git -C third_party/basalt submodule update --init --recursive
    git -C third_party/basalt apply "$ROOT/native/basalt/basalt.patch"
  fi
  mkdir -p third_party/vcpkg-triplets && cp native/basalt/arm64-osx.cmake third_party/vcpkg-triplets/
  (cd third_party/basalt \
    && ./thirdparty/vcpkg/bootstrap-vcpkg.sh -disableMetrics \
    && CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake --preset release -DVCPKG_OVERLAY_TRIPLETS="$ROOT/third_party/vcpkg-triplets" \
    && cmake --build --preset release --target basalt_zed_live_vio basalt_vio)
fi

echo
echo "done. next:"
echo "  .venv/bin/python scripts/live.py --neural realtime     # live neural depth"
echo "  .venv/bin/python scripts/setup_calib.py                 # once per camera (needed for VIO)"
echo "  .venv/bin/python scripts/live_vio.py                    # live VIO (needs ./setup.sh --vio)"
