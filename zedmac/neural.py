"""Neural stereo depth on Apple Silicon (MPS). Currently wraps RAFT-Stereo (MIT, princeton-vl)."""
from __future__ import annotations
import sys
import time
import warnings
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
RAFT_DIR = ROOT / "third_party" / "RAFT-Stereo"
WEIGHTS = ROOT / "weights" / "raft-stereo"
warnings.filterwarnings("ignore", message="torch.meshgrid", category=UserWarning)  # from RAFT-Stereo's coords_grid

_BASE = dict(hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False, corr_levels=4,
             corr_radius=4, n_downsample=2, context_norm="batch", slow_fast_gru=False, n_gru_layers=3,
             mixed_precision=False)
PRESETS = {
    # name: (checkpoint, arg overrides, default iters) -- args from RAFT-Stereo README
    "realtime":   ("raftstereo-realtime.pth", dict(shared_backbone=True, n_downsample=3, n_gru_layers=2, slow_fast_gru=True), 7),
    "middlebury": ("raftstereo-middlebury.pth", {}, 32),
    "eth3d":      ("raftstereo-eth3d.pth", {}, 32),
    "sceneflow":  ("raftstereo-sceneflow.pth", {}, 32),
    "rvc":        ("iraftstereo_rvc.pth", dict(context_norm="instance"), 32),
}


def _import_raft():
    if str(RAFT_DIR) not in sys.path:
        sys.path.insert(0, str(RAFT_DIR))
    import core.raft_stereo as rs  # noqa: E402
    return rs


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RaftStereo:
    """disparity = model(left_rgb, right_rgb) on rectified uint8 HxWx3 RGB images -> left-image disparity (px)."""

    def __init__(self, preset: str = "realtime", iters: int | None = None, fp16: bool = True,
                 scale: float = 1.0, device: torch.device | None = None):
        ckpt, over, default_iters = PRESETS[preset]
        self.device = device or pick_device()
        self.iters = iters or default_iters
        self.scale = scale
        self.preset = preset
        rs = _import_raft()
        dev_type = self.device.type
        # RAFT-Stereo hardcodes torch.cuda.amp.autocast; swap in a device-appropriate autocast.
        rs.autocast = lambda enabled=False: torch.autocast(device_type=dev_type, dtype=torch.float16,
                                                          enabled=enabled and dev_type != "cpu")
        args = Namespace(**{**_BASE, **over, "mixed_precision": fp16})
        model = rs.RAFTStereo(args)
        sd = torch.load(WEIGHTS / ckpt, map_location="cpu", weights_only=True)  # tensors only, no pickle code
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        h, w = left.shape[:2]
        t = lambda im: torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None].float().to(self.device)
        a, b = t(left), t(right)
        if self.scale != 1.0:
            sz = (round(h * self.scale), round(w * self.scale))
            a = F.interpolate(a, size=sz, mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=sz, mode="bilinear", align_corners=False)
        H, W = a.shape[-2:]
        ph, pw = (-H) % 32, (-W) % 32
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        a, b = F.pad(a, pad, mode="replicate"), F.pad(b, pad, mode="replicate")
        _, flow = self.model(a, b, iters=self.iters, test_mode=True)
        disp = -flow[:, :1, pad[2]:pad[2] + H, pad[0]:pad[0] + W]
        if self.scale != 1.0:
            disp = F.interpolate(disp, size=(h, w), mode="bilinear", align_corners=False) / self.scale
        return disp[0, 0].float().cpu().numpy()

    def benchmark(self, left, right, n=10) -> float:
        self(left, right)  # warmup / Metal shader compile
        t0 = time.perf_counter()
        for _ in range(n):
            self(left, right)  # .cpu() inside forces sync
        return (time.perf_counter() - t0) / n
