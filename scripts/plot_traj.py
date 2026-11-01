"""Plot a Basalt TUM trajectory (3 views) and print drift stats. usage: plot_traj.py <trajectory.txt> [out.png]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

T = np.loadtxt(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else "samples/vio_traj.png"
t, p, q = T[:, 0] - T[0, 0], T[:, 1:4], T[:, 4:8]
step = np.linalg.norm(np.diff(p, axis=0), axis=1)
d = np.linalg.norm(p[-1] - p[0])
rot = np.degrees((Rotation.from_quat(q[-1]) * Rotation.from_quat(q[0]).inv()).magnitude())
print(f"{len(T)} poses, {t[-1]:.1f} s, path {step.sum():.2f} m, start->end {d:.3f} m / {rot:.1f} deg "
      f"({100 * d / step.sum():.2f}% of path)")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, (i, j, n) in zip(ax, [(0, 1, "top view (x-y), gravity = -z"), (0, 2, "side (x-z)"), (1, 2, "side (y-z)")]):
    sc = a.scatter(p[:, i], p[:, j], c=t, s=2, cmap="viridis")
    a.plot(*p[0, [i, j]], "go", ms=9, label="start")
    a.plot(*p[-1, [i, j]], "rx", ms=11, mew=3, label="end")
    a.set_title(n); a.set_xlabel("m"); a.axis("equal"); a.grid(alpha=.3)
ax[0].legend()
fig.colorbar(sc, ax=ax, label="time (s)")
fig.savefig(out, dpi=85, bbox_inches="tight")
print("wrote", out)
