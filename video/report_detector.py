"""Performance report for our ball detector (the yellow ring), no labels used.
Coverage = share of lost-ball frames (tracking missing) that our detector fills.
Accuracy = distance to the broadcast ball where the broadcast also sees it (independent referee).
    python video/report_detector.py
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, W = 105.3, 68.0
t = np.load(f"{ROOT}/data/cache/tactical.npz"); ball, P = t["ball"], t["P"]
v = np.load(f"{ROOT}/data/cache/ball_video_cpu_v2.npz")["ball"]
b = np.load(f"{ROOT}/data/cache/broadcast_mapped.npz")["ball"]
lab = pd.read_csv(f"{ROOT}/evaluation/results/team_v11.csv").label.values
n = len(ball); g = np.zeros(n, bool); g[11185:90652] = True; g[113742:190198] = True
lost = g & np.isnan(ball[:, 0])
filled = lost & ~np.isnan(v[:, 0])
ref = filled & ~np.isnan(b[:, 0])
err = np.full(n, np.nan)
err[ref] = np.hypot((v[ref, 0] - b[ref, 0]) * L, (v[ref, 1] - b[ref, 1]) * W)

# situations, defined at each frame from the best ball estimate (ours, else last seen tracking ball)
idx = np.where(~np.isnan(ball[:, 0]), np.arange(n), 0); np.maximum.accumulate(idx, out=idx)
last = ball[idx]
est = np.where(np.isnan(v[:, :1]), last[:, :2], v[:, :2])
d = np.hypot(P[:, :, 0] * L - est[:, :1] * L, P[:, :, 1] * W - est[:, 1:2] * W)
crowd = np.nansum(d < 5, 1)
x = est[:, 0] * L; y = est[:, 1] * W
sp = np.full(n, np.nan); sp[5:] = np.hypot((v[5:, 0] - v[:-5, 0]) * L, (v[5:, 1] - v[:-5, 1]) * W) * 5
since = np.arange(n) - np.maximum.accumulate(np.where(~np.isnan(ball[:, 0]), np.arange(n), 0))
groups = {
    "ALL lost-ball frames": np.ones(n, bool),
    "live play (model: H/A)": np.isin(lab, ["H", "A"]),
    "stoppage (model: D)": lab == "D",
    "open (0-1 players within 5 m)": crowd <= 1,
    "crowded (4+ players within 5 m)": crowd >= 4,
    "near touchline (camera side)": y > W - 12,
    "far touchline": y < 12,
    "central strip": (y >= 12) & (y <= W - 12),
    "in a penalty box": (np.abs(x - L / 2) > L / 2 - 16.5) & (np.abs(y - W / 2) < 20.15),
    "first 1 s of a gap": since <= 25,
    "deep in a gap (> 5 s)": since > 125,
}
print(f"{'situation':<34}{'lost frames':>12}{'filled':>9}{'checked':>9}{'within 2 m':>12}{'within 5 m':>12}{'median err':>12}")
for name, m in groups.items():
    lm = lost & m; fm = filled & m; rm = ref & m
    e = err[rm]
    print(f"{name:<34}{lm.sum():>12}{fm.sum()/max(lm.sum(),1):>9.0%}{rm.sum():>9}"
          f"{np.mean(e<2) if len(e) else np.nan:>12.0%}{np.mean(e<5) if len(e) else np.nan:>12.0%}{np.median(e) if len(e) else np.nan:>10.1f} m")
fm = filled & ~np.isnan(sp)
print(f"\nball speed when filled: slow (<3 m/s) {np.mean(sp[fm]<3):.0%}, medium {np.mean((sp[fm]>=3)&(sp[fm]<12)):.0%}, fast (>12 m/s) {np.mean(sp[fm]>=12):.0%}")
np.save(f"{ROOT}/data/cache/detector_err.npy", err)
