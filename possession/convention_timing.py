"""When do labels switch, relative to what the tracking shows? (understanding the definition; no model scored)
  stoppage start  vs  ball crossing the pitch boundary (nearest crossing within +/- 5 s)
  stoppage end    vs  ball starting to move (speed rises above 3 m/s, nearest within +/- 5 s)
  team change     vs  ball direction/speed change (nearest 'kick' within +/- 2 s)
Official: public window (frames 11185-18684) on our tactical tracking. Lab: Metrica games 1-3 on their clean tracking."""
import numpy as np, pandas as pd
from possession.external import load_game
from possession.cache import load

FPS = 25


def ball_signals(ball, L, W):
    x = (ball[:, 0] - 0.5) * L; y = (ball[:, 1] - 0.5) * W
    out = np.maximum(np.abs(x) - L / 2, np.abs(y) - W / 2) > 0
    v = np.full(len(x), np.nan)
    v[2:-2] = np.hypot(x[4:] - x[:-4], y[4:] - y[:-4]) * FPS / 4
    vx = np.full(len(x), np.nan); vy = vx.copy()
    vx[2:-2] = (x[4:] - x[:-4]) * FPS / 4; vy[2:-2] = (y[4:] - y[:-4]) * FPS / 4
    dv = np.full(len(x), np.nan); dv[3:-3] = np.hypot(vx[6:] - vx[:-6], vy[6:] - vy[:-6])
    cross = np.flatnonzero(out[1:] & ~out[:-1] & ~np.isnan(x[:-1])) + 1          # ball leaves the pitch
    # a kick from a STILL ball: < 1 m/s for the previous 0.5 s, then > 3 m/s (the restart signature)
    vv = np.nan_to_num(v, nan=99.0)
    still = np.convolve((vv < 1).astype(float), np.ones(12), "full")[:len(vv)] >= 12
    start = np.flatnonzero((v[1:] > 3) & still[:-1]) + 1
    d = np.nan_to_num(dv)
    kick = np.flatnonzero((d > 4) & (d >= np.roll(d, 1)) & (d >= np.roll(d, -1)) & (d >= np.roll(d, 2)) & (d >= np.roll(d, -2)))
    return cross, start, kick


def nearest(events, t, tol):
    if not len(events):
        return None
    j = np.argmin(np.abs(events - t)); d = events[j] - t
    return -d if abs(d) <= tol else None      # positive = label switches AFTER the physical event


def offsets(label, ball, L, W, lo=0, hi=None):
    hi = hi or len(label)
    cross, start, kick = ball_signals(ball, L, W)
    res = {"stop start vs ball out": [], "stop end vs kick from still": [], "team change vs kick": []}
    for t in range(max(lo, 1), hi):
        a, b = label[t - 1], label[t]
        if a == b or "X" in (a, b):
            continue
        if b == "D" and a in "HA":
            o = nearest(cross, t, 125)
            if o is not None: res["stop start vs ball out"].append(o / FPS)
        elif a == "D" and b in "HA":
            o = nearest(start, t, 125)
            if o is not None: res["stop end vs kick from still"].append(o / FPS)
        elif a in "HA" and b in "HA":
            o = nearest(kick, t, 50)
            if o is not None: res["team change vs kick"].append(o / FPS)
    return res


def fmt(v):
    v = np.array(v)
    return f"n {len(v):>3}  median {np.median(v):+.2f}s  p25/p75 {np.percentile(v,25):+.2f}/{np.percentile(v,75):+.2f}" if len(v) else "n   0"


gt = pd.read_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv").label.values
_, tball, _ = load("tactical")
off_o = offsets(gt, tball, 105.3, 68.0, 11185, 18685)
off_l = {k: [] for k in off_o}
for g in (1, 2, 3):
    _, ball, _, label, _ = load_game(g)
    for k, v in offsets(label, ball, 105.0, 68.0).items():
        off_l[k] += v
print("positive = the label switches AFTER the physical event")
for k in off_o:
    print(f"{k:<24} official: {fmt(off_o[k]):<48} lab (Metrica events): {fmt(off_l[k])}")
