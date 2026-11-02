"""Upper bound: what would a PERFECT detector of fast / aerial balls be worth?  (lab)
Inside every degraded-ball gap, restore the true ball only on frames where the true ball moves faster than
--speed m/s (a stand-in for balls in the air: long passes, clearances, shots). Same fold models as exp_recover.
    python -m possession.exp_air <speed_m_s> [also_slow_frac]
"""
import sys, pickle
import numpy as np
from possession.external import load_game
from possession.noise import degrade
from possession.features import build
from possession.model import proba, decode, graded_segments, segments_from_period
from possession.featcache import get
from possession.exp_dead import columns
from possession.lab import score


def ball_speed(ball, fps=25):
    xy = ball[:, :2] * np.array([105.0, 68.0])
    v = np.r_[0, np.hypot(*np.diff(xy, axis=0).T)] * fps
    k = 5
    return np.convolve(np.nan_to_num(v), np.ones(k) / k, mode="same")


if __name__ == "__main__":
    thr = float(sys.argv[1])
    cols = columns(get("G1")[1], "v3")
    for t in (1, 2, 3):
        clf = pickle.load(open(f"data/cache/gb_fold{t}.pkl", "rb"))
        P, ball, team, label, period = load_game(t)
        Pn, bn = degrade(P, ball, label, seed=t, period=period)
        gap = np.isnan(bn[:, 0]) & ~np.isnan(ball[:, 0]) & (label != "X")
        sel = gap & (ball_speed(ball) > thr) if thr > 0 else np.zeros_like(gap)
        b = bn.copy(); b[sel, :2] = ball[sel, :2]
        F, _ = build(Pn, b, team, segments_from_period(period))
        sc = score(label, decode(proba(clf, F[:, cols]), graded_segments(label), 20, 80))
        print(f"speed>{thr:g}: G{t} restored {sel.sum()/max(gap.sum(),1):.1%} of gap frames  spellF1 {sc[0]:.3f}", flush=True)
