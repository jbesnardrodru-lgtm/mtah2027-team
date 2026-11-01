"""What is a recovered ball worth, and what does a WRONG recovered ball cost?  (lab, measurable)
For each external test match: the usual degraded ball, then recover a fraction `frac` of every gap from both
edges inward (like our detector's forward/backward walks); a fraction `wrong` of the recovered stretches gets
a position offset by 5-25 m (detector locked onto the wrong object). Score with the v7 model (gradient
boosting), so only the ball input changes.
    python -m possession.exp_recover fit            # train and save one model per held-out match
    python -m possession.exp_recover <frac> <wrong> # e.g. 0.2 0.1
"""
import os, sys, pickle
import numpy as np
from possession.external import load_game
from possession.noise import degrade
from possession.features import build
from possession.model import fit, proba, decode, graded_segments, segments_from_period
from possession.featcache import get
from possession.exp_dead import columns
from possession.lab import score

CACHE = "data/cache"


def recovered_ball(g, frac, wrong, seed=0):
    P, ball, team, label, period = load_game(g)
    Pn, bn = degrade(P, ball, label, seed=g, period=period)        # identical to the lab's G<g> cache
    rng = np.random.default_rng(1000 + seed + g)
    gap = np.isnan(bn[:, 0]) & ~np.isnan(ball[:, 0]) & (label != "X")
    e = np.diff(np.r_[0, gap.astype(int), 0]); starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    out = bn.copy()
    for s, t in zip(starts, ends):
        k = int(round((t - s) * frac / 2))
        for a, b in ((s, s + k), (t - k, t)):
            if b <= a:
                continue
            out[a:b, :2] = ball[a:b, :2]
            if rng.random() < wrong:                                 # this stretch tracks the wrong object
                ang = rng.uniform(0, 2 * np.pi); dist = rng.uniform(5, 25)
                out[a:b, 0] = np.clip(out[a:b, 0] + dist * np.cos(ang) / 105, 0, 1)
                out[a:b, 1] = np.clip(out[a:b, 1] + dist * np.sin(ang) / 68, 0, 1)
    return Pn, out, team, label, period


if __name__ == "__main__":
    cols = columns(get("G1")[1], "v3")
    if sys.argv[1] == "fit":
        for t in (1, 2, 3):
            tr = [k for k in (1, 2, 3) if k != t]
            clf = fit([get(f"G{k}")[0][:, cols] for k in tr], [get(f"G{k}")[2] for k in tr], step=4)
            pickle.dump(clf, open(f"{CACHE}/gb_fold{t}.pkl", "wb"))
        print("saved 3 fold models"); sys.exit()
    frac, wrong = float(sys.argv[1]), float(sys.argv[2])
    tests = [int(x) for x in sys.argv[3:]] or [1, 2, 3]
    for t in tests:
        clf = pickle.load(open(f"{CACHE}/gb_fold{t}.pkl", "rb"))
        Pn, b, team, label, period = recovered_ball(t, frac, wrong)
        F, _ = build(Pn, b, team, segments_from_period(period))
        sc = score(label, decode(proba(clf, F[:, cols]), graded_segments(label), 20, 80))
        g = label != "X"
        print(f"frac {frac:.2f} wrong {wrong:.2f} G{t}: ball missing {np.mean(np.isnan(b[g,0])):.1%}  spellF1 {sc[0]:.3f}  macroF1 {sc[1]:.3f}", flush=True)
