"""Consistent test: carrier imputation applied to BOTH training and test matches, GB retrained per fold."""
import sys, numpy as np
from possession.external import load_game
from possession.noise import degrade
from possession.features import build
from possession.model import fit, proba, decode, graded_segments, segments_from_period
from possession.featcache import get
from possession.exp_dead import columns
from possession.lab import score
from possession.carry import impute_carrier

max_s = float(sys.argv[1])
cols = columns(get("G1")[1], "v3")
FE = {}
for g in (1, 2, 3):
    P, ball, team, label, period = load_game(g)
    Pn, bn = degrade(P, ball, label, seed=g, period=period)
    b2, _ = impute_carrier(Pn, bn, team, max_s=max_s)
    F, _ = build(Pn, b2, team, segments_from_period(period))
    FE[g] = (F[:, cols], label)
r = []
for t in (1, 2, 3):
    tr = [k for k in (1, 2, 3) if k != t]
    clf = fit([FE[k][0] for k in tr], [FE[k][1] for k in tr], step=4)
    l = FE[t][1]
    r.append(score(l, decode(proba(clf, FE[t][0]), graded_segments(l), 20, 80)))
print(f"carrier fill up to {max_s}s, trained AND tested with it: spellF1 " + " ".join(f"{x[0]:.3f}" for x in r) +
      f"  mean {np.mean([x[0] for x in r]):.3f}  macroF1 {np.mean([x[1] for x in r]):.3f}   (without: 0.602 0.609 0.573 mean 0.595)")
