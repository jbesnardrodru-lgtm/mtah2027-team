"""Lab test of the carrier imputation, with the fold models trained WITHOUT it (inference-only effect) and
measured accuracy of the imputed positions against the true (clean) ball.
    python -m possession.exp_carry <max_s>
"""
import sys, pickle, numpy as np
from possession.external import load_game
from possession.noise import degrade
from possession.features import build
from possession.model import proba, decode, graded_segments, segments_from_period
from possession.featcache import get
from possession.exp_dead import columns
from possession.lab import score
from possession.carry import impute_carrier

max_s = float(sys.argv[1])
cols = columns(get("G1")[1], "v3")
for t in (1, 2, 3):
    P, ball, team, label, period = load_game(t)
    Pn, bn = degrade(P, ball, label, seed=t, period=period)
    b2, filled = impute_carrier(Pn, bn, team, max_s=max_s) if max_s > 0 else (bn, np.zeros(len(bn), bool))
    g = label != "X"
    ok = filled & ~np.isnan(ball[:, 0])
    err = np.hypot((b2[ok, 0] - ball[ok, 0]) * 105, (b2[ok, 1] - ball[ok, 1]) * 68)
    clf = pickle.load(open(f"data/cache/gb_fold{t}.pkl", "rb"))
    F, _ = build(Pn, b2, team, segments_from_period(period))
    sc = score(label, decode(proba(clf, F[:, cols]), graded_segments(label), 20, 80))
    print(f"max {max_s}s G{t}: filled {filled[g].mean():.1%} of frames (ball missing {np.mean(np.isnan(b2[g,0])):.1%}) | "
          f"imputed within 2 m of the true ball {np.mean(err<2) if len(err) else 0:.0%}, within 5 m {np.mean(err<5) if len(err) else 0:.0%} | spellF1 {sc[0]:.3f} macroF1 {sc[1]:.3f}", flush=True)
