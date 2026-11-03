"""Lab with the DFL (Bundesliga) matches in training, gradient boosting with possession weighting (as v11).
    python -m possession.exp_dfl feats            # build degraded features for the 7 DFL matches (cached)
    python -m possession.exp_dfl <test 1|2|3> <metrica|both|dfl>
Scores the held-out Metrica match; compare 'both' and 'dfl' with 'metrica'."""
import sys, numpy as np
from possession.featcache import get
from possession.dfl import MATCHES
from possession.model import fit, proba, decode, graded_segments
from possession.exp_dead import columns
from possession.lab import score
from possession.errors import spells

if sys.argv[1] == "feats":
    for m in (sys.argv[2:] or MATCHES):
        F, _, l = get(f"DFL{m}"); print(m, F.shape, flush=True)
    sys.exit()
test, mode = int(sys.argv[1]), sys.argv[2]
cols = columns(get("G1")[1], "v3")
Fs, Ls = [], []
if mode in ("metrica", "both", "teamonly"):
    for k in (1, 2, 3):
        if k != test:
            F, _, l = get(f"G{k}"); Fs.append(F[:, cols]); Ls.append(l)
if mode in ("both", "dfl", "teamonly"):
    for m in MATCHES:
        F, _, l = get(f"DFL{m}")
        if mode == "teamonly":                       # DFL teaches only who has the ball; dead ball from Metrica only
            l = np.where(l == "D", "X", l)
        Fs.append(F[:, cols]); Ls.append(l)
clf = fit(Fs, Ls, step=4, weight_power=1)
F, _, l = get(f"G{test}")
p = proba(clf, F[:, cols])
np.save(f"data/cache/dflexp_{mode}_P{test}.npy", p.astype(np.float32))
for ha in (10, 20):
    pred = decode(p, graded_segments(l), ha, 80)
    sc = score(l, pred)
    print(f"G{test} {mode:<8} ha {ha}: spellF1 {sc[0]:.3f} macroF1 {sc[1]:.3f}  pred possessions {len(spells(pred))} (true {len(spells(l))})  D {np.mean(pred[l!='X']=='D'):.0%}", flush=True)
