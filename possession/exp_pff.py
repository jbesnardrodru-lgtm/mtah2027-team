"""Lab with PFF in training. One fold per run (memory): python -m possession.exp_pff <test 1|2|3> <mode>
mode: metrica = train on the other 2 Metrica games only (reference)
      both    = other 2 Metrica games + 40 PFF games
      pff     = 40 PFF games only
PFF games are subsampled at load (every 2nd frame, then the usual every 4th) to fit in memory."""
import sys, gc, numpy as np
from possession.featcache import get
from possession.pff import list_games
from possession.model import fit, proba, STATES
from possession.exp_dead import columns

test, mode = int(sys.argv[1]), sys.argv[2]
names = get("G1")[1]
cols = columns(names, "v3")
Fs, Ls = [], []
if mode in ("metrica", "both"):
    for k in (1, 2, 3):
        if k != test:
            F, _, l = get(f"G{k}"); Fs.append(F[:, cols]); Ls.append(l)
if mode in ("both", "pff"):
    for g in list_games():
        F, _, l = get(f"PFF{g}")
        Fs.append(np.ascontiguousarray(F[::2][:, cols])); Ls.append(l[::2].copy())
        del F, l; gc.collect()
clf = fit(Fs, Ls, step=4)
del Fs, Ls; gc.collect()
F, _, l = get(f"G{test}")
np.save(f"data/cache/pffexp_{mode}_P{test}.npy", proba(clf, F[:, cols]))
print("done", test, mode)
