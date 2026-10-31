"""3-match lab: does temporal context (lagged copies) help?"""
import sys
import numpy as np
from possession.featcache import get
from possession.model import fit, proba, decode, graded_segments, add_lags
from possession.lab import score
from possession.exp_dead import columns
from possession.errors import classify, spells
from possession.timing import offsets, changes, kind

GAMES = (1, 2, 3)
G = {}
for k in GAMES:
    F, names, l = get(f"G{k}")
    F, names2 = add_lags(F, names)
    G[k] = (F, names2, l)
names2 = G[1][1]
base_cols = columns(names2, "v3")
P = {}
for test in GAMES:
    tr = [k for k in GAMES if k != test]
    P[test] = proba(fit([G[k][0][:, base_cols] for k in tr], [G[k][2] for k in tr], step=4), G[test][0][:, base_cols])
np.savez("data/cache/exp_lags_probs.npz", **{f"P{k}": P[k] for k in GAMES})
for sh in (10, 20, 40):
    r, rows, npred, tt, tn = [], [], 0, 0, 0
    for test in GAMES:
        l = G[test][2]
        pred = decode(P[test], graded_segments(l), sh, 80)
        r.append(score(l, pred)); rows += classify(l, pred); npred += len(spells(pred))
        o = offsets(l, pred).get("team->team", np.array([]))
        tt += len(o); tn += sum(1 for t, a, b in changes(l) if kind(a, b) == "team->team")
    k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows])
    print(f"lags ha {sh:>2}: spellF1 " + " ".join(f"{x[0]:.3f}" for x in r) + f"  mean {np.mean([x[0] for x in r]):.3f}"
          f"  macroF1 {np.mean([x[1] for x in r]):.3f} | pred {npred}/{len(rows)} | team changes caught {tt/tn:.0%} | "
          f"<2s {np.mean(k[d<2]=='hit'):.0%} 2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%}", flush=True)
