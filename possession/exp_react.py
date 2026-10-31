"""Do the team-reaction features help? Same lab, same model, with vs without them."""
import sys
import numpy as np
from possession.featcache import get
from possession.model import fit, proba, decode, graded_segments
from possession.lab import score
from possession.exp_dead import columns
from possession.errors import classify, spells

variant = sys.argv[1]
G = {k: get(f"G{k}") for k in (1, 2)}
cols = columns(G[1][1], variant)
P = {}
for test in (1, 2):
    tr = [k for k in (1, 2) if k != test]
    P[test] = proba(fit([G[k][0][:, cols] for k in tr], [G[k][2] for k in tr], step=4), G[test][0][:, cols])
for sh in (10, 20):
    r, rows, npred = [], [], 0
    for test in (1, 2):
        l = G[test][2]
        pred = decode(P[test], graded_segments(l), sh, 80)
        r.append(score(l, pred)); rows += classify(l, pred); npred += len(spells(pred))
    k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows])
    print(f"{variant:<8} ha {sh:>2}: spellF1 {r[0][0]:.3f} {r[1][0]:.3f} mean {np.mean([x[0] for x in r]):.3f} "
          f"macroF1 {np.mean([x[1] for x in r]):.3f} | pred {npred}/666 | <2s {np.mean(k[d<2]=='hit'):.0%} "
          f"2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%} 5-15s {np.mean(k[(d>=5)&(d<15)]=='hit'):.0%} 15s+ {np.mean(k[d>=15]=='hit'):.0%}")
