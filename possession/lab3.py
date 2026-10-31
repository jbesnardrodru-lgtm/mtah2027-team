"""3-match lab: train on two external games, test on the third, rotate. Official metric code.
    python -m possession.lab3 <variant> [ha ...]
"""
import sys
import numpy as np
from possession.featcache import get
from possession.model import fit, proba, decode, graded_segments
from possession.lab import score
from possession.exp_dead import columns
from possession.errors import classify, spells

GAMES = (1, 2, 3)


def run(variant="v3", grid_ha=(10, 20, 40), sd=80, decoder=None):
    G = {k: get(f"G{k}") for k in GAMES}
    cols = columns(G[1][1], variant)
    P = {}
    for test in GAMES:
        tr = [k for k in GAMES if k != test]
        P[test] = proba(fit([G[k][0][:, cols] for k in tr], [G[k][2] for k in tr], step=4), G[test][0][:, cols])
    out = {}
    for sh in grid_ha:
        r, rows, npred = [], [], 0
        for test in GAMES:
            l = G[test][2]
            pred = (decoder or decode)(P[test], graded_segments(l), sh, sd)
            r.append(score(l, pred)); rows += classify(l, pred); npred += len(spells(pred))
        k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows])
        m = np.mean([x[0] for x in r])
        out[sh] = m
        print(f"{variant:<8} ha {sh:>2}: spellF1 " + " ".join(f"{x[0]:.3f}" for x in r) +
              f"  mean {m:.3f}  macroF1 {np.mean([x[1] for x in r]):.3f} | pred {npred}/{len(rows)} | "
              f"<2s {np.mean(k[d<2]=='hit'):.0%} 2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%} "
              f"5-15s {np.mean(k[(d>=5)&(d<15)]=='hit'):.0%} 15s+ {np.mean(k[d>=15]=='hit'):.0%}", flush=True)
    return P, out


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "v3"
    ha = tuple(int(x) for x in sys.argv[2:]) or (10, 20, 40)
    run(v, ha)
