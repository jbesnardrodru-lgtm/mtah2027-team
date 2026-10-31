import sys, numpy as np
from possession.featcache import get
from possession.model import decode, graded_segments
from possession.lab import score
from possession.errors import classify, spells
mode = sys.argv[1]
ref = np.load("data/cache/exp_tv_probs.npz")
for sh in (10, 20, 40):
    for name, getp in (("metrica only", lambda g: ref[f"P{g}"]), (mode, lambda g: np.load(f"data/cache/pffexp_{mode}_P{g}.npy"))):
        r, rows, npred = [], [], 0
        for g in (1, 2, 3):
            l = get(f"G{g}")[2]; pred = decode(getp(g), graded_segments(l), sh, 80)
            r.append(score(l, pred)); rows += classify(l, pred); npred += len(spells(pred))
        k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows])
        print(f"ha {sh:>2} {name:<13}: spellF1 " + " ".join(f"{x[0]:.3f}" for x in r) + f"  mean {np.mean([x[0] for x in r]):.3f}"
              f"  macroF1 {np.mean([x[1] for x in r]):.3f} | pred {npred}/1043 | <2s {np.mean(k[d<2]=='hit'):.0%} 2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%} 5-15s {np.mean(k[(d>=5)&(d<15)]=='hit'):.0%} 15s+ {np.mean(k[d>=15]=='hit'):.0%}")
