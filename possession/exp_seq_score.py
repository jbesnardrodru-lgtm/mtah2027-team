import sys, numpy as np
from possession.featcache import get
from possession.model import decode, graded_segments
from possession.lab import score
from possession.errors import classify, spells
ref = np.load("data/cache/exp_tv_probs.npz")
tag = sys.argv[1] if len(sys.argv) > 1 else ""
for name, getp in (("gradient boosting (v7)", lambda g: ref[f"P{g}"]),
                   ("sequence model", lambda g: np.load(f"data/cache/seq{tag}_P{g}.npy")),
                   ("blend 50/50", lambda g: 0.5 * ref[f"P{g}"] + 0.5 * np.load(f"data/cache/seq{tag}_P{g}.npy"))):
    for sh in (10, 20, 40):
        r, rows, npred = [], [], 0
        for g in (1, 2, 3):
            l = get(f"G{g}")[2]; pred = decode(getp(g), graded_segments(l), sh, 80)
            r.append(score(l, pred)); rows += classify(l, pred); npred += len(spells(pred))
        k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows])
        print(f"{name:<23} ha {sh:>2}: spellF1 " + " ".join(f"{x[0]:.3f}" for x in r) + f"  mean {np.mean([x[0] for x in r]):.3f}"
              f"  macroF1 {np.mean([x[1] for x in r]):.3f} | pred {npred}/1043 | <2s {np.mean(k[d<2]=='hit'):.0%} 2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%}", flush=True)
