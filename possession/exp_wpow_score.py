"""v16 (weight power 1) vs power 1.5 for the CNN only, and for both models. Lab, 3 seeds + TTA, blend 0.4 / 0.6."""
import numpy as np
from possession.featcache import get
from possession.model import decode, graded_segments
from possession.lab import score
from possession.errors import classify, spells
C = "data/cache"
def cnn(g, w):
    ps = []
    for s in ("", "s1", "s2"):
        f = f"{C}/aug_seq_w{w}_dfl_swap{s}_P{g}.npy"
        ps.append(0.5 * (np.load(f) + np.load(f.replace(".npy", "_tta.npy"))))
    return np.mean(ps, 0)
for name, wg, wc in (("v16 (power 1)", "1", "1"), ("CNN power 1.5", "1", "1.5"), ("both power 1.5", "1.5", "1.5")):
    r, rows, n = [], [], 0
    for g in (1, 2, 3):
        l = get(f"G{g}")[2]
        p = np.round(0.4 * np.load(f"{C}/w{wg}_gb_P{g}.npy") + 0.6 * cnn(g, wc), 3)
        pred = decode(p, graded_segments(l), 10, 80); r.append(score(l, pred)[0]); rows += classify(l, pred); n += len(spells(pred))
    k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows])
    print(f"{name:<15} spellF1 {' '.join(f'{x:.3f}' for x in r)} mean {np.mean(r):.3f} | pred {n}/1043 | <2s {np.mean(k[d<2]=='hit'):.0%} 2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%}", flush=True)
