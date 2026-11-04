"""v15 lab: 3 vs 5 CNN seeds, with and without test-time augmentation (team-swapped prediction averaged in)."""
import os, numpy as np
from possession.featcache import get
from possession.model import decode, graded_segments
from possession.lab import score
C = "data/cache"
def seq(g, n, tta):
    ps = []
    for s in ("", "s1", "s2", "s3", "s4")[:n]:
        f = f"{C}/aug_seq_w1_dfl_swap{s}_P{g}.npy"
        if not os.path.exists(f) or (tta and not os.path.exists(f.replace(".npy", "_tta.npy"))):
            return None
        p = np.load(f)
        ps.append(0.5 * (p + np.load(f.replace(".npy", "_tta.npy"))) if tta else p)
    return np.mean(ps, 0)
for n in (3, 5):
    for tta in (False, True):
        r = []
        for g in (1, 2, 3):
            q = seq(g, n, tta)
            if q is None:
                break
            l = get(f"G{g}")[2]
            r.append(score(l, decode(np.round(0.4 * np.load(f"{C}/w1_gb_P{g}.npy") + 0.6 * q, 3), graded_segments(l), 10, 80))[0])
        if len(r) == 3:
            print(f"v15 {n} seeds{' + TTA' if tta else '      '}: spellF1 {' '.join(f'{x:.3f}' for x in r)} mean {np.mean(r):.3f}", flush=True)
