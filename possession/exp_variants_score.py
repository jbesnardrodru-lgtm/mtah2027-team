"""Screening: v14 blend (0.4 GB + 0.6 CNN) with ONE CNN seed, baseline vs each variant present in the cache."""
import os, numpy as np
from possession.featcache import get
from possession.model import decode, graded_segments
from possession.lab import score
from possession.errors import classify, spells
C = "data/cache"
for v in ("", "_pre", "_swap", "_mirror", "_m50"):
    if not all(os.path.exists(f"{C}/aug_seq_w1_dfl{v}_P{g}.npy") for g in (1, 2, 3)):
        continue
    res, rows = [], []
    for g in (1, 2, 3):
        l = get(f"G{g}")[2]
        p = np.round(0.4 * np.load(f"{C}/w1_gb_P{g}.npy") + 0.6 * np.load(f"{C}/aug_seq_w1_dfl{v}_P{g}.npy"), 3)
        pred = decode(p, graded_segments(l), 10, 80); res.append(score(l, pred)); rows += classify(l, pred)
    k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows]); s = [x[0] for x in res]
    print(f"{(v or '_v14 (1 seed)')[1:]:<14} spellF1 {' '.join(f'{x:.3f}' for x in s)} mean {np.mean(s):.3f} | <2s {np.mean(k[d<2]=='hit'):.0%} 2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%}", flush=True)
