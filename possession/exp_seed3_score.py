"""v14 vs a variant, both with 3 CNN seeds, blend 0.4 GB + 0.6 CNN.   python -m possession.exp_seed3_score swap"""
import sys, numpy as np
from possession.featcache import get
from possession.model import decode, graded_segments
from possession.lab import score
from possession.errors import classify
C = "data/cache"; v = sys.argv[1]
for tag in ("", f"_{v}"):
    res, rows = [], []
    for g in (1, 2, 3):
        l = get(f"G{g}")[2]
        seq = np.mean([np.load(f"{C}/aug_seq_w1_dfl{tag}{s}_P{g}.npy") for s in ("", "s1", "s2")], 0)
        pred = decode(np.round(0.4 * np.load(f"{C}/w1_gb_P{g}.npy") + 0.6 * seq, 3), graded_segments(l), 10, 80)
        res.append(score(l, pred)); rows += classify(l, pred)
    k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows]); s = [x[0] for x in res]
    print(f"{'v14' if not tag else 'v14 + ' + v:<12} (3 seeds) spellF1 {' '.join(f'{x:.3f}' for x in s)} mean {np.mean(s):.3f}"
          f" | macroF1 {np.mean([x[1] for x in res]):.3f} | <2s {np.mean(k[d<2]=='hit'):.0%} 2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%}", flush=True)
