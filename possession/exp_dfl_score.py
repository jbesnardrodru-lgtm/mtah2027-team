"""Score the v13 blend (0.4 gradient boosting + 0.6 CNN, 3 seeds) in the lab, without and with the DFL matches."""
import numpy as np
from possession.featcache import get
from possession.model import decode, graded_segments
from possession.lab import score
from possession.errors import classify, spells

C = "data/cache"
def blend(g, dfl):
    gb = np.load(f"{C}/dflexp_both_P{g}.npy" if dfl else f"{C}/w1_gb_P{g}.npy")
    tag = "_dfl" if dfl else ""
    seq = np.mean([np.load(f"{C}/aug_seq_w1{tag}{t}_P{g}.npy") for t in ("", "s1", "s2")], 0)
    return np.round(0.4 * gb + 0.6 * seq, 3)

for dfl in (False, True):
    res, rows, npred = [], [], 0
    for g in (1, 2, 3):
        l = get(f"G{g}")[2]; pred = decode(blend(g, dfl), graded_segments(l), 10, 80)
        res.append(score(l, pred)); rows += classify(l, pred); npred += len(spells(pred))
    k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows]); s = [x[0] for x in res]
    print(f"{'v13 + DFL' if dfl else 'v13      '}: spellF1 {' '.join(f'{x:.3f}' for x in s)} mean {np.mean(s):.3f}"
          f" | macroF1 {np.mean([x[1] for x in res]):.3f} | pred {npred}/1043 | <2s {np.mean(k[d<2]=='hit'):.0%} 2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%}")
