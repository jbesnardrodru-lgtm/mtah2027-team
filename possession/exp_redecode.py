"""Idea 2 (two-pass local re-decode) and blend weight, on the lab only (3 external matches, v11 models).
Pass 1: the usual sticky Viterbi. Pass 2: inside each H/A possession, a stretch where the OTHER team's blended
probability stays above `thr` for at least `L` frames is inserted as a short possession of that team
(the block Viterbi flattened). No label of the provided game is used anywhere.
    python -m possession.exp_redecode
"""
import itertools
import numpy as np
from possession.featcache import get
from possession.model import decode, graded_segments, min_dead
from possession.lab import score
from possession.errors import classify, spells

C = "data/cache"
OPP = {"H": 1, "A": 0}


def probs(g, w):
    gb = np.load(f"{C}/w1_gb_P{g}.npy")
    seq = np.mean([np.load(f"{C}/aug_seq_w1{t}_P{g}.npy") for t in ("", "s1", "s2")], 0)
    return np.round(w * gb + (1 - w) * seq, 3)


def insert_short(lab, p, thr, L):
    out = lab.copy()
    for team, k in OPP.items():
        hot = (lab == team) & (p[:, k] > thr)
        e = np.diff(np.r_[0, hot.astype(int), 0])
        for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            if b - a >= L:
                out[a:b] = "HA"[k]
    return out


def run(w, thr=None, L=None):
    res, rows, npred = [], [], 0
    for g in (1, 2, 3):
        l = get(f"G{g}")[2]; p = probs(g, w)
        pred = decode(p, graded_segments(l), 10, 80)
        if thr is not None:
            pred = insert_short(pred, p, thr, L)
        res.append(score(l, pred)); rows += classify(l, pred); npred += len(spells(pred))
    k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows])
    s = [x[0] for x in res]
    return (f"spellF1 {' '.join(f'{x:.3f}' for x in s)} mean {np.mean(s):.3f} | macroF1 {np.mean([x[1] for x in res]):.3f}"
            f" | pred {npred} | <2s {np.mean(k[d<2]=='hit'):.0%} 2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%}")


if __name__ == "__main__":
    ntrue = sum(len(spells(get(f"G{g}")[2])) for g in (1, 2, 3))
    print("true possessions in the lab:", ntrue, flush=True)
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        print(f"blend w_gb {w:.1f}                {run(w)}", flush=True)
    for thr, L in itertools.product((0.5, 0.6, 0.7, 0.8), (10, 25, 50)):
        print(f"w 0.5 insert thr {thr:.1f} L {L:>2}   {run(0.5, thr, L)}", flush=True)
