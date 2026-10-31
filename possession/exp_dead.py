"""Dead-ball experiment: lab score + a label-free diagnostic on our match.

Diagnostic: across 18 blocks (9 per half) of OUR match, the predicted stopped share should not track the
share of frames where the tracking lost the ball. corr ~ 0 = good; high = tracking failures read as stoppages.
    python -m possession.exp_dead <variant>
"""
import sys
import numpy as np
from possession.featcache import get
from possession.model import fit, proba, decode, graded_segments, STATES
from possession.lab import score
from possession.cache import load

NEW = ("line_", "gline_", "bx_last", "bx_next", "by_abs_last", "by_abs_next")
VIS = ("vis", "since_seen", "until_seen")


def columns(names, variant):
    keep = []
    for i, n in enumerate(names):
        is_new = n.startswith(NEW)
        is_vis = n.startswith(VIS)
        if variant == "v3" and is_new:
            continue
        if variant == "novis" and is_vis:
            continue
        if variant == "noreact" and (is_new or n.startswith(("rv_", "conv_", "dl_v_", "vy_"))):
            continue
        keep.append(i)
    return keep


def main(variant, sh=40, sd=80):
    G = {k: get(f"G{k}") for k in (1, 2)}               # realistic noise: always the TEST set
    T = {k: get(f"G{k}u") for k in (1, 2)} if variant.endswith("_u") else G   # training set
    variant = variant.replace("_u", "")
    names = G[1][1]
    cols = columns(names, variant)
    res = []
    for test in (1, 2):
        tr = [k for k in G if k != test]
        clf = fit([T[k][0][:, cols] for k in tr], [T[k][2] for k in tr], step=4)
        F, _, l = G[test]
        pred = decode(proba(clf, F[:, cols]), graded_segments(l), sh, sd)
        res.append(score(l, pred))
    clf = fit([T[k][0][:, cols] for k in T], [T[k][2] for k in T], step=4)
    F, _, lab = get("BARATH")
    segs = graded_segments(lab)
    p = proba(clf, F[:, cols])
    p[110808:113742] = [0, 0, 1]
    pred = decode(p, segs, sh, sd)
    _, ball, _ = load()
    miss = np.isnan(ball[:, 0])
    ds, ms = [], []
    for a, b in [(11185, 90652), (113742, 190198)]:
        for blk in np.array_split(np.arange(a, b), 9):
            ds.append(np.mean(pred[blk] == "D")); ms.append(np.mean(miss[blk]))
    corr = np.corrcoef(ds, ms)[0, 1]
    g = pred != "X"
    print(f"{variant:<6} cols {len(cols):>2} | lab spellF1 {res[0][0]:.3f} {res[1][0]:.3f} mean {np.mean([r[0] for r in res]):.3f} "
          f"macroF1 {np.mean([r[1] for r in res]):.3f} | ours: D share {np.mean(pred[g]=='D'):.2f}  "
          f"D-vs-missing corr {corr:+.2f}  D by block min/max {min(ds):.2f}/{max(ds):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
