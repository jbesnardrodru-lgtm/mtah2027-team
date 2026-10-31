"""Are our predicted changes early or late? Match each true change to the nearest predicted change of the
same kind within +/- 2 s and report the offset (predicted - true, in seconds)."""
import numpy as np


def changes(lab):
    i = np.flatnonzero((lab[1:] != lab[:-1]) & (lab[1:] != "X") & (lab[:-1] != "X")) + 1
    return [(int(t), lab[t - 1], lab[t]) for t in i]


def kind(a, b):
    if a in "HA" and b in "HA":
        return "team->team"
    if b == "D":
        return "play->stop"
    return "stop->play"


def offsets(gt, pred, tol=50):
    P = {}
    for t, a, b in changes(pred):
        P.setdefault(kind(a, b), []).append(t)
    P = {k: np.array(v) for k, v in P.items()}
    out = {}
    for t, a, b in changes(gt):
        k = kind(a, b)
        if k not in P:
            continue
        j = np.argmin(np.abs(P[k] - t))
        d = P[k][j] - t
        if abs(d) <= tol:
            out.setdefault(k, []).append(d)
    return {k: np.array(v) / 25 for k, v in out.items()}


if __name__ == "__main__":
    from possession.featcache import get
    from possession.model import decode, graded_segments
    d = np.load("data/cache/exp_tv_probs.npz")
    allo = {}
    for g in (1, 2, 3):
        l = get(f"G{g}")[2]
        pred = decode(d[f"P{g}"], graded_segments(l), 20, 80)
        n_true = {}
        for t, a, b in changes(l):
            n_true[kind(a, b)] = n_true.get(kind(a, b), 0) + 1
        for k, v in offsets(l, pred).items():
            allo.setdefault(k, []).append(v)
            allo.setdefault(k + "_n", []).append(n_true.get(k, 0))
    for k in ("team->team", "play->stop", "stop->play"):
        v = np.concatenate(allo[k]); n = sum(allo[k + "_n"])
        print(f"{k:<11} true {n:>4}  matched within 2 s {len(v)/n:4.0%}  offset median {np.median(v):+.2f}s  "
              f"mean {np.mean(v):+.2f}s  p25/p75 {np.percentile(v,25):+.2f}/{np.percentile(v,75):+.2f}s  "
              f"late>0.5s {np.mean(v>0.5):.0%} early<-0.5s {np.mean(v<-0.5):.0%}")
