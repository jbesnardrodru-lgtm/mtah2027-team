"""Self-training: add the test match's own confident predictions (algorithmic pseudo-labels) and refit."""
import sys, numpy as np
from possession.featcache import get
from possession.model import fit, proba, decode, graded_segments, STATES
from possession.lab import score
from possession.exp_dead import columns
thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
GAMES = (1, 2, 3)
G = {k: get(f"G{k}") for k in GAMES}
cols = columns(G[1][1], "v3")
d = np.load("data/cache/exp_tv_probs.npz")          # round-1 probabilities (same model as v6), cached
res = {0: [], 1: []}
for test in GAMES:
    tr = [k for k in GAMES if k != test]
    F, _, l = G[test]
    p1 = d[f"P{test}"]
    segs = graded_segments(l)
    pred1 = decode(p1, segs, 20, 80)
    res[0].append(score(l, pred1))
    # pseudo-labels: decoded label where the frame-level model agrees with confidence >= thr
    conf = p1.max(1) >= thr
    agree = STATES[p1.argmax(1)] == pred1
    pseudo = np.where(conf & agree & (l != "X"), pred1, "X")
    clf = fit([G[k][0][:, cols] for k in tr] + [F[:, cols]], [G[k][2] for k in tr] + [pseudo], step=4)
    pred2 = decode(proba(clf, F[:, cols]), segs, 20, 80)
    res[1].append(score(l, pred2))
    print(f"G{test}: pseudo-labelled {np.mean(pseudo[l!='X']!='X'):.0%} of frames | spellF1 {res[0][-1][0]:.3f} -> {res[1][-1][0]:.3f}", flush=True)
print(f"threshold {thr}: mean spellF1 {np.mean([x[0] for x in res[0]]):.3f} -> {np.mean([x[0] for x in res[1]]):.3f}  "
      f"macroF1 {np.mean([x[1] for x in res[0]]):.3f} -> {np.mean([x[1] for x in res[1]]):.3f}")
