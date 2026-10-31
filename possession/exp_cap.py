import sys, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from possession.featcache import get
from possession.model import proba, decode, graded_segments, STATES
from possession.lab import score
from possession.exp_dead import columns
step, iters, leaves, lr = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
GAMES = (1, 2, 3)
G = {k: get(f"G{k}") for k in GAMES}
cols = columns(G[1][1], "v3")
P = {}
for test in GAMES:
    tr = [k for k in GAMES if k != test]
    X = np.vstack([G[k][0][::step][np.isin(G[k][2][::step], STATES)][:, cols] for k in tr])
    y = np.concatenate([G[k][2][::step][np.isin(G[k][2][::step], STATES)] for k in tr])
    clf = HistGradientBoostingClassifier(max_iter=iters, learning_rate=lr, max_leaf_nodes=leaves,
                                         l2_regularization=1.0, random_state=0).fit(X, y)
    P[test] = proba(clf, G[test][0][:, cols])
for sh in (10, 20):
    r = [score(G[t][2], decode(P[t], graded_segments(G[t][2]), sh, 80)) for t in GAMES]
    print(f"step {step} iters {iters} leaves {leaves} lr {lr} ha {sh}: spellF1 " + " ".join(f"{x[0]:.3f}" for x in r) +
          f"  mean {np.mean([x[0] for x in r]):.3f}  macroF1 {np.mean([x[1] for x in r]):.3f}", flush=True)
