"""Ensemble of 4 models trained on frame offsets 0..3 (step 4), probabilities averaged. One test game per run."""
import sys, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from possession.featcache import get
from possession.model import proba, STATES
from possession.exp_dead import columns
test = int(sys.argv[1])
GAMES = (1, 2, 3)
G = {k: get(f"G{k}") for k in GAMES}
cols = columns(G[1][1], "v3")
tr = [k for k in GAMES if k != test]
ps = []
for off in range(4):
    X = np.vstack([G[k][0][off::4][np.isin(G[k][2][off::4], STATES)][:, cols] for k in tr])
    y = np.concatenate([G[k][2][off::4][np.isin(G[k][2][off::4], STATES)] for k in tr])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=off).fit(X, y)
    ps.append(proba(clf, G[test][0][:, cols]))
    print("model", off, "done", flush=True)
np.save(f"data/cache/ens_P{test}.npy", np.mean(ps, 0))
