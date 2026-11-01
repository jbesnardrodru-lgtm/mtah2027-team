"""Train so that every possession counts equally (like spellF1), not every frame.
Frame weight = 1 / (length of the run it belongs to) ** power; power 0 = current, 1 = full equalisation.
    python -m possession.exp_weight gb <power>      -> data/cache/w<power>_gb_P<test>.npy for tests 1-3
"""
import sys, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from possession.featcache import get
from possession.model import proba, STATES
from possession.exp_dead import columns


def run_weights(label, power, cap=None):
    w = np.zeros(len(label))
    s = 0
    for i in range(1, len(label) + 1):
        if i == len(label) or label[i] != label[s]:
            n = i - s
            w[s:i] = (1.0 / n) ** power if label[s] in STATES else 0.0
            s = i
    return w


if __name__ == "__main__":
    kind, power = sys.argv[1], float(sys.argv[2])
    G = {k: get(f"G{k}") for k in (1, 2, 3)}
    cols = columns(G[1][1], "v3")
    for test in (1, 2, 3):
        tr = [k for k in (1, 2, 3) if k != test]
        X, y, w = [], [], []
        for k in tr:
            F, _, l = G[k]
            wk = run_weights(l, power)
            m = np.isin(l, STATES)
            X.append(F[::4][m[::4]][:, cols]); y.append(l[::4][m[::4]]); w.append(wk[::4][m[::4]])
        X = np.vstack(X); y = np.concatenate(y); w = np.concatenate(w); w = w / w.mean()
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                             l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=w)
        np.save(f"data/cache/w{power:g}_gb_P{test}.npy", proba(clf, G[test][0][:, cols]).astype(np.float32))
    print("done", kind, power)
