"""Core: per-frame probabilities (learned on OTHER matches) + sticky H/A/D decoding (Viterbi).

    python -m possession.model lab      # cross-game evaluation on the lab (train G1 -> test G2 and back)
"""
import os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession.external import load_game
from possession.features import build
from possession.noise import degrade
from possession.lab import score

STATES = np.array(["H", "A", "D"])


def segments_from_period(period):
    p = np.nan_to_num(period.astype(float), nan=0)
    out = []
    for k in (1, 2):
        idx = np.flatnonzero(p == k)
        if len(idx):
            out.append((idx[0], idx[-1] + 1))
    return out


NOISE_V = 2  # 1 = gaps at random moments, 2 = gaps where our tracking loses the ball


def lab_game(g, noisy=True, seed=0, noise_v=None, placement="weighted", target_missing=0.41):
    P, ball, team, label, period = load_game(g)
    if noisy:
        v = noise_v or NOISE_V
        P, ball = degrade(P, ball, label, seed=seed + g, period=period if v == 2 else None, placement=placement,
                          target_missing=target_missing)
    F, names = build(P, ball, team, segments_from_period(period))
    return F, names, label


def fit(Fs, labels, step=3, seed=0):
    X = np.vstack([F[::step][np.isin(l[::step], STATES)] for F, l in zip(Fs, labels)])
    y = np.concatenate([l[::step][np.isin(l[::step], STATES)] for l in labels])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(X, y)
    return clf


def proba(clf, F):
    p = clf.predict_proba(F)
    order = [list(clf.classes_).index(s) for s in STATES]
    return p[:, order]


def viterbi(p, switch_ha, switch_d, prior=None):
    """Most likely H/A/D sequence. Changing state costs a fixed log-penalty (sticky possessions)."""
    n = len(p)
    e = np.log(np.clip(p, 1e-6, 1))
    if prior is not None:
        e = e - np.log(prior)
    T = np.array([[0, switch_ha, switch_d], [switch_ha, 0, switch_d], [switch_d, switch_d, 0]], float)
    score = e[0].copy()
    back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        cand = score[:, None] - T          # from i to j
        back[t] = cand.argmax(0)
        score = cand.max(0) + e[t]
    path = np.empty(n, np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


MIN_DEAD = 75  # frames (3 s). A restart takes time: a shorter "stoppage" is play the tracking lost.


def min_dead(lab, k=MIN_DEAD):
    """Turn D stretches shorter than k frames that follow live play back into that possession."""
    out = lab.copy()
    n, i = len(out), 0
    while i < n:
        if out[i] != "D":
            i += 1
            continue
        j = i
        while j < n and out[j] == "D":
            j += 1
        if j - i < k and i > 0 and out[i - 1] in ("H", "A"):
            out[i:j] = out[i - 1]
        i = j
    return out


def decode(p, label_or_mask_segments, switch_ha, switch_d, k_dead=MIN_DEAD):
    """Run Viterbi inside each graded segment, then the minimum-stoppage rule; X outside."""
    out = np.full(len(p), "X", "<U1")
    for a, b in label_or_mask_segments:
        out[a:b] = min_dead(STATES[viterbi(p[a:b], switch_ha, switch_d)], k_dead)
    return out


def graded_segments(label):
    g = (label != "X").astype(int)
    e = np.diff(np.r_[0, g, 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def lab(grid_ha=(10, 20, 40, 80), grid_d=(10, 20, 40)):
    games = {g: lab_game(g) for g in (1, 2)}
    names = games[1][1]
    results = {}
    for test in (1, 2):
        train = [g for g in games if g != test]
        clf = fit([games[g][0] for g in train], [games[g][2] for g in train])
        F, _, label = games[test]
        p = proba(clf, F)
        segs = graded_segments(label)
        raw = STATES[p.argmax(1)]
        raw[label == "X"] = "X"
        print(f"test G{test}: per-frame argmax  spellF1 {score(label, raw)[0]:.3f}  macroF1 {score(label, raw)[1]:.3f}")
        for sh in grid_ha:
            for sd in grid_d:
                pred = decode(p, segs, sh, sd)
                results.setdefault((sh, sd), []).append(score(label, pred))
    print("switch_ha switch_d   spellF1 (G1, G2, mean)   macroF1 mean")
    for k, v in sorted(results.items(), key=lambda kv: -np.mean([x[0] for x in kv[1]])):
        print(f"  {k[0]:>5} {k[1]:>5}     {v[0][0]:.3f} {v[1][0]:.3f}  {np.mean([x[0] for x in v]):.3f}      {np.mean([x[1] for x in v]):.3f}")


if __name__ == "__main__":
    if sys.argv[1:] == ["lab"]:
        lab()


# ---- turnover-aware decoding ------------------------------------------------------------------------
def turnover_target(label, k=12):
    """1 within +/- k frames of a direct hand-over between teams (H<->A), else 0. X frames -> -1."""
    y = np.zeros(len(label), np.int8)
    ch = np.flatnonzero((label[1:] != label[:-1]) & np.isin(label[1:], ["H", "A"]) & np.isin(label[:-1], ["H", "A"])) + 1
    for c in ch:
        y[max(0, c - k):c + k] = 1
    y[label == "X"] = -1
    return y


def fit_turnover(Fs, labels, step=4, seed=0):
    X = np.vstack([F[::step][turnover_target(l)[::step] >= 0] for F, l in zip(Fs, labels)])
    y = np.concatenate([turnover_target(l)[::step][turnover_target(l)[::step] >= 0] for l in labels])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(X, y)
    return clf


def viterbi_tv(p, cost_ha, switch_d):
    """Viterbi with a per-frame cost for switching between H and A (cheap where a turnover is likely)."""
    n = len(p)
    e = np.log(np.clip(p, 1e-6, 1))
    score = e[0].copy()
    back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        c = cost_ha[t]
        T = np.array([[0, c, switch_d], [c, 0, switch_d], [switch_d, switch_d, 0]])
        cand = score[:, None] - T
        back[t] = cand.argmax(0)
        score = cand.max(0) + e[t]
    path = np.empty(n, np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def turnover_costs(q, base, beta, prior):
    """Switch cost per frame: base, lowered where the turnover model is confident (log-odds vs prior)."""
    q = np.clip(q, 1e-4, 1 - 1e-4)
    lo = np.log(q / (1 - q)) - np.log(prior / (1 - prior))
    return np.clip(base - beta * lo, 1.0, None)


def decode_tv(p, q, segments, base, beta, prior, switch_d, k_dead=MIN_DEAD):
    out = np.full(len(p), "X", "<U1")
    cost = turnover_costs(q, base, beta, prior)
    for a, b in segments:
        out[a:b] = min_dead(STATES[viterbi_tv(p[a:b], cost[a:b], switch_d)], k_dead)
    return out


# ---- temporal context: time-shifted copies of the key signals -----------------------------------------
LAG_KEYS = ("rv_dnear_h", "rv_dnear_a", "rv_split", "conv_next_h", "conv_next_a", "conv_both", "ddiff",
            "tch_bal_12", "v_cxh", "v_cxa", "dl_v_h", "dl_v_a", "cdiff", "vall", "bspeed")
LAGS = (-50, -25, 25, 50)   # frames: 2 s and 1 s before / after


def add_lags(F, names, keys=LAG_KEYS, lags=LAGS):
    cols, new = [F], []
    for k in keys:
        x = F[:, names.index(k)]
        for L in lags:
            y = np.full_like(x, np.nan)
            if L > 0:
                y[:-L] = x[L:]
            else:
                y[-L:] = x[:L]
            cols.append(y[:, None]); new.append(f"{k}_lag{L:+d}")
    return np.hstack(cols), list(names) + new
