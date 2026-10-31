"""Turnover-aware decoder vs fixed-cost decoder in the 3-match lab."""
import numpy as np
from possession.featcache import get
from possession.model import fit, proba, decode, graded_segments, fit_turnover, decode_tv, turnover_target
from possession.lab import score
from possession.exp_dead import columns
from possession.errors import classify, spells

GAMES = (1, 2, 3)
G = {k: get(f"G{k}") for k in GAMES}
cols = columns(G[1][1], "v3")
P, Q, prior = {}, {}, {}
for test in GAMES:
    tr = [k for k in GAMES if k != test]
    P[test] = proba(fit([G[k][0][:, cols] for k in tr], [G[k][2] for k in tr], step=4), G[test][0][:, cols])
    tc = fit_turnover([G[k][0][:, cols] for k in tr], [G[k][2] for k in tr])
    Q[test] = tc.predict_proba(G[test][0][:, cols])[:, 1]
    ys = np.concatenate([turnover_target(G[k][2]) for k in tr]); prior[test] = np.mean(ys[ys >= 0])
np.savez("data/cache/exp_tv_probs.npz", **{f"P{k}": P[k] for k in GAMES}, **{f"Q{k}": Q[k] for k in GAMES})
print("turnover prior", {k: round(v, 3) for k, v in prior.items()}, flush=True)
for base, beta in ((20, 0), (30, 4), (40, 6), (40, 10)):
    r, rows, npred = [], [], 0
    for test in GAMES:
        l = G[test][2]
        pred = decode_tv(P[test], Q[test], graded_segments(l), base, beta, prior[test], 80)
        r.append(score(l, pred)); rows += classify(l, pred); npred += len(spells(pred))
    k = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows])
    print(f"base {base:>2} beta {beta:>2}: spellF1 " + " ".join(f"{x[0]:.3f}" for x in r) +
          f"  mean {np.mean([x[0] for x in r]):.3f} macroF1 {np.mean([x[1] for x in r]):.3f} | pred {npred}/{len(rows)} | "
          f"<2s {np.mean(k[d<2]=='hit'):.0%} 2-5s {np.mean(k[(d>=2)&(d<5)]=='hit'):.0%} 5-15s {np.mean(k[(d>=5)&(d<15)]=='hit'):.0%} 15s+ {np.mean(k[d>=15]=='hit'):.0%}", flush=True)
