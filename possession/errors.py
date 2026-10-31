"""Where do we lose spellF1? Classify every true possession the prediction fails to recover (IoU < 0.5).

  hit         a predicted same-team spell matches it with IoU >= 0.5
  split       the right team, but our spell is broken into 2+ pieces inside it (flicker or a false stoppage)
  merged      the right team, but our spell runs on across a hand-over or stoppage (too long)
  boundary    the right team, one spell, but start/end too far off
  wrong team  inside it we mostly say the other team
  said dead   inside it we mostly say D
"""
import numpy as np


def spells(lab):
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            if lab[s] in ("H", "A"):
                out.append((lab[s], s, i))
            s = i
    return out


def classify(gt, pred):
    ps = spells(pred)
    starts = np.array([p[1] for p in ps])
    rows = []
    for team, a, b in spells(gt):
        seg = pred[a:b]
        share_team = np.mean(seg == team)
        lo = np.searchsorted(starts, a, "right") - 1
        over = [p for p in ps[max(lo, 0):] if p[1] < b and p[2] > a]
        same = [p for p in over if p[0] == team]
        best = max((((min(b, e) - max(a, s)) / (max(b, e) - min(a, s))), s, e) for _, s, e in same) if same else (0, 0, 0)
        dur = (b - a) / 25
        if best[0] >= 0.5:
            kind = "hit"
        elif share_team < 0.5:
            kind = "said dead" if np.mean(seg == "D") > np.mean(seg != team) - np.mean(seg == "D") else "wrong team"
        elif len(same) >= 2:
            kind = "split"
        elif (best[2] - best[1]) > (b - a) * 1.5:
            kind = "merged"
        else:
            kind = "boundary"
        rows.append((kind, dur))
    return rows


def report(rows, label):
    kinds = ["hit", "split", "merged", "boundary", "wrong team", "said dead"]
    dur = np.array([r[1] for r in rows]); k = np.array([r[0] for r in rows])
    print(f"{label}: {len(rows)} true possessions")
    for name in kinds:
        m = k == name
        print(f"   {name:<11} {m.mean():5.0%}   median length {np.median(dur[m]) if m.any() else 0:5.1f}s")
    for lo, hi in ((0, 2), (2, 5), (5, 15), (15, 999)):
        m = (dur >= lo) & (dur < hi)
        print(f"   length {lo:>2}-{hi if hi < 999 else '+':<3}s: {m.sum():>3} possessions, recovered {np.mean(k[m] == 'hit'):4.0%}")


if __name__ == "__main__":
    from possession.featcache import get
    from possession.model import fit, proba, decode, graded_segments
    from possession.exp_dead import columns
    G = {k: get(f"G{k}") for k in (1, 2)}
    cols = columns(G[1][1], "v3")
    allrows = []
    for test in (1, 2):
        tr = [k for k in G if k != test]
        clf = fit([G[k][0][:, cols] for k in tr], [G[k][2] for k in tr], step=4)
        F, _, l = G[test]
        pred = decode(proba(clf, F[:, cols]), graded_segments(l), 40, 80)
        allrows += classify(l, pred)
        ps = spells(pred)
        print(f"G{test}: predicted {len(ps)} possessions vs {len(spells(l))} true")
    report(allrows, "both lab games")
