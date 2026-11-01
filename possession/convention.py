"""Which labelling convention does the official answer key resemble? Compares LABEL statistics only (no model
is scored): the public window (first 5 min of play of the 1st half) vs the first 5 min of play of every half in
the Metrica lab games and the PFF games."""
import glob
import numpy as np, pandas as pd
from possession.external import load_game

W = 7500  # 5 minutes at 25 fps


def stats(lab):
    runs, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            runs.append((lab[s], i - s)); s = i
    poss = [n for k, n in runs if k in "HA"]
    dead = [n for k, n in runs if k == "D"]
    return dict(poss=len(poss), short3=sum(n < 75 for n in poss), dead_share=np.mean(lab == "D"),
                dead_runs=len(dead), med_poss=np.median(poss) / 25 if poss else np.nan,
                med_dead=np.median(dead) / 25 if dead else 0.0)


def opening_windows(label, period):
    out = []
    for k in (1, 2):
        idx = np.flatnonzero((period == k) & (label != "X"))
        if len(idx) < W:
            continue
        # first frame of live play in that half = kick-off
        live = idx[np.isin(label[idx], ["H", "A"])]
        if not len(live):
            continue
        a = live[0]
        win = label[a:a + W]
        if np.all(win != "X"):
            out.append(win)
    return out


gt = pd.read_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv").label.values
pub = stats(gt[11185:11185 + W])
pops = {"Metrica": [], "PFF": []}
for g in (1, 2, 3):
    _, _, _, label, period = load_game(g)
    pops["Metrica"] += [stats(w) for w in opening_windows(label, np.nan_to_num(period.astype(float)))]
for f in sorted(glob.glob("data/pff/cache/pff_*.npz")):
    d = np.load(f)
    pops["PFF"] += [stats(w) for w in opening_windows(d["label"], d["period"])]

keys = [("poss", "possessions"), ("short3", "possessions < 3 s"), ("dead_share", "dead share"),
        ("dead_runs", "stoppages"), ("med_poss", "median possession (s)"), ("med_dead", "median stoppage (s)")]
print(f"{'first 5 min of play':<24}{'official':>10}   {'Metrica: median [p10-p90]  rank':>34}   {'PFF: median [p10-p90]  rank':>32}")
for k, name in keys:
    line = f"{name:<24}{pub[k]:>10.2f}   "
    for p in ("Metrica", "PFF"):
        v = np.array([s[k] for s in pops[p]], float)
        pct = np.mean(v < pub[k]) * 100 + np.mean(v == pub[k]) * 50
        line += f"{np.median(v):>8.2f} [{np.percentile(v,10):6.2f}-{np.percentile(v,90):6.2f}] {pct:4.0f}%   "
    print(line)
print({p: len(v) for p, v in pops.items()}, "windows")
