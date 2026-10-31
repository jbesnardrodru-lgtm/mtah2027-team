"""Convert every downloaded PFF match (cache) and build its lab features, a time budget per run.
Re-run until it reports nothing left. python -m possession.pff_prepare [seconds]"""
import os, sys, time
import numpy as np
from possession.pff import list_games, load_pff, PFF_DIR
from possession.featcache import get, DIR

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150
t0, done, bad = time.time(), 0, []
games = list_games()
for g in games:
    if time.time() - t0 > budget:
        break
    if os.path.exists(os.path.join(DIR, f"feat_PFF{g}.npz")):
        continue
    try:
        get(f"PFF{g}")
        done += 1
    except Exception as e:
        bad.append((g, str(e)[:120]))
left = sum(not os.path.exists(os.path.join(DIR, f"feat_PFF{g}.npz")) for g in games)
print(f"prepared {done} this run; {len(games) - left} of {len(games)} ready; failed {bad}")
