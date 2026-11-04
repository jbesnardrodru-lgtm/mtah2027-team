#!/bin/sh
# Queued after run_tta.sh: v15 CNN with the 40 PFF World Cup matches added (1 seed, 3 folds), vs v15 1 seed.
#   caffeinate -i nice -n 15 sh possession/run_pff_seq.sh > data/pff_seq.log 2>&1 &
cd "$(dirname "$0")/.."
while pgrep -f run_tta.sh > /dev/null; do sleep 60; done      # wait for the TTA test to finish
set -e
export PYTHONUNBUFFERED=1 SEQ_WPOW=1 SEQ_EXTRA=DFL,PFF SEQ_VAR=swap
for t in 1 2 3; do
  [ -f data/cache/aug_seq_w1_dfl_pff_swap_P$t.npy ] || python -m possession.seq aug $t 0
done
python - <<'PY'
import numpy as np
from possession.featcache import get
from possession.model import decode, graded_segments
from possession.lab import score
C = "data/cache"
for tag in ("_dfl_swap", "_dfl_pff_swap"):
    r = []
    for g in (1, 2, 3):
        l = get(f"G{g}")[2]
        p = np.round(0.4 * np.load(f"{C}/w1_gb_P{g}.npy") + 0.6 * np.load(f"{C}/aug_seq_w1{tag}_P{g}.npy"), 3)
        r.append(score(l, decode(p, graded_segments(l), 10, 80))[0])
    print(f"{'v15' if tag == '_dfl_swap' else 'v15 + PFF':<10} (1 seed) spellF1 {' '.join(f'{x:.3f}' for x in r)} mean {np.mean(r):.3f}", flush=True)
PY
