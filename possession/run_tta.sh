#!/bin/sh
# Lab test: test-time augmentation (predict the team-swapped match too) and 5 CNN seeds, on top of v15.
#   caffeinate -i nice -n 15 sh possession/run_tta.sh > data/tta.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 SEQ_WPOW=1 SEQ_EXTRA=DFL SEQ_VAR=swap SEQ_TTA=1
for s in 0 1 2 3 4; do for t in 1 2 3; do
  sfx=$( [ $s = 0 ] || echo s$s )
  [ -f data/cache/aug_seq_w1_dfl_swap${sfx}_P${t}_tta.npy ] || python -m possession.seq aug $t $s
done; python -m possession.exp_tta_score; done
