#!/bin/sh
# Last lab test: weight short possessions more (frame weight = 1 / run length ** 1.5 instead of ** 1), on top of v16.
#   caffeinate -i nice -n 15 sh possession/run_wpow.sh > data/wpow.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 SEQ_WPOW=1.5 SEQ_EXTRA=DFL SEQ_VAR=swap SEQ_TTA=1
[ -f data/cache/w1.5_gb_P3.npy ] || python -m possession.exp_weight gb 1.5
for s in 0 1 2; do for t in 1 2 3; do
  sfx=$( [ $s = 0 ] || echo s$s )
  [ -f data/cache/aug_seq_w1.5_dfl_swap${sfx}_P${t}_tta.npy ] || python -m possession.seq aug $t $s
done; done
python -m possession.exp_wpow_score
