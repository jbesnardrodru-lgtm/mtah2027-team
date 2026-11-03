#!/bin/sh
# Overnight lab test: sequence models trained with the 7 DFL matches added, 3 folds x 3 seeds, then the v13 blend
# scored against v13.   caffeinate -i nice -n 15 sh possession/run_dfl_seq.sh > data/dfl_seq.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 SEQ_WPOW=1 SEQ_EXTRA=DFL
for t in 1 2 3; do
  [ -f data/cache/dflexp_both_P$t.npy ] || python -m possession.exp_dfl $t both
  for s in 0 1 2; do
    f=data/cache/aug_seq_w1_dfl$( [ $s = 0 ] || echo s$s )_P$t.npy
    [ -f $f ] || python -m possession.seq aug $t $s
  done
done
python -m possession.exp_dfl_score
