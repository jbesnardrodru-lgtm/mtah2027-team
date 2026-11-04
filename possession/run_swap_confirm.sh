#!/bin/sh
# Confirm the team-swap augmentation with 3 seeds (seed 0 already done by run_variants.sh), then score.
#   caffeinate -i nice -n 15 sh possession/run_swap_confirm.sh > data/swap3.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 SEQ_WPOW=1 SEQ_EXTRA=DFL SEQ_VAR=${1:-swap}
for t in 1 2 3; do for s in 1 2; do
  f=data/cache/aug_seq_w1_dfl_${SEQ_VAR}s${s}_P$t.npy
  [ -f $f ] || python -m possession.seq aug $t $s
done; done
python -m possession.exp_seed3_score $SEQ_VAR
