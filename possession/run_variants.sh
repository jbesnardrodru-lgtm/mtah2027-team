#!/bin/sh
# Lab screening of 4 ideas against v14 (1 seed each, 3 folds):  swap | mirror | m50 | pre
#   caffeinate -i nice -n 15 sh possession/run_variants.sh > data/variants.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 SEQ_WPOW=1 SEQ_EXTRA=DFL
python -m possession.featcache_missing $(for k in 1 2 3; do for s in "" r1 r2; do printf "G$k${s}x G$k${s}y "; done; printf "G${k}m50 "; done)
for v in pre swap mirror m50; do
  for t in 1 2 3; do
    f=data/cache/aug_seq_w1_dfl_${v}_P$t.npy
    [ -f $f ] || SEQ_VAR=$v python -m possession.seq aug $t 0
  done
  python -m possession.exp_variants_score
done
