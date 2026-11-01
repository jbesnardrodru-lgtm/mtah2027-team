#!/bin/sh
# One command: organisers' data bundle -> final submission CSV (v9).
#
#   sh make_submission.sh <team_name>
#
# Needs: the data bundle unpacked in data/ (see data/README.md), Python 3.10+ with the requirements
# installed (requirements.txt, requirements-team.txt, video/requirements.txt), internet for the first run
# (Metrica sample data + the public YOLO11s weights). Everything runs on CPU, deterministically.
# Every stage is skipped when its output already exists, so an interrupted run resumes. A full run from
# scratch takes roughly 15-20 hours on a laptop CPU, almost all of it in the video stage.
set -e
TEAM="${1:-team}"
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
step() { echo; echo "== $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

step "0/7 checks"
for f in data/tactical/BAR-ATH_tactical.mp4 data/tactical/BAR-ATH_tactical_atd.txt data/tactical/BAR-ATH_tactical_atd.xml \
         data/tactical/BAR-ATH_tactical_homography.csv data/broadcast/BAR-ATH_broadcast_atd.txt \
         data/broadcast/BAR-ATH_broadcast_atd.xml data/broadcast/BAR-ATH_broadcast_homography.csv; do
  [ -e "$f" ] || { echo "missing $f: unpack the organisers' bundle into data/ first"; exit 1; }
done
python -c "import numpy, pandas, sklearn, torch, ultralytics, av" || { echo "install the requirements first"; exit 1; }

step "1/7 external training matches (Metrica sample data)"
python -m possession.get_external

step "2/7 tracking caches"
[ -f data/cache/tactical.npz ] || python -m possession.cache tactical
[ -f data/cache/broadcast.npz ] || python -m possession.cache broadcast

step "3/7 video ball layer (CPU, deterministic; the long stage)"
sh video/run_video_cpu.sh

step "4/7 lab features: 3 external matches x 3 noisy copies"
python -m possession.featcache_missing G1 G2 G3 G1r1 G2r1 G3r1 G1r2 G2r2 G3r2

step "5/7 gradient-boosting model + video and broadcast ball (v7 stage)"
python -m possession.submit --name "${TEAM}_v7" --video --broadcast

step "6/7 sequence models (3 seeds)"
for s in 0 1 2; do
  [ -f "data/cache/seq_final_aug_s$s.npy" ] || python -m possession.seq final $s
done

step "7/7 blend -> evaluation/results/${TEAM}.csv"
python -m possession.blend --name "$TEAM"
python evaluation/validate_submission.py "evaluation/results/${TEAM}.csv"
echo
echo "checksum (must be identical on every machine):"
python -c "import hashlib,sys; print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest(), sys.argv[1])" "evaluation/results/${TEAM}.csv"
