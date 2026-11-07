#!/bin/sh
# One command: organisers' data bundle -> final submission CSV (v17).
#
#   sh make_submission.sh <team_name>
#
# Needs: the data bundle unpacked in data/ (see data/README.md), Python 3.10+ with the requirements
# installed (requirements.txt, requirements-team.txt), internet for the first run (Metrica sample data).
# Everything runs on CPU, deterministically. Every stage is skipped when its output already exists, so an
# interrupted run resumes. Nothing is trained, fitted or tuned on the provided game: all models learn only
# from external matches (organisers' clarification of 3 Nov 2026). v17: the provided video is used as INPUT only, by a
# ball detector trained on external footage (SoccerTrack); its detections can be shipped as an intermediate file.
set -e
TEAM="${1:-team}"
export PYTHONUNBUFFERED=1
export SEQ_WPOW=1          # v11: every possession counts equally when training the sequence models
export SEQ_EXTRA=DFL       # v14: the sequence models also learn from the 7 DFL/IDSSE Bundesliga matches (CC-BY 4.0)
export SEQ_VAR=swap        # v15: ... and from team-swapped copies of the Metrica matches (labels H <-> A)
export SEQ_TTA=1           # v16: each sequence model also predicts the match with the teams exchanged; averaged
cd "$(dirname "$0")"
step() { echo; echo "== $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

step "0/6 checks"
for f in data/tactical/BAR-ATH_tactical_atd.txt data/tactical/BAR-ATH_tactical_atd.xml \
         data/tactical/BAR-ATH_tactical_homography.csv data/broadcast/BAR-ATH_broadcast_atd.txt \
         data/broadcast/BAR-ATH_broadcast_atd.xml data/broadcast/BAR-ATH_broadcast_homography.csv; do
  [ -e "$f" ] || { echo "missing $f: unpack the organisers' bundle into data/ first"; exit 1; }
done
python -c "import numpy, pandas, sklearn, torch" || { echo "install the requirements first"; exit 1; }

step "1/6 external training matches (Metrica sample data)"
python -m possession.get_external
sh possession/get_dfl.sh

step "2/6 tracking caches"
[ -f data/cache/tactical.npz ] || python -m possession.cache tactical
[ -f data/cache/broadcast.npz ] || python -m possession.cache broadcast

step "3/6 lab features: 3 external matches x 3 noisy copies (+ team-swapped) + 7 DFL matches"
python -m possession.featcache_missing G1 G2 G3 G1r1 G2r1 G3r1 G1r2 G2r2 G3r2 G1x G2x G3x G1r1x G2r1x G3r1x G1r2x G2r2x G3r2x DFLJ03WMX DFLJ03WN1 DFLJ03WOH DFLJ03WOY DFLJ03WPY DFLJ03WQQ DFLJ03WR9

step "4/6 video ball layer (detector trained on SoccerTrack only; skipped if data/cache/ball_video_st.npz exists)"
[ -f data/cache/ball_video_st.npz ] || sh video/run_video_st.sh
step "4b/6 gradient-boosting model + video and broadcast ball"
python -m possession.submit --name "${TEAM}_gb" --video data/cache/ball_video_st.npz --broadcast --wpow 1

step "5/6 sequence models (3 seeds)"
FP=$(python -c "from possession.seq import inputs_fingerprint; print(inputs_fingerprint())")
for s in 0 1 2; do
  [ "$(cat data/cache/seq_final_aug_w1_dfl_swap_tta_s$s.npy.inputs 2>/dev/null)" = "$FP" ] || python -m possession.seq final $s
done

step "6/6 blend -> evaluation/results/${TEAM}.csv"
python -m possession.blend --name "$TEAM" --wpow 1 --wgb 0.4 --dfl --var swap --tta
python evaluation/validate_submission.py "evaluation/results/${TEAM}.csv"
echo
echo "checksum (must be identical on every machine):"
python -c "import hashlib,sys; print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest(), sys.argv[1])" "evaluation/results/${TEAM}.csv"
