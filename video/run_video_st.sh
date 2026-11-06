#!/bin/sh
# Video ball layer v2: detector trained ONLY on external footage (SoccerTrack, video/run_st_detector.sh).
# Runs it over the provided video (inference only, nothing is trained or tuned on it), then links detections
# inside the tracking gaps. Confidence thresholds come from the held-out EXTERNAL match (video/eval_st.py):
# 0.3 -> 96% precision, 0.4 -> 98% precision.  CPU, deterministic, resumable.
#   caffeinate -i nice -n 15 sh video/run_video_st.sh > data/video_st.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
M=data/ballmodel/run_st/weights/best.pt
[ -f $M ] || sh video/run_st_detector.sh
if [ ! -f data/cache/video_ball_st.npz ]; then
  echo "== $(date) determinism check (same 200 frames twice)"
  python video/recover_ball.py --device cpu --limit 200 --model $M --out data/cache/det_st_a.npz
  python video/recover_ball.py --device cpu --limit 200 --model $M --out data/cache/det_st_b.npz
  python -c "import numpy as np; a=np.load('data/cache/det_st_a.npz')['cand']; b=np.load('data/cache/det_st_b.npz')['cand']; print('DETERMINISM CHECK:', 'IDENTICAL' if np.array_equal(a,b,equal_nan=True) else 'DIFFERENT')"
  echo "== $(date) detection over the whole match"
  python video/recover_ball.py --device cpu --model $M --out data/cache/video_ball_st.npz
fi
echo "== $(date) linking"
[ -f data/cache/ball_video_st.npz ] || python video/link_ball.py --tracklets --conf 0.3 --tconf 0.4 --inp data/cache/video_ball_st.npz --out data/cache/ball_video_st.npz
echo "== $(date) ALL DONE"
