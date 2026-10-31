#!/bin/sh
# Reproducible video pipeline: every stage on CPU with fixed seeds. Resumable stage by stage.
#   sh video/run_video_cpu.sh            (from the mtah folder, environment active)
# Writes progress to data/video_cpu.log via the caller's redirect.
set -e
export PYTHONUNBUFFERED=1
echo "== $(date) stage 0: determinism check (same 200 frames twice)"
python video/recover_ball.py --device cpu --limit 200 --model data/ballmodel/run/weights/best.pt --out data/cache/det_check_a.npz
python video/recover_ball.py --device cpu --limit 200 --model data/ballmodel/run/weights/best.pt --out data/cache/det_check_b.npz
python -c "import numpy as np; a=np.load('data/cache/det_check_a.npz')['cand']; b=np.load('data/cache/det_check_b.npz')['cand']; print('DETERMINISM CHECK:', 'IDENTICAL' if np.array_equal(a,b,equal_nan=True) else 'DIFFERENT')"
echo "== $(date) stage 1: training set (CPU)"
[ -f data/ball_ds_cpu/data.yaml ] || python video/make_pseudo_dataset.py --device cpu --n 6000 --out data/ball_ds_cpu
echo "== $(date) stage 2: training (CPU, deterministic)"
[ -f data/ballmodel/run_cpu/weights/best.pt ] || python video/train_ball.py --device cpu --epochs 20 --data data/ball_ds_cpu --name run_cpu
echo "== $(date) stage 3: ball recovery (CPU)"
python video/recover_ball.py --device cpu --model data/ballmodel/run_cpu/weights/best.pt --out data/cache/video_ball_cpu.npz
echo "== $(date) stage 4: linking"
python video/link_ball.py --inp data/cache/video_ball_cpu.npz --out data/cache/ball_video_cpu.npz
echo "== $(date) ALL DONE"
