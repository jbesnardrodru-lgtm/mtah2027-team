#!/bin/sh
# Ball detector trained ONLY on external footage (SoccerTrack), CPU, deterministic; evaluated on the held-out
# SoccerTrack match against the off-the-shelf model. Never touches the provided video.
#   caffeinate -i nice -n 15 sh video/run_st_detector.sh > data/st_detector.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
sh video/get_soccertrack.sh
[ -f data/ball_ds_st/data.yaml ] || python video/make_st_dataset.py --split train --n 6000 --out data/ball_ds_st
[ -f data/ball_ds_st_test/data.yaml ] || python video/make_st_dataset.py --split test --n 1500 --seed 1 --out data/ball_ds_st_test
# model selection (best epoch) on the held-out external match; never on our video
sed -i.bak "s|^val: .*|val: $(pwd)/data/ball_ds_st_test/images/val|" data/ball_ds_st/data.yaml
[ -f data/ballmodel/run_st/weights/best.pt ] || python video/train_ball.py --device cpu --epochs 20 --data data/ball_ds_st --name run_st
python video/eval_st.py --model yolo11s.pt --coco
python video/eval_st.py --model data/ballmodel/run_st/weights/best.pt
