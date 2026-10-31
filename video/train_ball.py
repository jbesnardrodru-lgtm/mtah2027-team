"""Fine-tune a small YOLO on the pseudo-labelled ball crops. Deterministic seed.

    python video/train_ball.py --epochs 30
Weights end up in data/ballmodel/run/weights/best.pt
"""
import argparse, os
from common import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", default="data/ball_ds")
    ap.add_argument("--name", default="run")
    a = ap.parse_args()
    import random
    import numpy as np
    import torch
    from ultralytics import YOLO
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    workers = 2
    if device == "cpu":                     # reproducible mode: single thread pool, no worker processes
        torch.manual_seed(0); random.seed(0); np.random.seed(0)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True, warn_only=True)
        workers = 0
    import shutil
    proj = os.path.join(ROOT, "data", "ballmodel")
    work = a.name + "_work"
    last = os.path.join(proj, work, "weights", "last.pt")
    if os.path.exists(last):
        print("resuming interrupted training from", last)
        YOLO(last).train(resume=True)
    else:
        YOLO(a.base).train(data=os.path.join(ROOT, a.data, "data.yaml"), imgsz=640, epochs=a.epochs,
                           batch=a.batch, device=device, seed=0, deterministic=True, workers=workers,
                           patience=8, mosaic=0.5, scale=0.2, fliplr=0.5, hsv_v=0.3, amp=False,
                           project=proj, name=work, exist_ok=True)
    final = os.path.join(proj, a.name, "weights")
    os.makedirs(final, exist_ok=True)
    shutil.copy(os.path.join(proj, work, "weights", "best.pt"), os.path.join(final, "best.pt"))
    print("best weights:", os.path.join(final, "best.pt"))


if __name__ == "__main__":
    main()
