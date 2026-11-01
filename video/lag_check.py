"""Is the organisers' tracking ball ahead of / behind the video? No labels used.

Runs our ball detector on blocks of consecutive frames where the tracking sees the ball, then finds the time
shift (in frames) at which the tracking ball and the detected ball agree best. 0 = in sync.
    python video/lag_check.py            (Mac GPU: a few minutes)
"""
import argparse, os
import numpy as np
from common import ROOT, asset_paths, tracking_ball, homography, project, graded_mask, read_frames, detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--length", type=int, default=150, help="frames per block (6 s)")
    ap.add_argument("--max_shift", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(os.path.join(ROOT, "data", "ballmodel", "run_cpu", "weights", "best.pt"))
    ball = tracking_ball(); M, _ = homography(); g = graded_mask(len(ball))
    seen = ~np.isnan(ball[:, 0])
    # candidate blocks: fully graded, tracking ball visible on >= 95% of the frames, ball on the ground
    starts = []
    for s in range(11185, 190000 - a.length, 997):
        w = slice(s - a.max_shift, s + a.length + a.max_shift)
        if g[w].all() and seen[w].mean() > 0.95 and np.isnan(ball[w, 2]).mean() > 0.95:
            starts.append(s)
    rng = np.random.default_rng(0)
    starts = sorted(rng.choice(starts, min(a.blocks, len(starts)), replace=False))
    video = asset_paths()["video"]
    err = {k: [] for k in range(-a.max_shift, a.max_shift + 1)}
    for s in starts:
        frames = list(range(s, s + a.length))
        imgs = dict(read_frames(video, frames))
        dets = {}
        for i in range(0, len(frames), a.batch):
            fs = [f for f in frames[i:i + a.batch] if f in imgs]
            for f, d in zip(fs, detect(model, [imgs[f] for f in fs], device, a.conf, tiled=True)):
                if len(d):
                    dets[f] = d[np.argmax(d[:, 2]), :2]
        for k in err:
            for f, (u, v) in dets.items():
                t = f + k                                   # tracking frame compared with video frame f
                if np.isnan(ball[t, 0]):
                    continue
                pu, pv = project(M[f], ball[t, 0], ball[t, 1])
                e = np.hypot(pu - u, pv - v)
                if e < 80:
                    err[k].append(e)
        print(f"block at frame {s}: {len(dets)} detections", flush=True)
    med = {k: np.median(v) for k, v in err.items() if len(v) > (50 if a.blocks > 2 else 3)}
    if not med:
        print("not enough detections matched the tracking ball: try more --blocks or a lower --conf"); return
    best = min(med, key=med.get)
    print("\nshift (frames)  median pixel distance   [negative shift = tracking is AHEAD of the video]")
    for k, m in med.items():
        print(f"   {k:+3d}           {m:6.1f}{'   <- best' if k == best else ''}")
    print(f"\nRESULT: tracking best aligned at shift {best:+d} frames ({best/25:+.2f} s)")


if __name__ == "__main__":
    main()
