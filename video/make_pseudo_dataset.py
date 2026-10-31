"""Build a ball-detection training set from THIS match with algorithmic pseudo-labels only.

Label rule (no human decides anything):
  a frame is used when the automatic tracking has the ball on the ground (no height reading) AND the
  off-the-shelf detector puts a box within --agree px of the tracking ball projected into the image.
  The detector supplies a tight box; the tracking confirms it is the ball and not a head or the spot.
  With --loose, frames where the detector found nothing get a fixed-size box at the projected point.

Training crops are 640x640 at native resolution (the ball stays a few pixels, as at inference),
plus ball-free negative crops so the model learns what is NOT the ball.
Held-out: every 5th two-minute block of the match is never used here (see common.block_split),
so the probe can test on frames the model has never seen.

    python video/make_pseudo_dataset.py --n 10000
"""
import argparse, os, time
import numpy as np, cv2
from common import (ROOT, W, H, TILE, asset_paths, tracking_ball, homography, project, graded_mask,
                    block_split, read_frames, detect)

OUT = os.path.join(ROOT, "data", "ball_ds")  # overridden by --out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="frames to sample")
    ap.add_argument("--agree", type=float, default=15.0, help="px between detector and tracking")
    ap.add_argument("--loose", action="store_true")
    ap.add_argument("--neg_every", type=int, default=3, help="one negative crop per this many positives")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="dataset folder (default data/ball_ds)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = os.path.join(ROOT, a.out)

    import torch
    from ultralytics import YOLO
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)

    ball = tracking_ball()
    M, valid = homography()
    n = len(ball)
    frames = np.arange(n)
    ok = graded_mask(n) & valid & ~np.isnan(ball[:, 0]) & np.isnan(ball[:, 2])  # ground ball only
    ok &= ~block_split(frames)
    cand = frames[ok]
    pick = np.sort(rng.choice(cand, min(a.n, len(cand)), replace=False))

    # project and keep balls safely inside the image
    uv = np.array([project(M[f], ball[f, 0], ball[f, 1]) for f in pick])
    inside = (uv[:, 0] > 8) & (uv[:, 0] < W - 8) & (uv[:, 1] > 8) & (uv[:, 1] < H - 8)
    pick, uv = pick[inside], uv[inside]
    target = {int(f): p for f, p in zip(pick, uv)}

    # train / val split by time block inside the training pool (val = for YOLO's own early stopping)
    val_blocks = (pick // (25 * 120)) % 7 == 3
    split = {int(f): ("val" if v else "train") for f, v in zip(pick, val_blocks)}
    for s in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", s), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", s), exist_ok=True)

    model = YOLO(a.model)
    video = asset_paths()["video"]
    stats = dict(frames=0, agreed=0, loose=0, skipped=0, pos=0, neg=0)
    t0, batch = time.time(), []

    def flush(batch):
        dets = detect(model, [img for _, img in batch], device, conf=0.05, tiled=False)
        for (f, img), d in zip(batch, dets):
            stats["frames"] += 1
            u, v = target[f]
            box = None
            if len(d):
                dist = np.hypot(d[:, 0] - u, d[:, 1] - v)
                if dist.min() <= a.agree:
                    k = int(dist.argmin())
                    box = (d[k, 0], d[k, 1])
                    stats["agreed"] += 1
            if box is None:
                if not a.loose:
                    stats["skipped"] += 1
                    continue
                box = (u, v)
                stats["loose"] += 1
            write_positive(f, img, box, split[f], rng)
            stats["pos"] += 1
            if stats["pos"] % a.neg_every == 0:
                if write_negative(f, img, box, split[f], rng):
                    stats["neg"] += 1

    for f, img in read_frames(video, pick):
        batch.append((f, img))
        if len(batch) == 16:
            flush(batch); batch = []
            if stats["frames"] % 480 == 0:
                print(f"  {stats['frames']}/{len(pick)} frames  {stats}  {time.time()-t0:.0f}s", flush=True)
    if batch:
        flush(batch)

    with open(os.path.join(OUT, "data.yaml"), "w") as fh:
        fh.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    print("done", stats, f"{time.time()-t0:.0f}s ->", OUT)


BOX = 14  # px, one fixed box size for every label (the ball is 6-12 px in this camera)


def crop_origin(cx, cy, rng, margin=48):
    x0 = int(np.clip(cx - rng.uniform(margin, TILE - margin), 0, W - TILE))
    y0 = int(np.clip(cy - rng.uniform(margin, TILE - margin), 0, H - TILE))
    return x0, y0


def write_positive(f, img, box, split, rng):
    cx, cy = box
    x0, y0 = crop_origin(cx, cy, rng)
    crop = img[y0:y0 + TILE, x0:x0 + TILE]
    name = f"f{f:06d}"
    cv2.imwrite(os.path.join(OUT, "images", split, name + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as fh:
        fh.write(f"0 {(cx - x0) / TILE:.6f} {(cy - y0) / TILE:.6f} {BOX / TILE:.6f} {BOX / TILE:.6f}\n")


def write_negative(f, img, box, split, rng):
    cx, cy = box
    for _ in range(20):
        x0, y0 = int(rng.integers(0, W - TILE + 1)), int(rng.integers(0, H - TILE + 1))
        if not (x0 - 64 <= cx <= x0 + TILE + 64 and y0 - 64 <= cy <= y0 + TILE + 64):
            name = f"f{f:06d}_neg"
            cv2.imwrite(os.path.join(OUT, "images", split, name + ".jpg"), img[y0:y0 + TILE, x0:x0 + TILE],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(os.path.join(OUT, "labels", split, name + ".txt"), "w").close()
            return True
    return False


if __name__ == "__main__":
    main()
