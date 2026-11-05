"""Ball-detection training set from EXTERNAL footage only: SoccerTrack wide view (Scott et al., CVPRW 2022,
LGPL-3.0; fixed panoramic camera over the whole pitch, ball box on every frame). Nothing from the provided match.

Frames (6500 x 1000) are shrunk by a random factor in [--smin, --smax] so the ball ends up a few pixels wide,
the size it has in a full-pitch broadcast-free camera; scale is randomised, not calibrated on our video.
640 x 640 crops: one around the ball (random offset) per sampled frame, plus ball-free negatives.
Split by MATCH: 2022 session = train, 2020 session = held-out test (never trained on).
    python video/make_st_dataset.py --n 6000 --out data/ball_ds_st
    python video/make_st_dataset.py --split test --n 1500 --out data/ball_ds_st_test
"""
import argparse, glob, os
import numpy as np, cv2, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST = os.path.join(ROOT, "data", "soccertrack", "wv")
C = 640


def clips(split):
    tag = "F_20220220" if split == "train" else "F_20200220"
    return sorted(glob.glob(os.path.join(ST, "videos", f"{tag}_*.mp4")))


def ball_boxes(video):
    a = pd.read_csv(os.path.join(ST, "annotations", os.path.basename(video)[:-4] + ".csv"),
                    header=[0, 1, 2], index_col=0)["BALL"]["BALL"]
    return {int(i) - 1: (r.bb_left, r.bb_top, r.bb_width, r.bb_height) for i, r in a.iterrows() if pd.notna(r.bb_left)}


def crop_around(img, cx, cy, rng):
    h, w = img.shape[:2]
    canvas = np.full((max(h, C), max(w, C), 3), 114, np.uint8)
    oy = (canvas.shape[0] - h) // 2
    canvas[oy:oy + h, :w] = img
    cy = cy + oy
    x0 = int(np.clip(cx - rng.uniform(40, C - 40), 0, canvas.shape[1] - C))
    y0 = int(np.clip(cy - rng.uniform(40, C - 40), 0, canvas.shape[0] - C))
    return canvas[y0:y0 + C, x0:x0 + C], x0, y0 - oy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=6000, help="positive crops")
    ap.add_argument("--neg_every", type=int, default=3)
    ap.add_argument("--smin", type=float, default=0.25)
    ap.add_argument("--smax", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/ball_ds_st")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = os.path.join(ROOT, a.out)
    sub = "train" if a.split == "train" else "val"
    for d in ("images", "labels"):
        os.makedirs(os.path.join(out, d, sub), exist_ok=True)
    vids = clips(a.split)
    per_clip = int(np.ceil(a.n / len(vids)))
    k = 0
    for v in vids:
        boxes = ball_boxes(v)
        pick = set(rng.choice(sorted(boxes), size=min(per_clip, len(boxes)), replace=False).tolist())
        cap = cv2.VideoCapture(v); f = -1
        while pick:
            ok, img = cap.read(); f += 1
            if not ok:
                break
            if f not in pick:
                continue
            pick.discard(f)
            s = rng.uniform(a.smin, a.smax)
            im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            bl, bt, bw, bh = (x * s for x in boxes[f])
            cx, cy = bl + bw / 2, bt + bh / 2
            crop, x0, y0 = crop_around(im, cx, cy, rng)
            name = f"st_{k:06d}"
            cv2.imwrite(os.path.join(out, "images", sub, name + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            bwn, bhn = max(bw, 2) / C, max(bh, 2) / C
            with open(os.path.join(out, "labels", sub, name + ".txt"), "w") as fh:
                fh.write(f"0 {(cx - x0) / C:.6f} {(cy - y0) / C:.6f} {bwn:.6f} {bhn:.6f}\n")
            if k % a.neg_every == 0:                       # a ball-free crop from the same frame
                for _ in range(10):
                    ncrop, nx0, ny0 = crop_around(im, rng.uniform(0, im.shape[1]), cy, rng)
                    if not (nx0 - 10 <= cx <= nx0 + C + 10 and ny0 - 10 <= cy <= ny0 + C + 10):
                        cv2.imwrite(os.path.join(out, "images", sub, name + "_neg.jpg"), ncrop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        open(os.path.join(out, "labels", sub, name + "_neg.txt"), "w").close()
                        break
            k += 1
        cap.release()
        print(os.path.basename(v), k, flush=True)
    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write(f"path: {out}\ntrain: images/train\nval: images/{'train' if a.split == 'train' else 'val'}\nnames:\n  0: ball\n")
    print("done", k, "positive crops in", out)


if __name__ == "__main__":
    main()
