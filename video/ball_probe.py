"""Go/no-go test for video ball detection. No human labels anywhere.

  Test A  held-out graded frames WHERE TRACKING HAS THE BALL (never used for training)
          -> is the detector's BEST candidate within 25 px of the tracking ball?
  Test B  frames just inside a tracking gap (0.2-1 s after the ball was lost)
          -> is the best candidate where a ball could physically be, given where it was last seen?
             (<= 30 m/s x elapsed time + 2 m). Near the gap edge this check is tight and meaningful.

    python video/ball_probe.py --n 300                                    # off-the-shelf model
    python video/ball_probe.py --n 300 --model data/ballmodel/run/weights/best.pt --tiled
Outputs: summary + data/probe/contact_A.jpg, contact_B.jpg (for a visual check, never for labelling).
"""
import argparse, os, time
import numpy as np, cv2
from common import (ROOT, W, H, PITCH, asset_paths, tracking_ball, homography, project, unproject,
                    graded_mask, block_split, read_frames, detect)

OUT = os.path.join(ROOT, "data", "probe")


def contact_sheet(items, path, size=160):
    tiles = []
    for img, (u, v), tag in items[:48]:
        u, v = int(np.clip(u, 120, W - 120)), int(np.clip(v, 120, H - 120))
        t = cv2.resize(img[v - 120:v + 120, u - 120:u + 120], (size, size))
        cv2.circle(t, (size // 2, size // 2), 14, (0, 255, 255), 1)
        cv2.putText(t, tag, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 8:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--tiled", action="store_true", help="3x2 native-resolution tiles (use for the fine-tuned model)")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)
    ball = tracking_ball()
    M, valid = homography()
    n = len(ball)
    f_all = np.arange(n)
    seen = ~np.isnan(ball[:, 0])
    base = graded_mask(n) & valid

    # Test A: held-out blocks only, ground ball, inside the image
    a_pool = f_all[base & seen & np.isnan(ball[:, 2]) & block_split(f_all)]
    fa = np.sort(rng.choice(a_pool, a.n, replace=False))

    # Test B: 5-25 frames after the tracking last saw the ball (gap frames near the edge)
    last = np.maximum.accumulate(np.where(seen, f_all, -1))
    since = f_all - last
    b_pool = f_all[base & ~seen & (last >= 0) & (since >= 5) & (since <= 25)]
    fb = np.sort(rng.choice(b_pool, min(a.n, len(b_pool)), replace=False))

    model = YOLO(a.model)
    os.makedirs(OUT, exist_ok=True)
    video = asset_paths()["video"]
    print(f"device {device}  model {os.path.basename(a.model)}  tiled {a.tiled}  frames {len(fa)}+{len(fb)}")

    def run(frames):
        t0 = time.time()
        imgs, dets, buf = {}, {}, []
        for f, img in read_frames(video, frames):
            buf.append((f, img))
            if len(buf) == 8:
                for (g, im), d in zip(buf, detect(model, [x for _, x in buf], device, a.conf, a.tiled)):
                    imgs[g], dets[g] = im, d
                buf = []
        if buf:
            for (g, im), d in zip(buf, detect(model, [x for _, x in buf], device, a.conf, a.tiled)):
                imgs[g], dets[g] = im, d
        print(f"  {len(imgs) / (time.time() - t0):.1f} frames/s (decode + detect)")
        return imgs, dets

    # ---- Test A
    imgs, dets = run(fa)
    err, conf, sheet = [], [], []
    for f in fa:
        d = dets[f]
        u, v = project(M[f], ball[f, 0], ball[f, 1])
        if len(d):
            k = int(d[:, 2].argmax())
            e, c = np.hypot(d[k, 0] - u, d[k, 1] - v), d[k, 2]
        else:
            e, c = np.inf, 0.0
        err.append(e); conf.append(c)
        sheet.append((imgs[f], (u, v), f"{f} {'hit' if e < 25 else 'miss'}"))
    err, conf = np.array(err), np.array(conf)
    print("TEST A  held-out frames, tracking has the ball. Best candidate within:")
    for px in (10, 25, 50):
        print(f"    {px:>3} px: {np.mean(err < px):.0%}")
    for t in (0.25, 0.5):
        m = conf >= t
        print(f"    confidence >= {t}: on {m.mean():.0%} of frames, correct (<25 px) {np.mean(err[m] < 25) if m.any() else 0:.0%}")
    contact_sheet(sheet, os.path.join(OUT, "contact_A.jpg"))

    # ---- Test B
    imgs, dets = run(fb)
    res, sheet = [], []
    for f in fb:
        d = dets[f]
        if not len(d):
            res.append((0.0, False)); continue
        k = int(d[:, 2].argmax())
        x, y = unproject(M[f], d[k, 0], d[k, 1])
        j = last[f]
        dist = np.hypot((x - ball[j, 0]) * PITCH[0], (y - ball[j, 1]) * PITCH[1])
        ok = dist <= 30 * (f - j) / 25 + 2
        res.append((d[k, 2], ok))
        sheet.append((imgs[f], (d[k, 0], d[k, 1]), f"{f} {d[k,2]:.2f} {'ok' if ok else 'far'}"))
    c = np.array([r[0] for r in res]); ok = np.array([r[1] for r in res])
    print("TEST B  0.2-1 s into a tracking gap. Best candidate physically reachable from the last seen ball:")
    print(f"    any candidate, reachable: {ok.mean():.0%}")
    for t in (0.25, 0.5):
        m = c >= t
        print(f"    confidence >= {t}: on {m.mean():.0%} of frames, reachable {np.mean(ok[m]) if m.any() else 0:.0%}")
    contact_sheet(sheet, os.path.join(OUT, "contact_B.jpg"))
    print(f"contact sheets in {OUT}")
    print("GO if Test A >= ~70% within 25 px AND Test B reachable >= ~50% at a usable confidence.")


if __name__ == "__main__":
    main()
