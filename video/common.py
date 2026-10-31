"""Shared helpers for the video work: tracking ball, homography, frame-exact decoding, tiled detection."""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, asset_paths, half_video_blocks  # noqa: E402

FPS = 25
PITCH = np.array([105.3, 68.0])
W, H = 1920, 1080
TILE = 640
# 3 x 2 tiles cover 1920x1080 at native resolution (vertical overlap 200 px)
TILE_ORIGINS = [(x, y) for y in (0, H - TILE) for x in (0, TILE, 2 * TILE)]


def tracking_ball():
    return np.load(os.path.join(ROOT, "data", "cache", "tactical.npz"))["ball"]


def homography():
    h = pd.read_csv(asset_paths()["homography"])
    M = h[[f"h{i}{j}" for i in range(3) for j in range(3)]].values.reshape(-1, 3, 3)
    return M, h["valid"].values.astype(bool)


def project(M, x, y):
    p = M @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def unproject(M, u, v):
    p = np.linalg.inv(M) @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


def graded_mask(n):
    g = np.zeros(n, bool)
    for _, s, e in half_video_blocks(GAME):
        g[s:e + 1] = True
    return g


def block_split(frames, every=5, block_s=120):
    """Time-block split so train and test frames never come from the same 2-minute stretch."""
    blk = np.asarray(frames) // (FPS * block_s)
    return blk % every == 0  # True = held out


def read_frames(path, frames, chunk=48):
    """Yield (frame_index, bgr image) for the requested frames, frame-exact, in sorted order."""
    import av
    want = sorted(set(int(f) for f in frames))
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.thread_count = 2
        tb, fps = float(s.time_base), float(s.average_rate)
        for f in want:
            c.seek(int(f / fps / tb), stream=s, backward=True, any_frame=False)
            for fr in c.decode(s):
                idx = int(round(fr.pts * tb * fps))
                if idx == f:
                    yield f, fr.to_ndarray(format="bgr24")
                    break
                if idx > f:
                    break


def ball_class(model):
    names = {v: k for k, v in model.names.items()}
    return names.get("sports ball", names.get("ball", 0))


def detect(model, imgs, device, conf, tiled=False, imgsz=1920):
    """Per image: array (k, 3) of ball candidates u, v, confidence in full-frame pixels."""
    cls = ball_class(model)
    if not tiled:
        res = model.predict(imgs, imgsz=imgsz, conf=conf, classes=[cls], device=device, verbose=False)
        return [np.c_[r.boxes.xywh.cpu().numpy()[:, :2], r.boxes.conf.cpu().numpy()] for r in res]
    tiles = [img[y:y + TILE, x:x + TILE] for img in imgs for (x, y) in TILE_ORIGINS]
    res = model.predict(tiles, imgsz=TILE, conf=conf, classes=[cls], device=device, verbose=False)
    out = []
    for i in range(len(imgs)):
        parts = []
        for j, (x, y) in enumerate(TILE_ORIGINS):
            r = res[i * len(TILE_ORIGINS) + j]
            if len(r.boxes):
                xy = r.boxes.xywh.cpu().numpy()[:, :2] + [x, y]
                parts.append(np.c_[xy, r.boxes.conf.cpu().numpy()])
        out.append(np.vstack(parts) if parts else np.zeros((0, 3)))
    return out
