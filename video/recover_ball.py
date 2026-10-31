"""Run the fine-tuned ball detector over every graded frame where the tracking lost the ball.

Keeps the top-5 candidates per frame (pitch coordinates + confidence), so a later linking step can
reject spare balls by the touchline (static, off the pitch) and pick the real one by trajectory.
Resumable: re-run the same command after an interruption and it continues where it stopped.

    python video/recover_ball.py
Output: data/cache/video_ball.npz  (frames, cand[n, 5, 3] = x, y (0-1 pitch), confidence; NaN = none)
"""
import argparse, os, time
import numpy as np
from common import ROOT, asset_paths, tracking_ball, homography, unproject, graded_mask, detect

OUT = os.path.join(ROOT, "data", "cache", "video_ball.npz")   # overridden by --out
PART = OUT.replace(".npz", "_partial.npz")


def quantise(x, y, c):
    """Round so tiny numerical differences between machines almost never change downstream results:
    position to 0.5 m on a 105 x 68 pitch, confidence to steps of 0.05."""
    return round(x * 210) / 210, round(y * 136) / 136, round(c * 20) / 20
K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "data", "ballmodel", "run", "weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after this many frames")
    ap.add_argument("--out", default=None, help="output npz (default data/cache/video_ball.npz)")
    a = ap.parse_args()
    global OUT, PART
    if a.out:
        OUT = os.path.join(ROOT, a.out)
        PART = OUT.replace(".npz", "_partial.npz")

    import av, torch
    from ultralytics import YOLO
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(4)
    ball = tracking_ball()
    M, valid = homography()
    want = np.flatnonzero(graded_mask(len(ball)) & valid & np.isnan(ball[:, 0]))
    if a.limit:
        want = want[: a.limit]
    cand = np.full((len(want), K, 3), np.nan, np.float32)
    done = np.zeros(len(want), bool)
    if os.path.exists(PART) and not a.limit:
        d = np.load(PART)
        if np.array_equal(d["frames"], want):
            cand, done = d["cand"], d["done"]
            print(f"resuming: {done.sum()} of {len(want)} frames already done")
    pos = {int(f): i for i, f in enumerate(want)}
    todo = set(int(f) for f in want[~done])
    model = YOLO(a.model)
    print(f"device {device}  frames to process {len(todo)}")

    t0, n_done, buf = time.time(), 0, []

    def flush():
        nonlocal n_done
        dets = detect(model, [img for _, img in buf], device, a.conf, tiled=True)
        for (f, _), d in zip(buf, dets):
            i = pos[f]
            if len(d):
                d = d[np.argsort(-d[:, 2])][:K]
                for k, (u, v, c) in enumerate(d):
                    x, y = unproject(M[f], u, v)
                    cand[i, k] = quantise(x, y, c)
            done[i] = True
        n_done += len(buf)
        buf.clear()

    with av.open(asset_paths()["video"]) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.thread_count = 2
        tb, fps = float(s.time_base), float(s.average_rate)
        first = min(todo) if todo else 0
        c.seek(int(first / fps / tb), stream=s, backward=True, any_frame=False)
        last_save = time.time()
        for fr in c.decode(s):
            idx = int(round(fr.pts * tb * fps))
            if idx > want[-1] or not todo:
                break
            if idx in todo:
                todo.discard(idx)
                buf.append((idx, fr.to_ndarray(format="bgr24")))
                if len(buf) == a.batch:
                    flush()
                if time.time() - last_save > 300:
                    np.savez(PART, frames=want, cand=cand, done=done)
                    last_save = time.time()
                    rate = n_done / (time.time() - t0)
                    print(f"  {done.sum()}/{len(want)}  {rate:.1f} frames/s  ~{(len(want)-done.sum())/rate/60:.0f} min left", flush=True)
        if buf:
            flush()
    if a.limit:
        print("debug run:", done.sum(), "frames,", np.mean(~np.isnan(cand[:, 0, 2])), "with a candidate")
        if a.out:
            np.savez(OUT, frames=want, cand=cand)
        return
    np.savez(PART, frames=want, cand=cand, done=done)
    if done.all():
        np.savez(OUT, frames=want, cand=cand)
        print(f"done: {len(want)} frames, {np.mean(~np.isnan(cand[:, 0, 2])):.0%} with a candidate -> {OUT}")
    else:
        print(f"stopped at {done.sum()}/{len(want)}; run again to resume")


if __name__ == "__main__":
    main()
