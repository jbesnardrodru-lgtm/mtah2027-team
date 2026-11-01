"""Turn per-frame detector candidates into a credible ball track inside each tracking gap.

For every gap in the tracking ball we walk forward from where the tracking last saw it, and backward
from where it sees it again, accepting a candidate only if:
  - confidence >= --conf
  - on the pitch (<= 1 m outside the lines)
  - physically reachable from the last accepted position (<= 30 m/s x elapsed + 2 m)
  - not a parked ball: a candidate that has not moved for 2 s while no player is within 8 m is a spare ball
Recovered positions get a light 5-frame median to match the smoothness of the tracking ball.

    python video/link_ball.py
Output: data/cache/ball_video.npz -> ball (N, 3) = tracking ball with gaps filled where the video is credible
"""
import argparse, os
import numpy as np
from common import ROOT, PITCH, tracking_ball

OUT = os.path.join(ROOT, "data", "cache", "ball_video.npz")


def walk(order, cand, start_xy, P_xy, a):
    """Greedy pass over gap frames in `order`; returns {frame: (x, y)} of accepted positions."""
    acc, last, last_f = {}, start_xy, None
    still_since, still_xy = None, None
    for f, c in order:
        best = None
        for x, y, conf in c:
            if np.isnan(conf) or conf < a.conf:
                continue
            if not (-1 / PITCH[0] <= x <= 1 + 1 / PITCH[0] and -1 / PITCH[1] <= y <= 1 + 1 / PITCH[1]):
                continue
            dt = abs(f - last_f) / 25 if last_f is not None else 1.0
            d = np.hypot((x - last[0]) * PITCH[0], (y - last[1]) * PITCH[1])
            if d > 30 * dt + 2:
                continue
            if best is None or conf > best[2]:
                best = (x, y, conf)
        if best is None:
            if last_f is not None and abs(f - last_f) > a.lost:   # lost it for too long: stop this walk
                break
            continue
        x, y, _ = best
        # parked-ball check
        if still_xy is not None and np.hypot((x - still_xy[0]) * PITCH[0], (y - still_xy[1]) * PITCH[1]) < 0.5:
            if abs(f - still_since) > 50:
                pl = P_xy[f]
                dmin = np.nanmin(np.hypot((pl[:, 0] - x) * PITCH[0], (pl[:, 1] - y) * PITCH[1])) if np.isfinite(pl).any() else np.inf
                if dmin > 8:
                    break
        else:
            still_xy, still_since = (x, y), f
        acc[f] = (x, y)
        last, last_f = (x, y), f
    return acc


def tracklets(frames, cand, P_xy, conf=0.4, min_len=15, max_gap=5):
    """Mid-gap tracks: chain confident candidates frame to frame (physically reachable), keep chains that last
    >= min_len frames on the pitch and either move (> 1 m overall) or stay near a player (not a parked ball)."""
    out = {}
    pos = {int(f): i for i, f in enumerate(frames)}
    active = []                                     # each: list of (frame, x, y, c)
    done = []
    for f in frames:
        f = int(f); c = cand[pos[f]]
        pts = [(x, y, q) for x, y, q in c if not np.isnan(q) and q >= conf
               and -1 / PITCH[0] <= x <= 1 + 1 / PITCH[0] and -1 / PITCH[1] <= y <= 1 + 1 / PITCH[1]]
        still = []
        for tr in active:
            lf, lx, ly, _ = tr[-1]
            if f - lf > max_gap:
                done.append(tr); continue
            best = None
            for x, y, q in pts:
                d = np.hypot((x - lx) * PITCH[0], (y - ly) * PITCH[1])
                if d <= 30 * (f - lf) / 25 + 1 and (best is None or d < best[0]):
                    best = (d, x, y, q)
            if best:
                tr.append((f, best[1], best[2], best[3])); pts = [p for p in pts if (p[0], p[1]) != (best[1], best[2])]
            still.append(tr)
        active = still + [[(f, x, y, q)] for x, y, q in pts]
    done += active
    for tr in done:
        if len(tr) < min_len:
            continue
        xs = np.array([[x, y] for _, x, y, _ in tr])
        moved = np.hypot((xs[:, 0].max() - xs[:, 0].min()) * PITCH[0], (xs[:, 1].max() - xs[:, 1].min()) * PITCH[1])
        fm, xm, ym, _ = tr[len(tr) // 2]
        pl = P_xy[fm]
        near = np.nanmin(np.hypot((pl[:, 0] - xm) * PITCH[0], (pl[:, 1] - ym) * PITCH[1])) if np.isfinite(pl).any() else np.inf
        if moved < 1 and near > 8:
            continue                                  # parked spare ball
        for fr, x, y, _ in tr:
            out[fr] = (x, y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--lost", type=int, default=50, help="frames without a credible candidate before a walk stops")
    ap.add_argument("--tracklets", action="store_true", help="also fill mid-gap frames with smooth confident tracks")
    ap.add_argument("--tconf", type=float, default=0.4)
    ap.add_argument("--tlen", type=int, default=15)
    ap.add_argument("--inp", default="data/cache/video_ball.npz")
    ap.add_argument("--out", default="data/cache/ball_video.npz")
    a = ap.parse_args()
    ball = tracking_ball()
    P = np.load(os.path.join(ROOT, "data", "cache", "tactical.npz"))["P"][:, :, :2]
    d = np.load(os.path.join(ROOT, a.inp))
    frames, cand = d["frames"], d["cand"]
    pos = {int(f): i for i, f in enumerate(frames)}
    out = ball.copy()
    seen = ~np.isnan(ball[:, 0])
    # gaps = maximal runs of consecutive processed frames
    breaks = np.flatnonzero(np.diff(frames) != 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(frames) - 1]
    filled = 0
    for s, e in zip(starts, ends):
        f0, f1 = int(frames[s]), int(frames[e])
        fwd_anchor = ball[f0 - 1, :2] if f0 > 0 and seen[f0 - 1] else None
        bwd_anchor = ball[f1 + 1, :2] if f1 + 1 < len(ball) and seen[f1 + 1] else None
        seq = [(int(frames[i]), cand[i]) for i in range(s, e + 1)]
        acc = {}
        if fwd_anchor is not None:
            acc.update(walk(seq, cand, fwd_anchor, P, a))
        if bwd_anchor is not None:
            back = walk(seq[::-1], cand, bwd_anchor, P, a)
            for f, xy in back.items():
                if f in acc and np.hypot((acc[f][0] - xy[0]) * PITCH[0], (acc[f][1] - xy[1]) * PITCH[1]) > 3:
                    continue      # the two walks disagree: trust neither
                acc.setdefault(f, xy)
        if a.tracklets:
            for f, xy in tracklets(frames[s:e + 1], cand[s:e + 1], P, conf=a.tconf, min_len=a.tlen).items():
                acc.setdefault(f, xy)
        if not acc:
            continue
        fs = np.array(sorted(acc))
        xy = np.array([acc[f] for f in fs])
        # light 5-frame median within contiguous runs
        for j in range(2):
            col = xy[:, j].copy()
            for i in range(len(fs)):
                w = (np.abs(fs - fs[i]) <= 2)
                col[i] = np.median(xy[w, j])
            out[fs, j] = col
        filled += len(fs)
    graded_missing = len(frames)
    np.savez(os.path.join(ROOT, a.out), ball=out)
    print(f"gaps {len(starts)}  frames filled {filled} of {graded_missing} lost-ball frames ({filled / graded_missing:.0%})")
    print(f"ball now visible on {np.mean(~np.isnan(out[11185:190198, 0])):.0%} of graded frames (was {np.mean(seen[11185:190198]):.0%})")


if __name__ == "__main__":
    main()
