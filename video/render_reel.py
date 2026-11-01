"""Highlight reel of OUR ball detector at work: the moments in live play where the tracking has lost the ball
and the yellow ring (our detector) follows a moving ball. Segments with 1 s of context, in match order.

    python video/render_reel.py --seconds 90
Output: data/clips/yellow_reel.mp4 (stays on this machine: it is match footage).
"""
import argparse, os
import numpy as np, pandas as pd, cv2
from common import ROOT, asset_paths, homography, project, half_video_blocks, GAME
from render_clip import Writer, COL, TXT


def segments(total_s, min_len=10, pad=25):
    t = np.load(os.path.join(ROOT, "data", "cache", "tactical.npz"))["ball"]
    v = np.load(os.path.join(ROOT, "data", "cache", "ball_video_cpu_v2.npz"))["ball"]
    lab = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "team_v11.csv")).label.values
    n = len(t); g = np.zeros(n, bool); g[11185:90652] = True; g[113742:190198] = True
    x = v[:, 0] * 105.3; y = v[:, 1] * 68
    sp = np.full(n, np.nan); sp[5:] = np.hypot(x[5:] - x[:-5], y[5:] - y[:-5]) * 5
    yl = g & np.isnan(t[:, 0]) & ~np.isnan(v[:, 0]) & np.isin(lab, ["H", "A"]) & (np.nan_to_num(sp) > 1.5)
    e = np.diff(np.r_[0, yl.astype(int), 0]); st, en = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    runs = sorted([(b - a, a, b) for a, b in zip(st, en) if b - a >= min_len], reverse=True)
    chosen, used = [], 0
    for L, a, b in runs:
        if used >= total_s * 25:
            break
        chosen.append((max(0, a - pad), b + pad)); used += b - a + 2 * pad
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90)
    a = ap.parse_args()
    segs = segments(a.seconds)
    lab = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "team_v11.csv")).label.values
    d = np.load(os.path.join(ROOT, "data", "cache", "tactical.npz")); P, tball, team = d["P"], d["ball"], d["team"]
    vball = np.load(os.path.join(ROOT, "data", "cache", "ball_video_cpu_v2.npz"))["ball"]
    M, _ = homography()
    blocks = {h: (s, e) for h, s, e in half_video_blocks(GAME)}
    os.makedirs(os.path.join(ROOT, "data", "clips"), exist_ok=True)
    out = os.path.join(ROOT, "data", "clips", "yellow_reel.mp4")
    W, H = 1280, 720
    writer = Writer(out, W, H)
    import av
    written = 0
    with av.open(asset_paths()["video"]) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        tb, fps = float(s.time_base), float(s.average_rate)
        for k, (f0, f1) in enumerate(segs):
            c.seek(int(f0 / fps / tb), stream=s, backward=True, any_frame=False)
            for fr in c.decode(s):
                idx = int(round(fr.pts * tb * fps))
                if idx < f0:
                    continue
                if idx >= f1:
                    break
                img = fr.to_ndarray(format="bgr24")
                for j in range(P.shape[1]):
                    if np.isnan(P[idx, j, 0]):
                        continue
                    u, v = project(M[idx], P[idx, j, 0], P[idx, j, 1])
                    cv2.circle(img, (int(u), int(v)), 9, (255, 255, 255) if team[j] == 0 else (0, 0, 0), 2)
                if not np.isnan(tball[idx, 0]):
                    u, v = project(M[idx], tball[idx, 0], tball[idx, 1]); cv2.circle(img, (int(u), int(v)), 16, (0, 220, 0), 3)
                elif not np.isnan(vball[idx, 0]):
                    u, v = project(M[idx], vball[idx, 0], vball[idx, 1]); cv2.circle(img, (int(u), int(v)), 16, (0, 220, 255), 4)
                img = cv2.resize(img, (W, H))
                L = lab[idx]
                cv2.rectangle(img, (0, 0), (W, 56), COL[L], -1)
                cv2.putText(img, TXT[L], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                h = 1 if idx < 100000 else 2
                mm = (idx - blocks[h][0] - (0 if h == 1 else 2934)) / 1500
                cv2.putText(img, f"moment {k+1}/{len(segs)}   half {h} {int(mm):02d}:{int((mm % 1) * 60):02d}", (W - 520, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img, "green = organisers' tracking   yellow = OUR detector (tracking had lost the ball)",
                            (20, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                writer.write(img); written += 1
    writer.close()
    print(f"wrote {out}: {len(segs)} moments, {written/25:.0f} s")


if __name__ == "__main__":
    main()
