"""Watch the system in action: a clip of the tactical video with our predictions drawn on top.

    python video/render_clip.py --minute 20 --seconds 60            (minute of the 1st half; --half 2 for the 2nd)
Output: data/clips/clip_h<half>_m<minute>.mp4 (stays on this machine: it is match footage).
Drawn: banner with the predicted possession (v9 by default), players coloured by team, and the ball by source:
green = tracking, yellow = our video detector, cyan = broadcast feed.
"""
import argparse, os, subprocess
import numpy as np, pandas as pd, cv2
from common import ROOT, asset_paths, homography, project, half_video_blocks, GAME

COL = {"H": (60, 20, 150), "A": (40, 40, 220), "D": (120, 120, 120), "X": (60, 60, 60)}   # BGR
TXT = {"H": "BARCELONA", "A": "ATHLETIC", "D": "BALL DEAD", "X": "-"}


def head_marker(img, Mf, x, y, side):
    """Small triangle above the player's head (keeps the feet and the ball visible). Height follows the local
    perspective: pixels per metre measured from the homography at the player's position."""
    u, v = project(Mf, x, y)
    u2, v2 = project(Mf, x, y + 1 / 68.0)
    ppm = float(np.hypot(u2 - u, v2 - v))
    half_w = max(11.0, 0.45 * ppm)          # in full-resolution pixels; the clip is shown at 2/3 size
    height = max(16.0, 0.65 * ppm)
    tip = v - max(48.0, 2.2 * ppm)          # tip of the triangle, just above the head
    pts = np.array([[u, tip], [u - half_w, tip - height], [u + half_w, tip - height]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255) if side == 0 else (0, 0, 0))
    cv2.polylines(img, [pts], True, (0, 0, 0) if side == 0 else (255, 255, 255), 2, cv2.LINE_AA)


class Writer:
    """MP4 writer that works without extra installs: ffmpeg if present, else macOS's own H.264 encoder via
    OpenCV ('avc1'), else plain MPEG-4 ('mp4v')."""
    def __init__(self, out, W, H):
        import shutil
        self.ff = None
        if shutil.which("ffmpeg"):
            self.ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                        "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-c:v", "libx264", "-preset",
                                        "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
            return
        for code in ("avc1", "mp4v"):
            self.cv = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*code), 25, (W, H))
            if self.cv.isOpened():
                print("encoder:", code)
                return
        raise RuntimeError("no video encoder available")

    def write(self, img):
        if self.ff:
            self.ff.stdin.write(img.tobytes())
        else:
            self.cv.write(img)

    def close(self):
        if self.ff:
            self.ff.stdin.close(); self.ff.wait()
        else:
            self.cv.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--minute", type=float, default=20)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--sub", default="evaluation/results/team_v11.csv")
    a = ap.parse_args()
    blocks = {h: (s, e) for h, s, e in half_video_blocks(GAME)}
    f0 = int(blocks[a.half][0] + a.minute * 60 * 25 + (2934 if a.half == 2 else 0))
    n = int(a.seconds * 25)
    lab = pd.read_csv(os.path.join(ROOT, a.sub)).label.values
    d = np.load(os.path.join(ROOT, "data", "cache", "tactical.npz")); P, tball, team = d["P"], d["ball"], d["team"]
    vball = np.load(os.path.join(ROOT, "data", "cache", "ball_video_cpu_v2.npz"))["ball"]
    bpath = os.path.join(ROOT, "data", "cache", "broadcast_mapped.npz")
    bball = np.load(bpath)["ball"] if os.path.exists(bpath) else np.full_like(tball, np.nan)
    M, _ = homography()
    os.makedirs(os.path.join(ROOT, "data", "clips"), exist_ok=True)
    out = os.path.join(ROOT, "data", "clips", f"clip_h{a.half}_m{a.minute:g}.mp4")
    W, H = 1280, 720
    writer = Writer(out, W, H)
    import av
    with av.open(asset_paths()["video"]) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        tb, fps = float(s.time_base), float(s.average_rate)
        c.seek(int(f0 / fps / tb), stream=s, backward=True, any_frame=False)
        written = 0
        for fr in c.decode(s):
            idx = int(round(fr.pts * tb * fps))
            if idx < f0:
                continue
            if idx >= f0 + n:
                break
            img = fr.to_ndarray(format="bgr24")
            for j in range(P.shape[1]):
                x, y = P[idx, j, 0], P[idx, j, 1]
                if np.isnan(x):
                    continue
                head_marker(img, M[idx], x, y, team[j])
            ball_drawn = False
            for src, col in ((tball, (0, 220, 0)), (vball, (0, 220, 255)), (bball, (255, 220, 0))):
                if not ball_drawn and not np.isnan(src[idx, 0]):
                    u, v = project(M[idx], src[idx, 0], src[idx, 1])
                    cv2.circle(img, (int(u), int(v)), 16, col, 3); ball_drawn = True
            img = cv2.resize(img, (W, H))
            L = lab[idx]
            cv2.rectangle(img, (0, 0), (W, 56), COL[L], -1)
            cv2.putText(img, TXT[L], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            mm = (idx - blocks[a.half][0]) / 25 / 60
            cv2.putText(img, f"half {a.half}  {int(mm):02d}:{int((mm % 1) * 60):02d}   frame {idx}", (W - 470, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "ball: green tracking | yellow our detector | cyan broadcast   players: white triangle Barca, black Athletic",
                        (20, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            writer.write(img); written += 1
    writer.close()
    print(f"wrote {out}: {written} frames ({written/25:.0f} s)")
    if written == 0:
        print("nothing written: check --half / --minute are inside the match")


if __name__ == "__main__":
    main()
