"""Ball detector on the HELD-OUT SoccerTrack match (2020 session, never trained on): for each crop, the most
confident ball detection; hit = within --tol px of the annotated ball. Off-the-shelf COCO model uses 'sports ball'.
    python video/eval_st.py --model data/ballmodel/run_st/weights/best.pt
    python video/eval_st.py --model yolo11s.pt --coco
"""
import argparse, glob, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--coco", action="store_true")
    ap.add_argument("--data", default="data/ball_ds_st_test")
    ap.add_argument("--tol", type=float, default=6.0)
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(os.path.join(ROOT, a.model) if not a.model.endswith("yolo11s.pt") else a.model)
    ims = sorted(glob.glob(os.path.join(ROOT, a.data, "images", "val", "*.jpg")))
    rows = []                                    # (has_ball, conf, hit)
    for i in range(0, len(ims), 16):
        batch = ims[i:i + 16]
        for f, r in zip(batch, m.predict(batch, imgsz=640, conf=0.01, device="cpu", verbose=False,
                                         classes=[32] if a.coco else None)):
            lab = open(f.replace("/images/", "/labels/")[:-4] + ".txt").read().split()
            gt = (float(lab[1]) * 640, float(lab[2]) * 640) if lab else None
            if len(r.boxes):
                j = int(r.boxes.conf.argmax()); c = float(r.boxes.conf[j]); x, y = r.boxes.xywh[j, :2].tolist()
                hit = gt is not None and np.hypot(x - gt[0], y - gt[1]) <= a.tol
            else:
                c, hit = 0.0, False
            rows.append((gt is not None, c, hit))
    R = np.array(rows, float)
    pos = R[:, 0] == 1
    print(f"{a.model}: {pos.sum():.0f} crops with ball, {(~pos).sum():.0f} without")
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        det = R[:, 1] >= t
        rec = np.mean(det[pos] & (R[pos, 2] == 1))
        prec = np.sum(det & (R[:, 2] == 1)) / max(det.sum(), 1)
        print(f"  conf >= {t:.1f}: recall {rec:.0%}  precision {prec:.0%}  (false alarms on empty crops {np.mean(det[~pos]):.0%})")


if __name__ == "__main__":
    main()
