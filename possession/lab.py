"""Score any per-frame predictor on the labelled external games with the OFFICIAL metric code."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession.external import load_game  # noqa: E402


def score(gt_labels, pred_labels):
    r = pm.evaluate(pm.label_codes(gt_labels), pm.label_codes(pred_labels), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"]


def nearest_codes(P, ball, team, pitch=(105.0, 68.0)):
    """The official baseline's step 1, vectorised: team of nearest player, -1 where no ball."""
    s = np.array(pitch, np.float32)
    d = np.linalg.norm((P[:, :, :2] - ball[:, None, :2]) * s, axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    best = d.argmin(1)
    out = team[best].astype(np.int8)
    out[np.isinf(d.min(1))] = -1
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    from baseline import smooth
    for g in (1, 2):
        P, ball, team, label, period = load_game(g)
        codes = nearest_codes(P, ball, team)
        for w in (0, 25, 50, 125):
            sf, mf = score(label, smooth(codes, w))
            print(f"game {g}  window {w:>3}  spellF1 {sf:.4f}  macroF1 {mf:.4f}")
