"""Cache feature matrices (lab games degraded + our match) so experiments run in seconds."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from possession.model import lab_game  # noqa: E402
from possession.cache import load  # noqa: E402
from possession.features import build  # noqa: E402

DIR = os.path.join(ROOT, "data", "cache")


def get(name, rebuild=False):
    path = os.path.join(DIR, f"feat_{name}.npz")
    if os.path.exists(path) and not rebuild:
        d = np.load(path, allow_pickle=True)
        return d["F"], list(d["names"]), d["label"]
    if name.startswith("G"):
        uniform = name.endswith("u")
        miss = float(name.split("m")[1]) / 100 if "m" in name else 0.41   # e.g. G1m27 = 27% ball missing
        F, names, label = lab_game(int(name[1:2]), seed=100 if uniform else 0,
                                   placement="uniform" if uniform else "weighted", target_missing=miss)
    else:
        P, ball, team = load("tactical")
        segs = [(s, e + 1) for _, s, e in half_video_blocks(GAME)]
        F, names = build(P, ball, team, segs, pitch=GAMES[GAME]["pitch"])
        label = np.full(len(F), "X", "<U1")
        for a, b in segs:
            label[a:b] = "?"
    np.savez(path, F=F, names=np.array(names), label=label)
    return F, names, label


if __name__ == "__main__":
    for n in sys.argv[1:]:
        F, names, _ = get(n, rebuild=True)
        print(n, F.shape)
