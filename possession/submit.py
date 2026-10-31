"""Build a full submission for BAR-ATH from the core model.

    python -m possession.submit --name team_v1
Trains on the (degraded) lab games only, never on any label of this match.
"""
import argparse, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from possession.cache import load  # noqa: E402
from possession.features import build  # noqa: E402
from possession.model import lab_game, fit, proba, decode, STATES  # noqa: E402

SWITCH_HA, SWITCH_D = 20, 80          # chosen in the lab (realistic ball noise + touch features, cross-game)
SECOND_HALF_KICKOFF = 113742          # published in FAQ.md / DATA.md: graded from 110808, all D before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="team_v1")
    ap.add_argument("--video", nargs="?", const="data/cache/ball_video_cpu.npz", default=None,
                    help="fill tracking gaps with the linked video ball (default: reproducible CPU version)")
    ap.add_argument("--broadcast", action="store_true", help="fill remaining gaps with the synced broadcast ball")
    a = ap.parse_args()

    from possession.featcache import get
    from possession.exp_dead import columns
    games = [get(f"G{g}") for g in (1, 2, 3)]        # built (and cached) from scratch if missing
    cols = columns(games[0][1], "v3")
    clf = fit([g[0][:, cols] for g in games], [g[2] for g in games], step=4)

    P, ball, team = load("tactical")
    if a.video:
        ball = np.load(os.path.join(ROOT, a.video))["ball"]
    if a.broadcast:
        path = os.path.join(ROOT, "data", "cache", "broadcast_mapped.npz")
        if not os.path.exists(path):
            from possession.broadcast_sync import build_mapping
            build_mapping(path)
        bm = np.load(path)["ball"]
        gap = np.isnan(ball[:, 0]) & ~np.isnan(bm[:, 0])
        ball = ball.copy()
        ball[gap, :2] = bm[gap, :2]
    segs = [(s, e + 1) for _, s, e in half_video_blocks(GAME)]
    F, _ = build(P, ball, team, segs, pitch=GAMES[GAME]["pitch"])
    p = proba(clf, F[:, cols])
    p[segs[1][0]:SECOND_HALF_KICKOFF] = [0, 0, 1]
    labels = decode(p, segs, SWITCH_HA, SWITCH_D)

    out = os.path.join(ROOT, "evaluation", "results", a.name + ".csv")
    with open(out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    g = labels != "X"
    runs = np.diff(np.r_[0, (labels[1:] != labels[:-1])]).sum()
    print(f"wrote {out}\n  graded H {np.mean(labels[g]=='H'):.2f} A {np.mean(labels[g]=='A'):.2f} D {np.mean(labels[g]=='D'):.2f}")


if __name__ == "__main__":
    main()
