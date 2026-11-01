"""v8: blend the gradient-boosting probabilities (v7) with the average of 3 sequence models, then decode.
Lab (3 external matches): v7 0.595 -> blend 0.611 spellF1, switching cost 10 (python -m possession.exp_seq_score bens).

    python -m possession.blend --name team_v8
Needs data/cache/final_inputs.npz (possession.submit) and data/cache/seq_final_s{0,1,2}.npy (possession.seq final).
"""
import argparse, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.model import decode  # noqa: E402
from possession.submit import SECOND_HALF_KICKOFF  # noqa: E402

SWITCH_HA, SWITCH_D = 10, 80     # best blend setting in the lab
SEEDS = (0, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="team_v8")
    a = ap.parse_args()
    d = np.load(os.path.join(ROOT, "data", "cache", "final_inputs.npz"))
    p_gb, segs = d["p_gb"], [tuple(x) for x in d["segs"]]
    p_seq = np.mean([np.load(os.path.join(ROOT, "data", "cache", f"seq_final_s{s}.npy")) for s in SEEDS], 0)
    p = 0.5 * p_gb + 0.5 * p_seq
    p[segs[1][0]:SECOND_HALF_KICKOFF] = [0, 0, 1]
    labels = decode(p, segs, SWITCH_HA, SWITCH_D)
    out = os.path.join(ROOT, "evaluation", "results", a.name + ".csv")
    with open(out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    g = labels != "X"
    print(f"wrote {out}\n  graded H {np.mean(labels[g]=='H'):.2f} A {np.mean(labels[g]=='A'):.2f} D {np.mean(labels[g]=='D'):.2f}")


if __name__ == "__main__":
    main()
