"""v9: blend the gradient-boosting probabilities (v7) with the average of 3 sequence models trained on
noise-augmented data (3 degraded copies of each external match), then decode.
Lab (3 external matches): v7 0.595 -> v8 0.611 -> v9 0.629 spellF1, switching cost 10.
v10 = v9 with the video ball linked with mid-gap tracks (31% of lost frames recovered instead of 19%).
v11 = v10 with every possession counting equally in training (frame weight 1/length of its run), both models:
lab 0.629 -> 0.644; possessions < 2 s recovered 14% -> 22%, 2-5 s 48% -> 57%.
Probabilities are rounded to 3 decimals before decoding, so microscopic numeric differences between
computers cannot flip a decision (reproducibility).

    python -m possession.blend --name team_v9            (v8: add --original)
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
    ap.add_argument("--name", default="team_v9")
    ap.add_argument("--original", action="store_true", help="v8: sequence models without noise augmentation")
    ap.add_argument("--wpow", type=float, default=0.0, help="1 = possession-weighted sequence models (v11)")
    ap.add_argument("--wgb", type=float, default=0.5, help="weight of gradient boosting in the blend (v13: 0.4, chosen in the lab)")
    a = ap.parse_args()
    d = np.load(os.path.join(ROOT, "data", "cache", "final_inputs.npz"))
    p_gb, segs = d["p_gb"], [tuple(x) for x in d["segs"]]
    tag = ("" if a.original else "_aug") + ("" if not a.wpow else f"_w{a.wpow:g}")
    from possession.seq import inputs_fingerprint
    fp = inputs_fingerprint(os.path.join(ROOT, "data", "cache", "final_inputs.npz"))
    for s_ in SEEDS:
        side = os.path.join(ROOT, "data", "cache", f"seq_final{tag}_s{s_}.npy.inputs")
        if not os.path.exists(side) or open(side).read() != fp:
            raise SystemExit(f"sequence model {s_} was computed from different inputs: re-run python -m possession.seq final {s_}")
    p_seq = np.mean([np.load(os.path.join(ROOT, "data", "cache", f"seq_final{tag}_s{s}.npy")) for s in SEEDS], 0)
    p = np.round(a.wgb * p_gb + (1 - a.wgb) * p_seq, 3)
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
