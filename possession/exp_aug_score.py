import sys, numpy as np
from possession.featcache import get
from possession.model import decode, graded_segments
from possession.lab import score
from possession.errors import spells
ref = np.load("data/cache/exp_tv_probs.npz")
src = {
    "GB (v7)": lambda g: ref[f"P{g}"],
    "GB augmented": lambda g: np.load(f"data/cache/aug_gb_P{g}.npy"),
    "seq (v8 part)": lambda g: np.load(f"data/cache/seqbens_P{g}.npy"),
}
import os
if os.path.exists("data/cache/aug_seq_P1.npy"):
    src["seq augmented"] = lambda g: np.load(f"data/cache/aug_seq_P{g}.npy")
blends = {"blend v8": ("GB (v7)", "seq (v8 part)"), "blend GB-aug + seq": ("GB augmented", "seq (v8 part)")}
if "seq augmented" in src:
    blends["blend both augmented"] = ("GB augmented", "seq augmented")
for k, (a, b) in blends.items():
    src[k] = (lambda a, b: lambda g: 0.5 * src[a](g) + 0.5 * src[b](g))(a, b)
for name, f in src.items():
    best = None
    for sh in (10, 20):
        r = []; n = 0
        for g in (1, 2, 3):
            l = get(f"G{g}")[2]; pred = decode(f(g), graded_segments(l), sh, 80); r.append(score(l, pred)); n += len(spells(pred))
        m = np.mean([x[0] for x in r])
        print(f"{name:<22} ha {sh}: spellF1 " + " ".join(f"{x[0]:.3f}" for x in r) + f"  mean {m:.3f}  macroF1 {np.mean([x[1] for x in r]):.3f} | pred {n}/1043", flush=True)
