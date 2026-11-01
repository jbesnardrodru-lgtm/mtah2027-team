"""Noise-augmented training: each training match in 3 differently-degraded copies (G k, G k r1, G k r2).
Tests stay on the original copies (G1-G3), so scores compare with earlier results.
    python -m possession.exp_aug gb <test>      -> data/cache/aug_gb_P<test>.npy
"""
import sys, gc, numpy as np
from possession.featcache import get
from possession.model import fit, proba
from possession.exp_dead import columns

kind, test = sys.argv[1], int(sys.argv[2])
cols = columns(get("G1")[1], "v3")
tr = [k for k in (1, 2, 3) if k != test]
Fs, Ls = [], []
for k in tr:
    for suf in ("", "r1", "r2"):
        F, _, l = get(f"G{k}{suf}")
        Fs.append(F[:, cols]); Ls.append(l)
clf = fit(Fs, Ls, step=4)
del Fs, Ls; gc.collect()
np.save(f"data/cache/aug_gb_P{test}.npy", proba(clf, get(f"G{test}")[0][:, cols]).astype(np.float32))
print("done", kind, test)
