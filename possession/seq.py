"""Sequence model: a small non-causal temporal CNN over the per-frame features (5 steps per second).
Sees ~12 s before and after each moment. Outputs P(H), P(A), P(D) per frame (upsampled back to 25 fps).

    python -m possession.seq <test game 1|2|3>     # trains on the other two, saves data/cache/seq_P<test>.npy
"""
import sys, time
import numpy as np
import torch, torch.nn as nn

from possession.featcache import get
from possession.exp_dead import columns
from possession.model import STATES

import os
STEP = int(os.environ.get("SEQ_STEP", 2))     # 25 fps -> 25/STEP steps per second
CROP = int(os.environ.get("SEQ_CROP", 1024))   # training crop length (steps)
DIL = tuple(int(x) for x in os.environ.get("SEQ_DIL", "1,2,4,8,16,32").split(","))
TAG = os.environ.get("SEQ_TAG", "")
torch.set_num_threads(4)


class TCN(nn.Module):
    def __init__(self, n_in, ch=64, dil=DIL, k=5, p=0.2):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.blocks = nn.ModuleList()
        for d in dil:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, k, padding=d * (k - 1) // 2, dilation=d), nn.GELU(), nn.Dropout(p),
                nn.Conv1d(ch, ch, 1)))
        self.out = nn.Conv1d(ch, 3, 1)

    def forward(self, x):                      # x: (B, n_in, T)
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)                     # (B, 3, T)


def prep(F, cols, mu=None, sd=None):
    X = F[::STEP][:, cols].astype(np.float32)
    if mu is None:
        mu = np.nanmean(X, 0); sd = np.nanstd(X, 0) + 1e-6
    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).clip(-6, 6)
    return X, mu, sd


def target(label):
    y = np.full(len(label), -100, np.int64)
    for i, s in enumerate(STATES):
        y[label == s] = i
    return y[::STEP]


def train(Xs, ys, n_in, epochs=30, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = TCN(n_in)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    n_steps = sum(len(x) for x in Xs) // CROP * 2
    for ep in range(epochs):
        net.train(); tot = 0
        for _ in range(max(1, n_steps // 16)):
            xb, yb = [], []
            for _ in range(16):
                k = rng.integers(len(Xs)); s = rng.integers(0, len(Xs[k]) - CROP)
                xb.append(Xs[k][s:s + CROP].T); yb.append(ys[k][s:s + CROP])
            xb = torch.tensor(np.stack(xb)); yb = torch.tensor(np.stack(yb))
            loss = lossf(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    return net


def predict(net, X, n_frames):
    net.eval()
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor(X.T[None])), 1)[0].T.numpy()
    up = np.repeat(p, STEP, 0)[:n_frames]
    if len(up) < n_frames:
        up = np.vstack([up, np.repeat(up[-1:], n_frames - len(up), 0)])
    return up


def final(seed, epochs=None, augmented=True):
    """Train on all three external games, predict our match (features saved by possession.submit).
    augmented=True (v9): each game in 3 differently-degraded copies, 7 epochs (same training effort as the lab)."""
    names = [f"G{k}{suf}" for k in (1, 2, 3) for suf in (("", "r1", "r2") if augmented else ("",))]
    epochs = epochs or (7 if augmented else 20)
    G = {k: get(k) for k in names}
    cols = columns(G[names[0]][1], "v3")
    allX = np.vstack([G[k][0][::STEP][:, cols] for k in G])
    mu = np.nanmean(allX, 0); sd = np.nanstd(allX, 0) + 1e-6
    del allX
    Xs = [prep(G[k][0], cols, mu, sd)[0] for k in G]
    ys = [target(G[k][2]) for k in G]
    del G
    net = train(Xs, ys, len(cols), epochs, seed=seed)
    d = np.load("data/cache/final_inputs.npz")
    F = d["F"]
    X = (F[::STEP] - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).clip(-6, 6).astype(np.float32)
    out = f"data/cache/seq_final{'_aug' if augmented else ''}_s{seed}.npy"
    np.save(out, predict(net, X, len(F)).astype(np.float32))
    return out


def aug_fold(test, seed, epochs=10):
    """Lab fold with noise-augmented training (3 noisy copies per training match, same total training steps)."""
    names = [f"G{k}{suf}" for k in (1, 2, 3) if k != test for suf in ("", "r1", "r2")]
    G = {n: get(n) for n in names}
    cols = columns(get("G1")[1], "v3")
    allX = np.vstack([G[n][0][::STEP][:, cols] for n in names])
    mu = np.nanmean(allX, 0); sd = np.nanstd(allX, 0) + 1e-6
    del allX
    Xs = [prep(G[n][0], cols, mu, sd)[0] for n in names]
    ys = [target(G[n][2]) for n in names]
    del G
    net = train(Xs, ys, len(cols), epochs, seed=seed)
    F = get(f"G{test}")[0]
    Xt, _, _ = prep(F, cols, mu, sd)
    out = f"data/cache/aug_seq{'' if seed == 0 else f's{seed}'}_P{test}.npy"
    np.save(out, predict(net, Xt, len(F)).astype(np.float32))
    return out


if __name__ == "__main__" and sys.argv[1] == "aug":
    t0 = time.time()
    print("saved", aug_fold(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0), f"in {time.time()-t0:.0f}s")
    sys.exit(0)

if __name__ == "__main__" and sys.argv[1] == "final":
    t0 = time.time()
    print("saved", final(int(sys.argv[2])), f"in {time.time()-t0:.0f}s")
    sys.exit(0)

if __name__ == "__main__":
    test = int(sys.argv[1]); epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    t0 = time.time()
    G = {k: get(f"G{k}") for k in (1, 2, 3)}
    cols = columns(G[1][1], "v3")
    tr = [k for k in (1, 2, 3) if k != test]
    allX = np.vstack([G[k][0][::STEP][:, cols] for k in tr])
    mu = np.nanmean(allX, 0); sd = np.nanstd(allX, 0) + 1e-6
    Xs = [prep(G[k][0], cols, mu, sd)[0] for k in tr]
    ys = [target(G[k][2]) for k in tr]
    net = train(Xs, ys, len(cols), epochs, seed=seed)
    Xt, _, _ = prep(G[test][0], cols, mu, sd)
    np.save(f"data/cache/seq{TAG}{'' if seed == 0 else f's{seed}'}_P{test}.npy", predict(net, Xt, len(G[test][0])).astype(np.float32))
    print(f"test G{test} done in {time.time()-t0:.0f}s")
