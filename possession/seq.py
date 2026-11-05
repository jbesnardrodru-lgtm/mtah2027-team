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
EXTRA = os.environ.get("SEQ_EXTRA", "")      # "DFL": add the Bundesliga matches to the lab folds
VAR = os.environ.get("SEQ_VAR", "")          # lab variants: swap | mirror | m50 | pre (see aug_fold)
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


WEIGHT_POWER = float(os.environ.get("SEQ_WPOW", 0))     # 0 = every frame equal; 1 = every possession equal


def run_weight(label):
    w = np.zeros(len(label), np.float32)
    s = 0
    for i in range(1, len(label) + 1):
        if i == len(label) or label[i] != label[s]:
            w[s:i] = (1.0 / (i - s)) ** WEIGHT_POWER if label[s] in STATES else 0.0
            s = i
    w = w[::STEP]
    return w / max(w[w > 0].mean(), 1e-9) if (w > 0).any() else w


def target(label):
    y = np.full(len(label), -100, np.int64)
    for i, s in enumerate(STATES):
        y[label == s] = i
    return y[::STEP]


def train(Xs, ys, n_in, epochs=30, seed=0, ws=None, net=None, lr=2e-3):
    """net: continue training an existing network (fine-tuning) instead of starting from scratch."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = net or TCN(n_in)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    ws = ws or [np.ones(len(y), np.float32) for y in ys]
    n_steps = sum(len(x) for x in Xs) // CROP * 2
    for ep in range(epochs):
        net.train(); tot = 0
        for _ in range(max(1, n_steps // 16)):
            xb, yb, wb = [], [], []
            for _ in range(16):
                k = rng.integers(len(Xs)); s = rng.integers(0, len(Xs[k]) - CROP)
                xb.append(Xs[k][s:s + CROP].T); yb.append(ys[k][s:s + CROP]); wb.append(ws[k][s:s + CROP])
            xb = torch.tensor(np.stack(xb)); yb = torch.tensor(np.stack(yb)); wb = torch.tensor(np.stack(wb))
            l_ = lossf(net(xb), yb)
            m_ = (yb != -100).float()
            loss = (l_ * wb * m_).sum() / (m_.sum() + 1e-9)
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
    if VAR == "swap":                          # v15: + every Metrica copy with the two teams exchanged
        names += [n + "x" for n in names]
    if EXTRA == "DFL":                         # v14: + the 7 Bundesliga matches (provider flags as labels)
        from possession.dfl import MATCHES
        names += [f"DFL{m}" for m in MATCHES]
    G = {k: get(k) for k in names}
    cols = columns(G[names[0]][1], "v3")
    allX = np.vstack([G[k][0][::STEP][:, cols] for k in G])
    mu = np.nanmean(allX, 0); sd = np.nanstd(allX, 0) + 1e-6
    del allX
    Xs = [prep(G[k][0], cols, mu, sd)[0] for k in G]
    ys = [target(G[k][2]) for k in G]
    ws = [run_weight(G[k][2]) for k in G]
    del G
    net = train(Xs, ys, len(cols), epochs, seed=seed, ws=ws)
    d = np.load("data/cache/final_inputs.npz")
    F = d["F"]
    X = (F[::STEP] - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).clip(-6, 6).astype(np.float32)
    wtag = "" if WEIGHT_POWER == 0 else f"_w{WEIGHT_POWER:g}"
    tta = bool(os.environ.get("SEQ_TTA"))
    out = (f"data/cache/seq_final{'_aug' if augmented else ''}{wtag}{'_dfl' if EXTRA else ''}{f'_{VAR}' if VAR else ''}"
           f"{'_tta' if tta else ''}_s{seed}.npy")
    p = predict(net, X, len(F))
    if tta:                                    # v16: average with the prediction on the team-swapped match
        Xx = (d["Fx"][::STEP] - mu) / sd
        Xx = np.nan_to_num(Xx, nan=0.0, posinf=0.0, neginf=0.0).clip(-6, 6).astype(np.float32)
        p = 0.5 * (p + predict(net, Xx, len(F))[:, [1, 0, 2]])
    np.save(out, p.astype(np.float32))
    open(out + ".inputs", "w").write(inputs_fingerprint())
    return out


def inputs_fingerprint(path="data/cache/final_inputs.npz"):
    import hashlib
    return hashlib.md5(np.load(path)["F"].tobytes()).hexdigest()


def aug_fold(test, seed, epochs=10):
    """Lab fold with noise-augmented training (3 noisy copies per training match, same total training steps).
    SEQ_VAR variants (with SEQ_EXTRA=DFL):
      swap   + every Metrica copy with the two teams exchanged (labels H <-> A)
      mirror + every Metrica copy mirrored top <-> bottom
      m50    + one extra copy per Metrica match with heavier noise (50% ball missing)
      pre    pre-train on the DFL matches only, then fine-tune on the Metrica copies"""
    met = [f"G{k}{suf}" for k in (1, 2, 3) if k != test for suf in ("", "r1", "r2")]
    if VAR == "swap":
        met += [n + "x" for n in met]
    if VAR == "mirror":
        met += [n + "y" for n in met]
    if VAR == "m50":
        met += [f"G{k}m50" for k in (1, 2, 3) if k != test]
    ext = []
    if "DFL" in EXTRA.split(","):              # + the 7 Bundesliga matches, provider flags as labels (exp_dfl)
        from possession.dfl import MATCHES
        ext = [f"DFL{m}" for m in MATCHES]
    if "PFF" in EXTRA.split(","):              # + the 40 PFF World Cup 2022 matches (kloppy-logic labels)
        from possession.pff import list_games
        ext += [f"PFF{g}" for g in list_games()]
    names = met + ext
    G = {n: get(n) for n in names}
    cols = columns(get("G1")[1], "v3")
    allX = np.vstack([G[n][0][::STEP][:, cols] for n in names])
    mu = np.nanmean(allX, 0); sd = np.nanstd(allX, 0) + 1e-6
    del allX
    X = {n: prep(G[n][0], cols, mu, sd)[0] for n in names}
    Y = {n: target(G[n][2]) for n in names}
    W = {n: run_weight(G[n][2]) for n in names}
    del G
    if VAR == "pre":
        net = train([X[n] for n in ext], [Y[n] for n in ext], len(cols), 6, seed=seed, ws=[W[n] for n in ext])
        net = train([X[n] for n in met], [Y[n] for n in met], len(cols), 6, seed=seed, ws=[W[n] for n in met],
                    net=net, lr=5e-4)
    else:
        net = train([X[n] for n in names], [Y[n] for n in names], len(cols), epochs, seed=seed,
                    ws=[W[n] for n in names])
    F = get(f"G{test}")[0]
    Xt, _, _ = prep(F, cols, mu, sd)
    wtag = "" if WEIGHT_POWER == 0 else f"_w{WEIGHT_POWER:g}"
    tag = ("_dfl" if EXTRA else "") + ("_pff" if "PFF" in EXTRA else "") + (f"_{VAR}" if VAR else "")
    out = f"data/cache/aug_seq{wtag}{tag}{'' if seed == 0 else f's{seed}'}_P{test}.npy"
    np.save(out, predict(net, Xt, len(F)).astype(np.float32))
    if os.environ.get("SEQ_TTA"):              # test-time augmentation: also predict the team-swapped test match
        Fx = get(f"G{test}x")[0]
        Xx, _, _ = prep(Fx, cols, mu, sd)
        px = predict(net, Xx, len(Fx))[:, [1, 0, 2]]          # swap H/A back
        np.save(out.replace(".npy", "_tta.npy"), px.astype(np.float32))
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
