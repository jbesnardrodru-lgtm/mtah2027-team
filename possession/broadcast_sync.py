"""Feasibility: align each valid stretch of the broadcast feed to the tactical timeline by ball position.
For every stretch (homography valid, >= 10 s) find the tactical offset where the two ball tracks agree best."""
import numpy as np, pandas as pd

L, W = 105.3, 68.0


def stretches(valid, min_len=250):
    e = np.diff(np.r_[0, valid.astype(int), 0])
    s, t = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    return [(a, b) for a, b in zip(s, t) if b - a >= min_len]


def best_offset(bb, bt, a, b, step=4):
    idx = np.arange(a, b, step)
    idx = idx[~np.isnan(bb[idx, 0])]
    if len(idx) < 20:
        return None
    n = len(bt)
    offs = np.arange(-idx[0], n - idx[-1] - 1)
    # coarse: every 5th offset, vectorised in chunks
    best = (np.inf, None, np.inf)
    costs = np.full(len(offs), np.nan, np.float32)
    for c0 in range(0, len(offs), 4000):
        o = offs[c0:c0 + 4000]
        T = idx[None, :] + o[:, None]
        dx = (bt[T, 0] - bb[idx, 0][None, :]) * L
        dy = (bt[T, 1] - bb[idx, 1][None, :]) * W
        d = np.hypot(dx, dy)
        ok = ~np.isnan(d)
        cnt = ok.sum(1)
        med = np.where(cnt >= 0.4 * len(idx), np.nanmedian(np.where(ok, d, np.nan), 1), np.nan)
        costs[c0:c0 + 4000] = med
    j = np.nanargmin(costs)
    # runner-up at least 5 s away from the best
    far = np.abs(offs - offs[j]) > 125
    j2 = np.nanargmin(np.where(far, costs, np.nan))
    return offs[j], costs[j], costs[j2], len(idx)


def build_mapping(out="data/cache/broadcast_mapped.npz", n_probe=25, verbose=True):
    """Find one offset per half (median over probed stretches) and map the broadcast ball onto the
    tactical timeline (only where the broadcast homography is valid)."""
    t = np.load("data/cache/tactical.npz")["ball"]
    b = np.load("data/cache/broadcast.npz")["ball"]
    valid = pd.read_csv("data/broadcast/BAR-ATH_broadcast_homography.csv", usecols=["valid"])["valid"].values.astype(bool)
    st = stretches(valid)
    found = []
    for a, e in st[::max(1, len(st) // n_probe)][:n_probe]:
        r = best_offset(b, t, a, e)
        if r is not None and r[2] / max(r[1], 0.1) >= 2.0:     # keep only clear matches
            found.append(r[0])
    found = np.array(found)
    # two halves = two clusters of offsets
    split = (found.min() + found.max()) / 2
    off1, off2 = int(np.median(found[found < split])), int(np.median(found[found >= split]))
    n = len(t)
    mapped = np.full((n, 3), np.nan, np.float32)
    bi = np.arange(len(b))
    first_half_end = 100000   # any frame inside the published halftime gap (90652-110807) separates halves
    for off, lo, hi in ((off1, 0, first_half_end), (off2, first_half_end, n)):
        ti = bi + off
        ok = (ti >= lo) & (ti < hi) & (ti < n) & valid & ~np.isnan(b[:, 0])
        mapped[ti[ok], :2] = b[bi[ok], :2]
    np.savez(out, ball=mapped, offsets=np.array([off1, off2]))
    if verbose:
        print(f"offsets: first half {off1:+d}, second half {off2:+d} frames ({len(found)} clear stretches)")
    return mapped


if __name__ == "__main__":
    build_mapping()
