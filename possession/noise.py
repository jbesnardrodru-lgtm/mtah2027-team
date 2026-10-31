"""Make clean lab tracking look like our automatic tracking (statistics of OUR data, never its labels).

Measured on BAR-ATH tactical ATD (graded frames):
  ball visible 59%; gaps are long blocks: median 268 frames, p90 845, up to several minutes
  ball visible on about two thirds of dead-ball time (it is placed for the restart)
  20.5 of 22 players tracked on average
"""
import numpy as np


# Where OUR tracking loses the ball (measured on BAR-ATH tracking only, no labels):
#   by player-speed quintile (mean speed of tracked players, 1 s smoothed), slow -> fast
SPEED_CUTS = (0.93, 1.31, 1.80, 2.54)          # m/s
MISS_BY_SPEED = (0.83, 0.60, 0.36, 0.14, 0.10)
#   by position within the half, 9 equal blocks (average of both halves)
MISS_BY_TIME = (0.19, 0.18, 0.37, 0.27, 0.43, 0.69, 0.52, 0.52, 0.50)


def gap_weights(P, period):
    """Relative chance that a tracking gap starts at each frame, mimicking our match."""
    with np.errstate(all="ignore"):
        v = np.nanmean(P[:, :, 2], 1)
    v = np.convolve(np.nan_to_num(v), np.ones(25) / 25, "same")
    w = np.array(MISS_BY_SPEED)[np.digitize(v, SPEED_CUTS)]
    per = np.nan_to_num(period.astype(float), nan=0)
    tw = np.zeros(len(P))
    for k in (1, 2):
        idx = np.flatnonzero(per == k)
        if len(idx):
            pos = (np.arange(len(idx)) * 9 // len(idx)).clip(0, 8)
            tw[idx] = np.array(MISS_BY_TIME)[pos]
    return w * tw


def ball_gaps(ball, label, rng, target_missing=0.41, med=268, p90=845, weights=None):
    """Remove the ball in long blocks during play; put a static ball back in most dead-ball stretches."""
    b = ball.copy()
    n = len(b)
    graded = label != "X"
    # 1) dead ball: the lab has no ball there; ours mostly does (held still near the restart spot)
    vis = ~np.isnan(b[:, 0])
    last = np.maximum.accumulate(np.where(vis, np.arange(n), 0))
    dead = (label == "D") & ~vis
    runs = np.flatnonzero(np.diff(np.r_[0, dead.astype(int), 0]))
    for s, e in zip(runs[::2], runs[1::2]):
        if rng.random() < 0.66:
            # held for the restart: on the pitch (on the line), not where it crossed it
            b[s:e, :2] = np.clip(b[last[s], :2], 0.0, 1.0) + rng.normal(0, 0.002, (e - s, 2))
    # 2) remove the ball in long blocks until the graded missing share reaches the target
    sigma = np.log(p90 / med) / 1.2816
    while np.isnan(b[graded, 0]).mean() < target_missing:
        length = int(np.clip(rng.lognormal(np.log(med), sigma), 25, 6000))
        if weights is None:
            s = int(rng.integers(0, n - length))
        else:
            s = int(min(rng.choice(n, p=weights / weights.sum()), n - length))
        b[s:s + length] = np.nan
    return b


def _nan_smooth(a, k):
    w = np.ones(k) / k
    out = np.convolve(np.nan_to_num(a), w, "same")
    cnt = np.convolve((~np.isnan(a)).astype(float), w, "same")
    r = out / np.where(cnt > 0, cnt, np.nan)
    r[np.isnan(a)] = np.nan
    return r


def ball_offset(ball, rng, sd_m=0.6, pitch=(105.0, 68.0), smooth_k=19, drift_win=50, white_m=0.05):
    """Make the lab ball move like OUR ball (calibrated on our tracking, no labels):
      - the lab ball runs in sharp straight segments (velocity change p50/p90/p99 0.2/7.3/23 m/s);
        ours is smooth (1.3/4.3/8.7). A 19-frame smoothing + small jitter reproduces 1.3/4.8/8.7.
      - ours sits ~0.6 m from the nearest player even at the feet: a slowly drifting offset of sd_m.
    """
    n = len(ball)
    b = ball.copy()
    for j, L in ((0, pitch[0]), (1, pitch[1])):
        x = _nan_smooth(ball[:, j] * L, smooth_k)
        z = rng.normal(0, 1, n)
        k = np.ones(drift_win) / drift_win
        z = np.convolve(np.convolve(z, k, "same"), k, "same")
        x = x + z / z.std() * sd_m + rng.normal(0, white_m, n)
        b[:, j] = x / L
    return b


def player_dropout(P, rng, keep=20.5 / 22, mean_gap=250):
    """Drop each player in random blocks so that about `keep` of player-frames remain."""
    Q = P.copy()
    n, s, _ = Q.shape
    drop_share = 1 - keep
    for j in range(s):
        missing = 0
        while missing < drop_share * n:
            length = int(rng.exponential(mean_gap)) + 1
            start = int(rng.integers(0, n - length))
            Q[start:start + length, j] = np.nan
            missing += length
    return Q


def degrade(P, ball, label, seed=0, period=None, placement="weighted", target_missing=0.41):
    """period given -> realistic ball offset; placement 'weighted' = gaps where our tracking loses the ball
    (TEST realism), 'uniform' = gaps at random moments (TRAINING: a missing ball says nothing about play)."""
    rng = np.random.default_rng(seed)
    w = gap_weights(P, period) if (period is not None and placement == "weighted") else None
    b = ball_offset(ball, rng) if period is not None else ball
    return player_dropout(P, rng), ball_gaps(b, label, rng, target_missing=target_missing, weights=w)
