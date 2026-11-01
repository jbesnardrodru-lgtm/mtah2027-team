"""'The ball stays with its carrier' (idea: Jorge, 1 Nov).

When the ball disappears at a player's feet while moving slowly (carried, not kicked), it is assumed to stay
at that player's feet; when it reappears at a player's feet, that player is assumed to have had it just before.
The missing ball is filled with the carrier's position, forward from the disappearance and backward from the
reappearance, for at most `max_s` seconds each way and only while that player's track is continuous.
"""
import numpy as np

FPS = 25


def impute_carrier(P, ball, team, pitch=(105.0, 68.0), max_s=2.0, reach=1.5, max_speed=8.0):
    L, W = pitch
    n = len(ball)
    out = ball.copy()
    filled = np.zeros(n, bool)
    vis = ~np.isnan(ball[:, 0])
    X = P[:, :, 0] * L; Y = P[:, :, 1] * W
    bx = ball[:, 0] * L; by = ball[:, 1] * W
    e = np.diff(np.r_[0, (~vis).astype(int), 0]); starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    k = int(max_s * FPS)

    def carrier(t):
        d = np.hypot(X[t] - bx[t], Y[t] - by[t])
        if np.all(np.isnan(d)):
            return None
        j = int(np.nanargmin(d))
        return j if d[j] <= reach else None

    def fill(j, frames, ref_t):
        dx = bx[ref_t] - X[ref_t, j]; dy = by[ref_t] - Y[ref_t, j]       # keep the ball's offset to the feet
        for t in frames:
            if np.isnan(X[t, j]):
                break                                                     # track lost: stop trusting it
            out[t, 0] = (X[t, j] + dx) / L; out[t, 1] = (Y[t, j] + dy) / W
            filled[t] = True

    for s, t_end in zip(starts, ends):
        if s >= 6 and vis[s - 1] and vis[s - 6]:
            sp = np.hypot(bx[s - 1] - bx[s - 6], by[s - 1] - by[s - 6]) * FPS / 5
            j = carrier(s - 1)
            if j is not None and sp <= max_speed:
                fill(j, range(s, min(t_end, s + k)), s - 1)
        if t_end < n and t_end + 5 < n and vis[t_end] and vis[min(n - 1, t_end + 5)]:
            sp = np.hypot(bx[t_end + 5] - bx[t_end], by[t_end + 5] - by[t_end]) * FPS / 5
            j = carrier(t_end)
            if j is not None and sp <= max_speed:
                fill(j, range(t_end - 1, max(s, t_end - k) - 1, -1), t_end)
    return out, filled
