"""Per-frame features from team-level tracking. Same function for our match and the lab games.

Everything is computed per team (no player identities), oriented so +x is the direction the HOME
team attacks, so a model trained on other matches transfers to ours.
"""
import numpy as np

FPS = 25


def _nanmean_rows(a):
    with np.errstate(invalid="ignore"):
        cnt = (~np.isnan(a)).sum(1)
        s = np.nansum(a, 1)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan), cnt


def _roll_mean(x, half):
    """Centred nan-aware moving mean over +/- half frames."""
    v = ~np.isnan(x)
    c = np.concatenate(([0], np.cumsum(np.where(v, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(v)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    cnt = n[hi] - n[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _ffill(x):
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]  # leading NaNs stay NaN (index 0 is NaN there)


def _since(mask):
    """Frames since mask was last True (capped)."""
    i = np.arange(len(mask))
    last = np.maximum.accumulate(np.where(mask, i, -10**6))
    return np.minimum(i - last, 500)


def touches(bx, by, X, Y, team, dv_min=5.0, reach=2.5):
    """Touch events: the ball's velocity changes sharply (kick, block, deflection, control) next to a player.

    Returns per-frame team code of the touch: 0 home, 1 away, -1 none, and the velocity change (m/s).
    """
    n = len(bx)
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    vx[2:-2] = (bx[4:] - bx[:-4]) * FPS / 4
    vy[2:-2] = (by[4:] - by[:-4]) * FPS / 4
    dv = np.full(n, np.nan)
    dv[3:-3] = np.hypot(vx[6:] - vx[:-6], vy[6:] - vy[:-6])
    d = np.nan_to_num(dv, nan=0.0)
    peak = (d >= dv_min) & (d >= np.roll(d, 1)) & (d >= np.roll(d, -1))
    for k in (2, 3):
        peak &= (d >= np.roll(d, k)) & (d >= np.roll(d, -k))
    who = np.full(n, -1, np.int8)
    idx = np.flatnonzero(peak)
    if len(idx):
        dist = np.hypot(X[idx] - bx[idx, None], Y[idx] - by[idx, None])
        dist = np.where(np.isnan(dist), np.inf, dist)
        j = dist.argmin(1)
        ok = dist[np.arange(len(idx)), j] <= reach
        who[idx[ok]] = team[j[ok]]
    return who, d


def orientation(P, team, segments):
    """+1 where home attacks towards x=1, -1 otherwise. segments: list of (start, stop) per half."""
    s = np.ones(len(P), np.float32)
    for a, b in segments:
        xh, _ = _nanmean_rows(P[a:b, team == 0, 0])
        xa, _ = _nanmean_rows(P[a:b, team == 1, 0])
        # a team's average position sits in its own half: home on the left -> attacks right
        s[a:b] = 1.0 if np.nanmean(xh) < np.nanmean(xa) else -1.0
    return s


def build(P, ball, team, segments, pitch=(105.0, 68.0)):
    L, Wd = pitch
    n = len(P)
    sgn = orientation(P, team, segments)
    X = (P[:, :, 0] - 0.5) * L * sgn[:, None]
    Y = (P[:, :, 1] - 0.5) * Wd
    S = P[:, :, 2]
    bx = (ball[:, 0] - 0.5) * L * sgn
    by = (ball[:, 1] - 0.5) * Wd
    vis = ~np.isnan(bx)
    f = {}
    f["vis"] = vis.astype(np.float32)
    f["bx"] = bx
    f["by_abs"] = np.abs(by)
    f["out"] = np.maximum.reduce([np.abs(bx) - L / 2, np.abs(by) - Wd / 2, np.zeros(n)])
    # ball speed over 5 frames
    bs = np.full(n, np.nan)
    bs[5:] = np.hypot(bx[5:] - bx[:-5], by[5:] - by[:-5]) * FPS / 5
    f["bspeed"] = np.clip(bs, 0, 40)
    f["bstill"] = _roll_mean((f["bspeed"] < 1.0).astype(float) + np.where(vis, 0, np.nan), 25)
    # distances ball -> nearest player of each team
    d = np.hypot(X - bx[:, None], Y - by[:, None])
    for k, name in ((0, "h"), (1, "a")):
        m = team == k
        dk = np.where(np.isnan(d[:, m]), np.inf, d[:, m]).min(1)
        dk[np.isinf(dk)] = np.nan
        f["d" + name] = np.minimum(dk, 60)
        f["n3" + name] = np.nansum(d[:, m] < 3, 1).astype(np.float32)
        cx, cnt = _nanmean_rows(X[:, m])
        f["cx" + name] = cx
        f["n" + name] = cnt.astype(np.float32)
        with np.errstate(all="ignore"):
            f["sx" + name] = np.nanstd(X[:, m], 1)
            f["sy" + name] = np.nanstd(Y[:, m], 1)
            f["front" + name] = np.nanmax(X[:, m], 1)
            f["back" + name] = np.nanmin(X[:, m], 1)
        f["v" + name], _ = _nanmean_rows(S[:, m])
    # touches: who touched the ball last / next, how long ago, how many recently
    who, dvel = touches(bx, by, X, Y, team)
    f["dvel"] = dvel
    code = np.where(who == 0, 1.0, np.where(who == 1, -1.0, np.nan))   # +1 home, -1 away
    f["tch_last"] = np.nan_to_num(_ffill(code.copy()), nan=0.0)
    f["tch_next"] = np.nan_to_num(_ffill(code[::-1].copy())[::-1], nan=0.0)
    f["tch_since"] = _since(who >= 0).astype(np.float32)
    f["tch_until"] = _since((who >= 0)[::-1])[::-1].astype(np.float32)
    for h in (12, 50):
        for k, name in ((0, "h"), (1, "a")):
            c = np.concatenate(([0], np.cumsum(who == k)))
            i = np.arange(n)
            f[f"tch_{name}_{h}"] = (c[np.minimum(n, i + h + 1)] - c[np.maximum(0, i - h)]).astype(np.float32)
        f[f"tch_bal_{h}"] = f[f"tch_h_{h}"] - f[f"tch_a_{h}"]
    f["ddiff"] = f["dh"] - f["da"]
    f["cdiff"] = f["cxh"] - f["cxa"]
    # team-shape movement: a turnover flips which way both blocks run (m/s, +x = home attack direction)
    for name in ("cxh", "cxa", "fronth", "backa"):
        v = np.full(n, np.nan)
        v[12:-12] = (f[name][24:] - f[name][:-24]) / (24 / FPS)
        f["v_" + name] = v
    f["v_shape"] = f["v_cxh"] + f["v_cxa"]
    # --- team-reaction signals (from the team's football answers) ---------------------------------
    # player velocities over 0.5 s, in the oriented frame (+x = home attack direction)
    k = 6
    VX = np.full_like(X, np.nan); VY = np.full_like(Y, np.nan)
    VX[k:-k] = (X[2 * k:] - X[:-2 * k]) * FPS / (2 * k)
    VY[k:-k] = (Y[2 * k:] - Y[:-2 * k]) * FPS / (2 * k)
    # reference point for "near the ball": the ball, or where it was last / will next be seen
    rx = np.where(vis, bx, np.nan); ry = np.where(vis, by, np.nan)
    rx_f, ry_f = _ffill(rx.copy()), _ffill(ry.copy())
    rx_b, ry_b = _ffill(rx[::-1].copy())[::-1], _ffill(ry[::-1].copy())[::-1]
    refx = np.where(np.isnan(rx_f), rx_b, rx_f); refy = np.where(np.isnan(ry_f), ry_b, ry_f)
    dref = np.hypot(X - refx[:, None], Y - refy[:, None])
    for kk, name in ((0, "h"), (1, "a")):
        m = team == kk
        d = np.where(np.isnan(dref[:, m]), np.inf, dref[:, m])
        order = np.argsort(d, 1)[:, :5]
        vxn = np.take_along_axis(VX[:, m], order, 1)
        with np.errstate(all="ignore"):
            near = np.nanmean(vxn, 1)
        f["rv_near_" + name] = near
        # reversal: next second minus previous second (future - past), near players and whole team
        for key, src in (("rv_dnear_", near), ("rv_dteam_", f["v_cx" + name])):
            dd = np.full(n, np.nan)
            dd[25:-25] = _roll_mean(src, 12)[50:] - _roll_mean(src, 12)[:-50]
            f[key + name] = dd
        # convergence: players running (>2 m/s) towards the ball point (cos > 0.8, within 20 m)
        ux = (refx[:, None] - X[:, m]); uy = (refy[:, None] - Y[:, m])
        dist = np.hypot(ux, uy)
        sp = np.hypot(VX[:, m], VY[:, m])
        with np.errstate(all="ignore"):
            cos = (VX[:, m] * ux + VY[:, m] * uy) / (sp * dist)
        conv = np.nansum((sp > 2) & (cos > 0.8) & (dist < 20), 1).astype(float)
        c = np.concatenate(([0], np.cumsum(conv)))
        i = np.arange(n)
        f["conv_next_" + name] = (c[np.minimum(n, i + 51)] - c[i]) / 50       # avg over next 2 s
        f["conv_prev_" + name] = (c[i] - c[np.maximum(0, i - 50)]) / 50
        # defensive line: home defends -x (its 3 lowest x), away defends +x (its 3 highest x)
        xs = np.where(np.isnan(X[:, m]), np.inf if kk == 0 else -np.inf, X[:, m])
        xs = np.sort(xs, 1)
        line = xs[:, :3] if kk == 0 else xs[:, -3:]
        line = np.where(np.isinf(line), np.nan, line)
        with np.errstate(all="ignore"):
            dl = np.nanmean(line, 1)
        v = np.full(n, np.nan); v[12:-12] = (dl[24:] - dl[:-24]) / (24 / FPS)
        f["dl_v_" + name] = v
        # sideways shift of the team
        with np.errstate(all="ignore"):
            cy = np.nanmean(Y[:, m], 1)
        v = np.full(n, np.nan); v[12:-12] = (cy[24:] - cy[:-24]) / (24 / FPS)
        f["vy_" + name] = v
    f["rv_split"] = f["rv_dnear_h"] - f["rv_dnear_a"]
    f["conv_both"] = np.minimum(f["conv_next_h"], f["conv_next_a"])
    f["vall"] = (f["vh"] + f["va"]) / 2
    f["since_seen"] = _since(vis).astype(np.float32)
    f["until_seen"] = _since(vis[::-1])[::-1].astype(np.float32)
    # where the ball was last seen / is next seen: tells a stoppage from play the tracking lost,
    # without needing the ball to be visible now
    line = np.minimum(L / 2 - np.abs(bx), Wd / 2 - np.abs(by))       # m to nearest boundary line (<0 = out)
    goal_line = L / 2 - np.abs(bx)
    for name, x in (("line", line), ("gline", goal_line), ("bx", bx), ("by_abs", np.abs(by))):
        f[name + "_last"] = _ffill(x.copy())
        f[name + "_next"] = _ffill(x[::-1].copy())[::-1]
    f["ddiff_last"] = _ffill(f["ddiff"].copy())
    f["ddiff_next"] = _ffill(f["ddiff"][::-1].copy())[::-1]
    for key in ("ddiff", "bspeed", "vall", "cdiff", "bx", "vis", "out", "vh", "va", "sxh", "sxa"):
        for h in (12, 50, 125):
            f[f"{key}_m{h}"] = _roll_mean(f[key].astype(float), h)
    names = sorted(f)
    return np.column_stack([np.asarray(f[k], np.float32) for k in names]), names
