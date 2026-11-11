"""PathCRF features (Kim et al., KDD 2026; github.com/hyunsungkim-ds/pathcrf, MPL-2.0; pretrained on the DFL/IDSSE
open data = external). From player positions only, PathCRF infers who holds the ball or between whom it travels.
We run its released model (trial 140, dynamic masked CRF) and keep, per frame:
  pc_h_prev / pc_a_prev / pc_o_prev : probability that the last possessor is home / away / outside (ball out)
  pc_h_next / pc_a_next / pc_o_next : same for the next possessor
  pc_src_h, pc_src_a, pc_fly       : decoded path: team of the sender/holder, ball in flight between two players
No ball, no labels and no video are used.  Needs the PathCRF repo at $PATHCRF (default ~/pathcrf, commit 1b7c6f8).
    python -m possession.pathcrf_feat G1          (writes data/cache/pathcrf_G1.npz)
"""
import os, sys
import numpy as np, pandas as pd

PATHCRF = os.path.expanduser(os.environ.get("PATHCRF", "~/pathcrf"))
L, W, FPS = 105.0, 68.0, 25
COLS = ["pc_h_prev", "pc_a_prev", "pc_o_prev", "pc_h_next", "pc_a_next", "pc_o_next", "pc_src_h", "pc_src_a", "pc_fly"]


def assign_nodes(xy, team, k, n=11):
    """Slots of team k -> n stable nodes (a node keeps its slot while the slot lives; new slots take the free node
    nearest to where it was last seen). Returns (T, n, 2) metres with NaN where a node is empty."""
    slots = np.flatnonzero(team == k)
    T = len(xy)
    out = np.full((T, n, 2), np.nan, np.float32)
    node_of = {}                     # slot -> node
    last = np.full((n, 2), np.nan)
    for t in range(T):
        live = [s for s in slots if not np.isnan(xy[t, s, 0])]
        for s in list(node_of):
            if s not in live:
                del node_of[s]
        used = set(node_of.values())
        for s in live:
            if s in node_of:
                continue
            free = [j for j in range(n) if j not in used]
            if not free:
                break
            p = xy[t, s]
            d = [np.hypot(*(p - last[j])) if not np.isnan(last[j, 0]) else 1e9 for j in free]
            j = free[int(np.argmin(d))]
            node_of[s] = j; used.add(j)
        for s, j in node_of.items():
            out[t, j] = xy[t, s]; last[j] = xy[t, s]
    return out


def fill(a):
    """Interpolate gaps per node and coordinate; nodes never seen -> team centroid."""
    T, n, _ = a.shape
    df = pd.DataFrame(a.reshape(T, -1)).interpolate(limit_direction="both")
    a = df.to_numpy(np.float32).reshape(T, n, 2)
    cen = np.nanmean(a, axis=1, keepdims=True)
    cen = pd.DataFrame(cen.reshape(T, 2)).interpolate(limit_direction="both").fillna(0).to_numpy(np.float32)[:, None, :]
    return np.where(np.isnan(a), np.broadcast_to(cen, a.shape), a)


def kinematics(a):
    v = np.gradient(pd.DataFrame(a.reshape(len(a), -1)).rolling(5, center=True, min_periods=1).mean()
                    .to_numpy().reshape(a.shape), axis=0) * FPS
    sp = np.hypot(v[..., 0], v[..., 1])
    acc = np.gradient(sp, axis=0) * FPS
    return v, sp, acc


def segment_frame(P, team, a, b):
    xy = P[a:b, :, :2].astype(np.float64) * np.array([L, W])
    cols = {}
    for k, name in ((0, "home"), (1, "away")):
        nodes = fill(assign_nodes(xy, team, k))
        v, sp, acc = kinematics(nodes)
        for j in range(nodes.shape[1]):
            p = f"{name}_{j + 1}"
            cols[f"{p}_x"] = nodes[:, j, 0]; cols[f"{p}_y"] = nodes[:, j, 1]
            cols[f"{p}_vx"] = v[:, j, 0]; cols[f"{p}_vy"] = v[:, j, 1]
            cols[f"{p}_speed"] = sp[:, j]; cols[f"{p}_accel"] = acc[:, j]
    df = pd.DataFrame(cols, index=pd.RangeIndex(a, b, name="frame_id"))
    return df


def load_model():
    sys.path.insert(0, PATHCRF)
    import torch
    from models.utils import build_model, load_trial_args, resolve_model_path
    save = os.path.join(PATHCRF, "saved", "140")
    args = load_trial_args(save)
    model = build_model(args, device="cpu")
    model.load_state_dict(torch.load(resolve_model_path(save, "state_dict_best_acc.pt"), map_location="cpu", weights_only=False))
    model.eval()
    return model, args


def run(P, team, segments, model=None, chunk=25 * 120):
    import torch
    sys.path.insert(0, PATHCRF)
    from inference import inference_episode
    from datatools import utils
    if model is None:
        model, _ = load_model()
    torch.set_num_threads(4)
    out = np.full((len(P), len(COLS)), np.nan, np.float32)
    for a, b in segments:
        seg = segment_frame(P, team, a, b)
        lgk, rgk = utils.detect_keepers(seg, left_first=True)
        for c0 in range(a, b, chunk):
            c1 = min(b, c0 + chunk + 25 * 10)              # 10 s overlap, the later chunk wins on the overlap
            ep = seg.loc[c0:c1 - 1]
            if len(ep) < 50:
                continue
            with torch.no_grad():
                mp, mn, micro, order, _ = inference_episode(model, ep, left_gk=lgk, right_gk=rgk, use_crf=True,
                                                            evaluate=False, window_seconds=10.0, device="cpu")
            if mp is None:
                continue
            idx = ep.index.to_numpy() 
            def team_sum(m, pre):
                return m[[c for c in m.columns if c.startswith(pre)]].sum(axis=1).to_numpy()
            src = micro["edge_src"].astype(str); dst = micro["edge_dst"].astype(str)
            vals = np.stack([team_sum(mp, "home"), team_sum(mp, "away"), team_sum(mp, "out"),
                             team_sum(mn, "home"), team_sum(mn, "away"), team_sum(mn, "out"),
                             src.str.startswith("home").to_numpy(float), src.str.startswith("away").to_numpy(float),
                             ((src != dst) & ~src.str.startswith("out") & ~dst.str.startswith("out")).to_numpy(float)], 1)
            out[idx] = vals
    return out


def lab_inputs(name):
    """Degraded tracking of a lab copy, exactly as featcache builds it."""
    from possession.external import load_game
    from possession.noise import degrade
    from possession.model import segments_from_period
    swap = name.endswith("x"); base = name.rstrip("xy")
    if base.startswith("DFL"):
        from possession.dfl import load_dfl, MATCHES
        P, ball, team, label, period = load_dfl(base[3:]); seed = 500 + MATCHES.index(base[3:])
    else:
        g = int(base[1:2]); rep = int(base.split("r")[1]) if "r" in base else 0
        P, ball, team, label, period = load_game(g); seed = 1000 * rep + (7 if swap else 0) + g
    if swap:
        team = (1 - team).astype(team.dtype)
    P, ball = degrade(P, ball, label, seed=seed, period=period)
    return P, team, segments_from_period(period)


if __name__ == "__main__":
    import time
    model, _ = load_model()
    for name in sys.argv[1:]:
        path = f"data/cache/pathcrf_{name}.npz"
        if os.path.exists(path):
            continue
        t0 = time.time()
        if name == "BARATH":
            from possession.cache import load
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from games import GAME, half_video_blocks
            P, ball, team = load("tactical"); segs = [(s, e + 1) for _, s, e in half_video_blocks(GAME)]
        else:
            P, team, segs = lab_inputs(name)
        f = run(P, team, segs, model)
        np.savez(path, F=f, cols=np.array(COLS))
        print(name, f"{time.time() - t0:.0f}s", "coverage %.0f%%" % (100 * np.mean(~np.isnan(f[:, 0]))), flush=True)
