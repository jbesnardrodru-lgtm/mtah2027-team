"""PFF FC World Cup 2022 (open data, other matches) -> the same format as external.load_game.

Labels follow kloppy's PFF reader: the ball is DEAD after a ball-out or end-of-half game event and ALIVE
after a kick-off or any on-the-ball event; the owning team switches at each on-the-ball event. That mirrors
how the official answer key is built (dead/alive + owning team from tracking), touch-based short possessions
included.

Folder layout expected (set PFF_DIR or pass folder=); file names are matched by game id:
    data/pff/
      metadata/  <id>.json   (or pff_metadata_<id>.json)
      rosters/   <id>.json   (or pff_rosters_<id>.json)
      tracking/  <id>.jsonl.bz2  (or .jsonl)
Usage:
    from possession.pff import load_pff, list_games
    P, ball, team, label, period = load_pff(10517)
    python -m possession.pff            # summary of every game found (sanity check)
"""
import glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFF_DIR = os.environ.get("PFF_DIR") or os.path.join(ROOT, "data", "pff")
FPS = 25


def _find(folder, sub, gid, exts):
    """Find the <sub> file ('metadata' / 'rosters' / 'tracking') of game gid anywhere under folder.
    Matches on the folder or file name containing the kind, case-insensitive (PFF's own layout uses e.g.
    'Metadata/<id>.json', 'Rosters/<id>.json', 'Tracking Data/<id>.jsonl.bz2')."""
    key = {"metadata": "metadata", "rosters": "roster", "tracking": "tracking"}[sub]
    hits = sorted(glob.glob(os.path.join(folder, "**", f"*{gid}*{exts}"), recursive=True))
    hits = [h for h in hits if os.sep + "cache" + os.sep not in h]
    if sub == "tracking":
        pick = [h for h in hits if "metadata" not in h.lower() and "roster" not in h.lower()]
    else:
        pick = [h for h in hits if key in os.path.relpath(h, folder).lower()]
    if not pick:
        raise FileNotFoundError(f"no {sub} file ({exts}) for game {gid} under {folder}")
    return pick[0]


def list_games(folder=PFF_DIR):
    ids = set()
    for f in glob.glob(os.path.join(folder, "**", "*.jsonl*"), recursive=True):
        digits = "".join(ch if ch.isdigit() else " " for ch in os.path.basename(f)).split()
        if digits:
            ids.add(int(digits[-1]))
    return sorted(ids)


def load_pff(gid, folder=PFF_DIR, cache=True):
    out = os.path.join(folder, "cache", f"pff_{gid}.npz")
    if cache and os.path.exists(out):
        d = np.load(out)
        return d["P"], d["ball"], d["team"], d["label"], d["period"]
    from kloppy import pff
    meta = _find(folder, "metadata", gid, ".json")
    rost = _find(folder, "rosters", gid, ".json")
    try:
        raw = _find(folder, "tracking", gid, ".jsonl.bz2")
    except FileNotFoundError:
        raw = _find(folder, "tracking", gid, ".jsonl")
    ds = pff.load_tracking(meta, rost, raw, coordinates="kloppy")
    df = ds.to_df()
    home, away = ds.metadata.teams[0], ds.metadata.teams[1]
    hid = {str(p.player_id) for p in home.players}
    aid = {str(p.player_id) for p in away.players}
    pids = [c[:-2] for c in df.columns if c.endswith("_x") and c != "ball_x"]
    pids = [p for p in pids if p in hid] + [p for p in pids if p in aid]
    team = np.array([0 if p in hid else 1 for p in pids], np.int8)

    t = df["timestamp"].dt.total_seconds().values
    per = df["period_id"].values
    rows, period, holes = [], [], []
    for k in (1, 2):                                   # regulation halves only
        idx = np.flatnonzero(per == k)
        if not len(idx):
            continue
        grid = np.arange(t[idx[0]], t[idx[-1]], 1 / FPS)  # resample ~29.97 fps -> 25 fps (nearest frame)
        pos = np.clip(np.searchsorted(t[idx], grid), 0, len(idx) - 1)
        prev = np.clip(pos - 1, 0, len(idx) - 1)
        use = np.where(np.abs(t[idx][prev] - grid) < np.abs(t[idx][pos] - grid), prev, pos)
        j = idx[use]
        rows.append(j); period.append(np.full(len(j), float(k)))
        holes.append(np.abs(t[j] - grid) > 0.2)        # no PFF frame within 0.2 s: a hole, not a copy
    j = np.concatenate(rows); period = np.concatenate(period); hole = np.concatenate(holes)

    xy = np.stack([df[[p + "_x", p + "_y"]].values[j] for p in pids], 1).astype(np.float32)
    m = xy * np.array([105, 68], np.float32)
    v = np.full(xy.shape[:2], np.nan, np.float32)
    v[5:] = np.linalg.norm(m[5:] - m[:-5], axis=2) * FPS / 5
    xy[hole] = np.nan
    P = np.concatenate([xy, v[..., None]], 2)
    P[hole] = np.nan
    ball = np.full((len(j), 3), np.nan, np.float32)
    ball[:, :2] = df[["ball_x", "ball_y"]].values[j]
    ball[hole] = np.nan

    state = df["ball_state"].astype(str).values[j]
    owner = df["ball_owning_team_id"].astype(str).values[j]
    label = np.where(state == "dead", "D", np.where(owner == str(home.team_id), "H",
                     np.where(owner == str(away.team_id), "A", "X"))).astype("<U1")
    label[hole] = "X"
    if cache:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, P=P, ball=ball, team=team, label=label, period=period)
    return P, ball, team, label, period


if __name__ == "__main__":
    from possession.external import runs
    folder = sys.argv[1] if len(sys.argv) > 1 else PFF_DIR
    for gid in list_games(folder):
        P, ball, team, label, period = load_pff(gid, folder)
        r = [x for x in runs(label) if x[0] in "HA"]
        d = np.array([x[2] for x in r]) / FPS
        print(f"PFF {gid}: frames {len(label)} ({len(label)/FPS/60:.0f} min)  players {P.shape[1]} "
              f"(home {np.sum(team==0)})  H {np.mean(label=='H'):.2f} A {np.mean(label=='A'):.2f} "
              f"D {np.mean(label=='D'):.2f} | possessions {len(r)} mean {d.mean() if len(d) else 0:.1f}s "
              f"<2s {np.mean(d<2) if len(d) else 0:.0%} | ball visible {np.mean(~np.isnan(ball[:,0])):.2f}")
