"""PFF FC World Cup 2022 (open data, other matches) -> the same format as external.load_game.

Labels follow the same logic as kloppy's PFF reader (re-implemented as a memory-light stream): the ball is DEAD after a ball-out or end-of-half game event and ALIVE
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


DEAD_EVENTS = {"OUT", "END"}
ALIVE_EVENTS = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OTB"}


def _open(path):
    import bz2
    return bz2.open(path, "rb") if path.endswith(".bz2") else open(path, "rb")


def load_pff(gid, folder=PFF_DIR, cache=True, max_slots=30):
    """Streaming reader (memory-light): one line at a time, same ball-state / owner logic as kloppy's
    PFF reader. Raw (unsmoothed) positions, players identified by team + shirt number."""
    out = os.path.join(folder, "cache", f"pff_{gid}.npz")
    if cache and os.path.exists(out):
        d = np.load(out)
        return d["P"], d["ball"], d["team"], d["label"], d["period"]
    try:
        import orjson as js
        loads = js.loads
    except ImportError:
        import json
        loads = json.loads
    import json
    meta = json.load(open(_find(folder, "metadata", gid, ".json")))
    meta = meta[0] if isinstance(meta, list) else meta
    pitch = meta["stadium"]["pitches"][0] if meta.get("stadium", {}).get("pitches") else {"length": 105.0, "width": 68.0}
    Lp, Wp = float(pitch["length"]), float(pitch["width"])
    try:
        raw = _find(folder, "tracking", gid, ".jsonl.bz2")
    except FileNotFoundError:
        raw = _find(folder, "tracking", gid, ".jsonl")

    slots = [{}, {}]                                    # shirt number -> slot, per team
    rows = {1: [], 2: []}                               # per period: (time, xy[2*max_slots,2], ball[3], state, owner)
    state, owner = "dead", None
    with _open(raw) as fh:
        for line in fh:
            fr = loads(line)
            ge = fr.get("game_event")
            if ge:
                et = ge.get("game_event_type")
                if et in DEAD_EVENTS:
                    state = "dead"
                elif et in ALIVE_EVENTS:
                    state = "alive"
                if ge.get("home_ball") is not None:
                    owner = "H" if ge["home_ball"] else "A"
            per = fr.get("period")
            if per not in (1, 2):
                continue
            t = fr.get("periodElapsedTime")
            if t is None:
                continue
            xy = np.full((2 * max_slots, 2), np.nan, np.float32)
            for k, key in ((0, "homePlayers"), (1, "awayPlayers")):
                for pl in fr.get(key) or []:
                    j = slots[k].setdefault(str(pl.get("jerseyNum")), len(slots[k]))
                    if j < max_slots and pl.get("x") is not None:
                        xy[k * max_slots + j] = (pl["x"] / Lp + 0.5, pl["y"] / Wp + 0.5)
            b = np.full(3, np.nan, np.float32)
            balls = fr.get("balls") or []
            if balls and balls[0].get("x") is not None:
                b[:] = (balls[0]["x"] / Lp + 0.5, balls[0]["y"] / Wp + 0.5, balls[0].get("z") or np.nan)
            rows[per].append((float(t), xy, b, state, owner))

    nh, na = min(len(slots[0]), max_slots), min(len(slots[1]), max_slots)
    keep = list(range(nh)) + [max_slots + j for j in range(na)]
    team = np.r_[np.zeros(nh), np.ones(na)].astype(np.int8)
    Ps, balls_, labels, periods = [], [], [], []
    for per in (1, 2):
        R = rows[per]
        if not R:
            continue
        t = np.array([r[0] for r in R])
        order = np.argsort(t, kind="stable"); t = t[order]
        grid = np.arange(t[0], t[-1], 1 / FPS)            # ~29.97 fps -> 25 fps, nearest frame
        pos = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
        prev = np.clip(pos - 1, 0, len(t) - 1)
        use = np.where(np.abs(t[prev] - grid) < np.abs(t[pos] - grid), prev, pos)
        hole = np.abs(t[use] - grid) > 0.2
        src = order[use]
        xy = np.stack([R[i][1][keep] for i in src]).astype(np.float32)
        bl = np.stack([R[i][2] for i in src]).astype(np.float32)
        lab = np.array(["D" if R[i][3] == "dead" else (R[i][4] or "X") for i in src], "<U1")
        xy[hole] = np.nan; bl[hole] = np.nan; lab[hole] = "X"
        Ps.append(xy); balls_.append(bl); labels.append(lab); periods.append(np.full(len(src), float(per)))
    xy = np.concatenate(Ps); ball = np.concatenate(balls_); label = np.concatenate(labels)
    period = np.concatenate(periods)
    m = xy * np.array([105, 68], np.float32)
    v = np.full(xy.shape[:2], np.nan, np.float32)
    v[5:] = np.linalg.norm(m[5:] - m[:-5], axis=2) * FPS / 5
    P = np.concatenate([xy, v[..., None]], 2)
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
