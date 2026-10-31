"""Labelled possession data from OTHER matches (Metrica public sample games 1-2), allowed by RULES.md.

Labels follow CHALLENGE.md:
  - D from a stoppage (BALL OUT, foul, offside, goal, end of half) until the next SET PIECE starts
  - while alive, possession = team of the last *controlled* action (PASS, SET PIECE, RECOVERY, SHOT);
    CHALLENGE (duels) and deflections do not change it; a change happens at the other team's RECOVERY
Output per game: P (N,S,3) x,y,speed | ball (N,3) | team (S,) 0=home 1=away | label (N,) in 'HAD'
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "data", "external")
FPS = 25
CONTROL = {"PASS", "SET PIECE", "RECOVERY", "SHOT"}


def _tracking(g):
    sides = []
    for side in ("Home", "Away"):
        df = pd.read_csv(os.path.join(EXT, f"G{g}_RawTrackingData_{side}_Team.csv"), skiprows=2)
        cols = [c for c in df.columns if c.startswith("Player")]
        xy = np.stack([df[[c, df.columns[df.columns.get_loc(c) + 1]]].values for c in cols], 1)
        sides.append((xy, df))
    (hxy, hdf), (axy, adf) = sides
    n = min(len(hdf), len(adf))
    xy = np.concatenate([hxy[:n], axy[:n]], 1).astype(np.float32)
    # speed in m/s from positions (pitch 105x68), smoothed over 5 frames
    m = xy * np.array([105, 68], np.float32)
    v = np.full(xy.shape[:2], np.nan, np.float32)
    v[5:] = np.linalg.norm(m[5:] - m[:-5], axis=2) * FPS / 5
    P = np.concatenate([xy, v[..., None]], 2)
    bcol = hdf.columns.get_loc("Ball")
    ball = np.full((n, 3), np.nan, np.float32)
    ball[:, :2] = hdf.iloc[:n, [bcol, bcol + 1]].values
    team = np.r_[np.zeros(hxy.shape[1]), np.ones(axy.shape[1])].astype(np.int8)
    period = hdf["Period"].values[:n]
    return P, ball, team, period


def _labels(events, n, period):
    lab = np.full(n, "X", "<U1")
    side = {"Home": "H", "Away": "A"}
    e = events.sort_values("Start Frame").reset_index(drop=True)
    stop_types = {("BALL OUT",), ("FAULT RECEIVED",)}
    state, dead_from = None, None
    marks = []  # (frame, new_state)
    for _, r in e.iterrows():
        f0, f1, t, sub = int(r["Start Frame"]), int(r["End Frame"]), r.Type, str(r.Subtype)
        if t == "SET PIECE":
            marks.append((f0 - 1, side[r.Team]))
        elif t in CONTROL:
            marks.append((f0 - 1, side[r.Team]))
        if t == "BALL OUT" or t == "FAULT RECEIVED" or "OFFSIDE" in sub or "GOAL" == sub[-4:] or "END HALF" in sub:
            marks.append((max(f1, f0) - 1, "D"))
    marks.sort()
    cur = "D"
    j = 0
    for i in range(n):
        while j < len(marks) and marks[j][0] <= i:
            cur = marks[j][1]
            j += 1
        lab[i] = cur
    lab[np.isnan(period.astype(float))] = "X"
    # ungraded before first kickoff of each period is handled by marks starting at the kick-off set piece
    return lab


def _tracking_g3():
    """Sample game 3: FIFA EPTS (like our match). 22 slots whose team can change between the 11
    data-format blocks (substitutions) -> reorder each block so slots 0-10 = Team A, 11-21 = Team B."""
    import json, xml.etree.ElementTree as ET
    r = ET.parse(os.path.join(EXT, "G3_metadata.xml")).getroot()
    teams = [t.get("id") for t in r.iter("Team")]
    pteam = {p.get("id"): teams.index(p.get("teamId")) for p in r.iter("Player")}
    chan = {c.get("id"): c.get("playerId") for c in r.iter("PlayerChannel")}
    blocks = []
    for spec in r.iter("DataFormatSpecification"):
        refs = [c.get("playerChannelId") for c in spec.iter("PlayerChannelRef")][::2]
        tm = np.array([pteam[chan[x]] for x in refs])
        blocks.append((int(spec.get("startFrame")), int(spec.get("endFrame")), np.argsort(tm, kind="stable")))
    lines = open(os.path.join(EXT, "G3_tracking.txt")).read().splitlines()
    n = len(lines)
    xy = np.full((n, 22, 2), np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    for i, line in enumerate(lines):
        fid, pl, b = line.split(":")
        xy[i] = np.array([g.split(",") for g in pl.split(";")], np.float32)
        ball[i, :2] = np.array(b.split(","), np.float32)
    for a, b_, order in blocks:
        xy[a - 1:b_] = xy[a - 1:b_][:, order]
    m = xy * np.array([105, 68], np.float32)
    v = np.full(xy.shape[:2], np.nan, np.float32)
    v[5:] = np.linalg.norm(m[5:] - m[:-5], axis=2) * FPS / 5
    P = np.concatenate([xy, v[..., None]], 2)
    team = np.r_[np.zeros(11), np.ones(11)].astype(np.int8)
    period = np.where(np.arange(n) < 69661, 1.0, 2.0)
    ev = json.load(open(os.path.join(EXT, "G3_events.json")))["data"]
    side = {"Team A": "Home", "Team B": "Away"}
    rows = []
    for e in ev:
        st = e.get("subtypes")
        sub = st.get("name") if isinstance(st, dict) else ("-".join(x["name"] for x in st) if isinstance(st, list) else "")
        rows.append({"Team": side.get(e["team"]["name"], "Home"), "Type": e["type"]["name"], "Subtype": sub,
                     "Start Frame": e["start"]["frame"], "End Frame": e["end"]["frame"] or e["start"]["frame"]})
    return P, ball, team, period, pd.DataFrame(rows)


def load_game(g):
    cache = os.path.join(EXT, f"G{g}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["P"], d["ball"], d["team"], d["label"], d["period"]
    if g == 3:
        P, ball, team, period, ev = _tracking_g3()
    else:
        P, ball, team, period = _tracking(g)
        ev = pd.read_csv(os.path.join(EXT, f"G{g}_RawEventsData.csv"))
    label = _labels(ev, len(ball), period)
    np.savez(cache, P=P, ball=ball, team=team, label=label, period=period)
    return P, ball, team, label, period


def runs(label):
    out, s = [], 0
    for i in range(1, len(label) + 1):
        if i == len(label) or label[i] != label[s]:
            out.append((label[s], s, i - s))
            s = i
    return out


if __name__ == "__main__":
    for g in (1, 2, 3):
        P, ball, team, label, period = load_game(g)
        r = [x for x in runs(label) if x[0] in "HA"]
        d = np.array([x[2] for x in r]) / FPS
        print(f"game {g}: frames {len(label)}  H {np.mean(label=='H'):.2f} A {np.mean(label=='A'):.2f} "
              f"D {np.mean(label=='D'):.2f} | possessions {len(r)} mean {d.mean():.1f}s median {np.median(d):.1f}s "
              f"<2s {np.mean(d<2):.0%} | ball visible {np.mean(~np.isnan(ball[:,0])):.2f}")
