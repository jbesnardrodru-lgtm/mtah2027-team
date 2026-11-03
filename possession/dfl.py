"""IDSSE / DFL open data (7 Bundesliga matches 2022/23, CC-BY 4.0; Bassek et al., Sci Data 12, 195, 2025).
Download with `sh possession/get_dfl.sh`. Streams the positions XML (one Frame per line) without kloppy.
Output like possession.external.load_game: P (N,S,3) x,y in 0-1 + speed | ball (N,3) | team (S,) 0 home 1 away |
label (N,) | period (N,).  Labels (mode "flags"): the provider's per-frame flags -- BallStatus 0 -> D,
otherwise BallPossession 1 -> H, 2 -> A.
    python -m possession.dfl            # parse all 7 and print label statistics
"""
import os, re, glob, sys
import numpy as np
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "dfl")
FPS = 25
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
FS = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
FR = re.compile(r'<Frame N="(\d+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)"')
BF = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def _file(kind, mid):
    f = glob.glob(os.path.join(DIR, f"DFL_{kind}_*DFL-MAT-{mid}.xml"))
    if not f:
        raise SystemExit(f"missing DFL file for {mid}: run sh possession/get_dfl.sh")
    return f[0]


def _meta(mid):
    r = ET.parse(_file("02_01_matchinformation", mid)).getroot()
    env = next(r.iter("Environment"))
    L, W = float(env.get("PitchX")), float(env.get("PitchY"))
    home = next(t.get("TeamId") for t in r.iter("Team") if t.get("Role") == "home")
    return L, W, home


def load_dfl(mid):
    cache = os.path.join(DIR, f"{mid}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["P"], d["ball"], d["team"], d["label"], d["period"]
    L, W, home = _meta(mid)
    sec = {"firstHalf": 1, "secondHalf": 2}
    obj = {}                      # (section, team, person) -> list of (N, x, y)
    ballinfo = {}                 # section -> list of (N, x, y, poss, status)
    cur = None
    with open(_file("04_03_positions_raw_observed", mid)) as fh:
        for line in fh:
            if "<FrameSet" in line:
                m = FS.search(line); s = sec.get(m.group(1))
                cur = None if s is None or m.group(2) == "referee" else (s, m.group(2), m.group(3))
                if cur and cur[1] == "BALL":
                    ballinfo.setdefault(s, [])
                elif cur:
                    obj.setdefault(cur, [])
                continue
            if cur is None or "<Frame " not in line:
                continue
            m = FR.search(line)
            n, x, y = int(m.group(1)), float(m.group(2)), float(m.group(3))
            if cur[1] == "BALL":
                b = BF.search(line)
                ballinfo[cur[0]].append((n, x, y, int(b.group(1)), int(b.group(2))))
            else:
                obj[cur].append((n, x, y))
    # frame index: each half's ball frame range, concatenated
    offs, total = {}, 0
    for s in (1, 2):
        ns = np.array([r[0] for r in ballinfo[s]])
        offs[s] = (ns.min(), total); total += ns.max() - ns.min() + 1
    persons = sorted({(t, p) for (_, t, p) in obj})
    team = np.array([0 if t == home else 1 for t, _ in persons], np.int8)
    order = np.argsort(team, kind="stable"); persons = [persons[i] for i in order]; team = team[order]
    xy = np.full((total, len(persons), 2), np.nan, np.float32)
    for k, (t, p) in enumerate(persons):
        for s in (1, 2):
            rows = obj.get((s, t, p))
            if rows:
                a = np.array(rows, np.float64); i = a[:, 0].astype(int) - offs[s][0] + offs[s][1]
                ok = (i >= 0) & (i < total)
                xy[i[ok], k, 0] = (a[ok, 1] + L / 2) / L
                xy[i[ok], k, 1] = (W / 2 - a[ok, 2]) / W
    ball = np.full((total, 3), np.nan, np.float32)
    label = np.full(total, "X", "<U1"); period = np.full(total, np.nan)
    for s in (1, 2):
        a = np.array(ballinfo[s], np.float64); i = a[:, 0].astype(int) - offs[s][0] + offs[s][1]
        ball[i, 0] = (a[:, 1] + L / 2) / L; ball[i, 1] = (W / 2 - a[:, 2]) / W
        label[i] = np.where(a[:, 4] == 0, "D", np.where(a[:, 3] == 1, "H", "A"))
        period[i] = s
    m = xy * np.array([105, 68], np.float32)
    v = np.full(xy.shape[:2], np.nan, np.float32)
    v[5:] = np.linalg.norm(m[5:] - m[:-5], axis=2) * FPS / 5
    P = np.concatenate([xy, v[..., None]], 2)
    np.savez(cache, P=P, ball=ball, team=team, label=label, period=period)
    return P, ball, team, label, period


def stats(label):
    g = label != "X"; l = label[g]
    r = l[np.r_[True, l[1:] != l[:-1]]]
    runs = np.diff(np.r_[np.flatnonzero(np.r_[True, l[1:] != l[:-1]]), len(l)])
    live = r != "D"
    return (f"D {np.mean(l=='D'):.0%}  possessions {live.sum():>4} ({live.sum()/(len(l)/FPS/60)*90:.0f} per 90 min)"
            f"  <5 s {np.mean(runs[live] < 5*FPS):.0%}  median {np.median(runs[live])/FPS:.1f} s  stoppages {(~live).sum()}")


if __name__ == "__main__":
    from possession.external import load_game
    for g in (1, 2, 3):
        print(f"Metrica G{g}: {stats(load_game(g)[3])}", flush=True)
    for mid in (sys.argv[1:] or MATCHES):
        P, ball, team, label, period = load_dfl(mid)
        print(f"DFL {mid}: frames {len(label)} slots {P.shape[1]} ball seen {np.mean(~np.isnan(ball[:,0])):.0%} | {stats(label)}", flush=True)


# ---- labels from the EVENTS, with the same logic as the Metrica labels (possession/external.py, CHALLENGE.md) ----
#  - D from a stoppage (the referee's decision time of the next set piece, a foul, an offside, a goal, the final
#    whistle) until the set piece / kick-off is taken
#  - while alive, possession = team of the last controlled action (pass, cross, shot, ball claiming, other ball
#    action, set piece); duels (TacklingGame) and deflections do not change it
SET_PIECES = {"ThrowIn", "GoalKick", "CornerKick", "FreeKick", "Penalty", "KickOff"}
CONTROL = {"Play", "ShotAtGoal", "BallClaiming", "OtherBallAction"}


def _ts(s):
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()


def _half_starts(mid):
    """(epoch time of the first ball frame, frame offset in our array) per half."""
    out, total, want = {}, 0, None
    sec = {"firstHalf": 1, "secondHalf": 2}
    first, last = {}, {}
    with open(_file("04_03_positions_raw_observed", mid)) as fh:
        for line in fh:
            if "<FrameSet" in line:
                m = FS.search(line); want = sec.get(m.group(1)) if m.group(2) == "BALL" else None
                continue
            if want and "<Frame " in line:
                n = int(line.split('N="')[1].split('"')[0]); t = line.split('T="')[1].split('"')[0]
                if want not in first:
                    first[want] = (n, _ts(t))
                last[want] = n
    for s in (1, 2):
        out[s] = (first[s][1], total); total += last[s] - first[s][0] + 1
    return out, total


def event_labels(mid, other_ball_action=True):
    """other_ball_action=False: OtherBallAction (loose touches, clearances) does not give possession."""
    cache = os.path.join(DIR, f"{mid}_evlabels" + ("" if other_ball_action else "_nooba") + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    L, W, home = _meta(mid)
    starts, n = _half_starts(mid)
    period = load_dfl(mid)[4]
    side = lambda team: "H" if team == home else "A"
    marks = []                               # (epoch time, state)
    for ev in ET.parse(_file("03_02_events_raw", mid)).getroot().iter("Event"):
        ch = list(ev)
        if not ch:
            continue
        e, t = ch[0], _ts(ev.get("EventTime"))
        if e.tag in SET_PIECES:
            if e.get("DecisionTimestamp"):
                marks.append((_ts(e.get("DecisionTimestamp")), "D"))
            team = e.get("Team") or next((g.get("Team") for g in e if g.get("Team")), None)
            if team:
                marks.append((t, side(team)))
        elif e.tag in CONTROL and e.get("Team") and (other_ball_action or e.tag != "OtherBallAction"):
            marks.append((t, side(e.get("Team"))))
            if e.tag == "ShotAtGoal" and any(g.tag == "SuccessfulShot" for g in e):
                marks.append((t + 1.0, "D"))         # goal: dead until the kick-off
        elif e.tag in ("Foul", "Offside", "FinalWhistle"):
            marks.append((t, "D"))
    marks.sort()
    # epoch time of every frame
    tf = np.full(n, np.nan)
    for s in (1, 2):
        t0, off = starts[s]
        idx = np.flatnonzero(period == s)
        tf[idx] = t0 + (idx - off) / FPS
    lab = np.full(n, "X", "<U1")
    cur, j = "D", 0
    for i in np.argsort(tf):
        if np.isnan(tf[i]):
            continue
        while j < len(marks) and marks[j][0] <= tf[i]:
            cur = marks[j][1]; j += 1
        lab[i] = cur
    lab[np.isnan(period)] = "X"
    np.save(cache, lab)
    return lab


def compare(mid):
    a, b = load_dfl(mid)[3], event_labels(mid)
    g = (a != "X") & (b != "X")
    return f"agree with provider flags {np.mean(a[g]==b[g]):.0%}"


def snapped_labels(mid, window=50):
    """Event-logic labels whose change times are moved to the nearest provider-flag change into the same state
    (within +-window frames): the events say WHICH changes happen, the frame-accurate flags say WHEN."""
    cache = os.path.join(DIR, f"{mid}_snapped{window}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    ev, fl = event_labels(mid), load_dfl(mid)[3]
    n = len(ev)
    fch = np.flatnonzero(np.r_[False, fl[1:] != fl[:-1]])
    by_state = {s: fch[fl[fch] == s] for s in "HAD"}
    ech = np.flatnonzero(np.r_[False, ev[1:] != ev[:-1]])
    bounds = [0]
    for i in ech:
        s = ev[i]; c = by_state.get(s, np.array([], int))
        j = np.searchsorted(c, i)
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(c) and abs(c[k] - i) <= window and (best is None or abs(c[k] - i) < abs(best - i)):
                best = c[k]
        b = best if best is not None else i
        bounds.append(max(b, bounds[-1]))
    states = [ev[0]] + [ev[i] for i in ech]
    out = np.empty(n, "<U1")
    bounds.append(n)
    for k, s in enumerate(states):
        out[bounds[k]:bounds[k + 1]] = s
    out[ev == "X"] = "X"
    np.save(cache, out)
    return out
