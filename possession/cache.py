"""Parse the raw ATD once into numpy arrays (data/cache/<source>.npz). Every later step loads in <1 s.

Arrays (video-frame indexed, row i = submission row i):
  P      float32 (N, S, 3)  per track slot: x, y (normalised 0-1), speed m/s (NaN = missing)
  ball   float32 (N, 3)     x, y, third channel (probably height)
  team   int8    (S,)       0 = home (Barcelona), 1 = away (Athletic)
"""
import os, sys, xml.etree.ElementTree as ET
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths, source_frames  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache")


def slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    home = [t.get("id") for t in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], np.int8)


def build(source="tactical"):
    paths = asset_paths(source=source)
    team = slot_teams(paths["atd_meta"])
    n, s = source_frames(source), len(team)
    P = np.full((n, s, 3), np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    with open(paths["atd_raw"]) as fh:
        for line in fh:
            fid, players, b = line.rstrip("\n").split(":")
            i = int(fid) - 1  # ATD frame_id counts from 1
            P[i] = np.array([g.split(",") for g in players.split(";")], np.float32)
            ball[i] = np.array(b.split(","), np.float32)
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, source + ".npz")
    np.savez(out, P=P, ball=ball, team=team)
    return out


def load(source="tactical"):
    path = os.path.join(CACHE, source + ".npz")
    if not os.path.exists(path):
        build(source)
    d = np.load(path)
    return d["P"], d["ball"], d["team"]


if __name__ == "__main__":
    for src in sys.argv[1:] or ["tactical"]:
        print("cached", build(src))
