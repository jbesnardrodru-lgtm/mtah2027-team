"""Download Metrica Sports' public sample data (games 1-3) into data/external/ — the lab's training matches.
Source: https://github.com/metrica-sports/sample-data (please acknowledge Metrica Sports when used publicly)."""
import os, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "data", "external")
BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = {
    **{f"G{g}_{f}.csv": f"{BASE}/Sample_Game_{g}/Sample_Game_{g}_{f}.csv"
       for g in (1, 2) for f in ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team")},
    **{f"G3_{f}": f"{BASE}/Sample_Game_3/Sample_Game_3_{f}" for f in ("events.json", "metadata.xml", "tracking.txt")},
}

if __name__ == "__main__":
    os.makedirs(EXT, exist_ok=True)
    for name, url in FILES.items():
        out = os.path.join(EXT, name)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        print("downloading", name)
        urllib.request.urlretrieve(url, out)
    print("external sample data ready in", EXT)
