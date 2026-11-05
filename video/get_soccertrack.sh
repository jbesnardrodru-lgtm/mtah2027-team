#!/bin/sh
# Download SoccerTrack (Scott et al., CVPRW 2022; Kaggle atomscott/soccertrack, LGPL-3.0): wide-view videos
# (fixed panoramic full-pitch camera) + annotations, ~0.6 GB, into data/soccertrack/wv/{videos,annotations}.
# Needs a Kaggle API token (~/.kaggle/access_token). Re-run to resume.
set -e
cd "$(dirname "$0")/.."
[ -f data/kaggle_files.txt ] || sh video/list_soccertrack.sh
mkdir -p data/soccertrack/wv/videos data/soccertrack/wv/annotations data/soccertrack/raw
for f in $(grep -E "^wide_view/(videos|annotations)/" data/kaggle_files.txt | cut -d, -f1); do
  kind=$(echo "$f" | cut -d/ -f2); base=$(basename "$f")
  [ -f "data/soccertrack/wv/$kind/$base" ] && continue
  kaggle datasets download atomscott/soccertrack -f "$f" -p data/soccertrack/raw -q
  python - "$f" "$kind" "$base" <<'PY'
import sys, os, glob, zipfile, shutil
f, kind, base = sys.argv[1:]
dst = os.path.join("data/soccertrack/wv", kind, base)
cands = glob.glob(os.path.join("data/soccertrack/raw", "**", base + "*"), recursive=True)
for c in cands:
    if c.endswith(".zip"):
        with zipfile.ZipFile(c) as z, z.open(z.namelist()[0]) as s, open(dst, "wb") as d:
            shutil.copyfileobj(s, d)
    else:
        shutil.copyfile(c, dst)
    os.remove(c)
PY
done
echo "SoccerTrack wide view ready: $(ls data/soccertrack/wv/videos | wc -l) videos, $(ls data/soccertrack/wv/annotations | wc -l) annotation files"
