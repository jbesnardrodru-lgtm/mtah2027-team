#!/bin/sh
# Download SoccerTrack (Scott et al., CVPRW 2022; Kaggle atomscott/soccertrack, LGPL-3.0): the wide-view
# (fixed full-pitch camera) videos + annotations, ~0.6 GB, into data/soccertrack/. Needs a Kaggle API token.
#   sh video/get_soccertrack.sh            (re-run to resume)
set -e
cd "$(dirname "$0")/.."
[ -f data/kaggle_files.txt ] || sh video/list_soccertrack.sh
mkdir -p data/soccertrack
for f in $(grep -E "^wide_view/(videos|annotations)/" data/kaggle_files.txt | cut -d, -f1); do
  dst=data/soccertrack/$f
  [ -f "$dst" ] && continue
  mkdir -p "$(dirname "$dst")"
  kaggle datasets download atomscott/soccertrack -f "$f" -p "$(dirname "$dst")" -q
  z="$(dirname "$dst")/$(basename "$f").zip"
  [ -f "$z" ] && { unzip -oq "$z" -d "$(dirname "$dst")" && rm -f "$z"; }
done
echo "SoccerTrack wide view ready: $(ls data/soccertrack/wide_view/videos | wc -l) videos"
