#!/bin/sh
# Download the IDSSE / DFL open data (7 Bundesliga matches 2022/23, CC-BY 4.0; Bassek et al., Sci Data 2025)
# into data/dfl/. Resumable: re-run it if it stops. ~2.8 GB.
#   sh possession/get_dfl.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p data/dfl
B=https://huggingface.co/datasets/pysport/idsse-data/resolve/main
for m in J03WMX:000001 J03WN1:000001 J03WOH:000002 J03WOY:000002 J03WPY:000002 J03WQQ:000002 J03WR9:000002; do
  id=${m%%:*}; c=${m##*:}
  for f in "DFL_02_01_matchinformation_DFL-COM-${c}_DFL-MAT-${id}.xml" \
           "DFL_03_02_events_raw_DFL-COM-${c}_DFL-MAT-${id}.xml" \
           "DFL_04_03_positions_raw_observed_DFL-COM-${c}_DFL-MAT-${id}.xml"; do
    if [ -f "data/dfl/$f.done" ]; then continue; fi
    echo "downloading $f"
    curl -L --fail --retry 5 -C - -o "data/dfl/$f" "$B/$f"
    touch "data/dfl/$f.done"
  done
done
echo "DFL data ready in data/dfl"
