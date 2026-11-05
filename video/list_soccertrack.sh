#!/bin/sh
# List every file of the SoccerTrack Kaggle dataset (all pages) into data/kaggle_files.txt
cd "$(dirname "$0")/.."
out=data/kaggle_files.txt; : > $out; tok=""
while :; do
  if [ -z "$tok" ]; then kaggle datasets files atomscott/soccertrack --page-size 200 -v > data/_page.txt 2>&1
  else kaggle datasets files atomscott/soccertrack --page-size 200 -v --page-token "$tok" > data/_page.txt 2>&1; fi
  grep -v "^Next Page Token" data/_page.txt >> $out
  tok=$(grep "^Next Page Token" data/_page.txt | sed 's/Next Page Token = //')
  [ -z "$tok" ] && break
done
rm -f data/_page.txt; echo "listed $(grep -c , $out) lines"
