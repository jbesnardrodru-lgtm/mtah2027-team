# Lobos — More than a Hack 2027, Phase 1: ball possession

Our submission for the [More than a Hack 2027](CHALLENGE_README.md) ball-possession challenge (FC Barcelona vs
Athletic Club). For every frame we output `H`, `A` or `D` from the automatic tracking alone.

| | |
|---|---|
| Final version | **v16** (git tag `v16-final`) |
| Lab score (spellF1, 3 held-out external matches) | **0.719** (0.738 / 0.734 / 0.684); official baseline in the same lab 0.25–0.29 |
| Output | `evaluation/results/Lobos.csv` |
| MD5 of the output | `8aec2cd984100e3921b7c0cb51e18d6d` (reproduced on two machines) |
| Runtime | about 35 minutes on a laptop CPU, no GPU |
| Technical document | `Lobos.pdf` (submitted with the form) |

## Reproduce it

```bash
# 1. Python 3.14 with the exact versions we used
pip install -r requirements-lock.txt

# 2. Unpack the organisers' data bundle into data/ (tactical + broadcast tracking, homography; not in git)

# 3. One command: downloads the external training data, trains everything, writes and validates the CSV
sh make_submission.sh Lobos
```

The script prints the MD5 at the end; it must be `8aec2cd984100e3921b7c0cb51e18d6d`. Every stage is skipped when its
output already exists, so an interrupted run resumes. The first run needs internet for the external data
(~2.8 GB, see below).

## How it works

```
tactical tracking (+ ball gaps filled from the synced broadcast tracking)
  -> ~100 team-level features per frame            possession/features.py
  -> gradient boosting (3 Metrica matches)          possession/submit.py
  -> temporal CNN, 3 seeds (Metrica + 7 DFL matches, team-swap augmentation,
     test-time team swap)                           possession/seq.py
  -> blend 0.4 / 0.6 -> sticky Viterbi decoder     possession/blend.py, possession/model.py
  -> evaluation/results/Lobos.csv
```

- **Training data: other matches only.** Nothing is trained or tuned on the provided game, its ground truth or its
  video. Every choice was made in a cross-validated lab of external matches (`possession/lab.py`), degraded by a
  noise simulator to look like our tracking (`possession/noise.py`).
- **Full history** of what we tried, kept and dropped, with lab numbers: [`STATUS.md`](STATUS.md).

## Repository map

| path | what |
|---|---|
| `make_submission.sh` | the one command |
| `possession/` | features, models, decoder, lab, external-data loaders, experiments (`exp_*.py`, `run_*.sh`) |
| `video/` | ball-detector experiments (not used by the submission; see STATUS.md) |
| `evaluation/`, `starter/`, `games.py` | the organisers' scorer, validator and starter code (MIT) |
| `STATUS.md` | lab results and decisions |

## External data and credits

| data | source | licence | used for |
|---|---|---|---|
| Metrica Sports sample data, games 1–3 | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) | as published by Metrica Sports; acknowledged here | training both models; the lab |
| DFL / IDSSE open data, 7 Bundesliga matches | Bassek, Rein, Weber et al., *An integrated dataset of spatiotemporal and event data in elite soccer*, Sci Data 12, 195 (2025), [doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y); downloaded from the [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data) mirror | CC BY 4.0 | training the temporal CNN |

Experiments only (not in the submission): PFF FC World Cup 2022 tracking, SoccerTrack (Scott et al., CVPRW 2022,
LGPL-3.0) and PathCRF (Kim et al., KDD 2026, MPL-2.0). The provided match data is not redistributed here.

The starter code, scorer and public 5-minute ground truth come from the organisers' repository under its MIT
licence ([LICENSE](LICENSE)); the challenge description is in [CHALLENGE_README.md](CHALLENGE_README.md).

An AI coding assistant (Claude, Anthropic) helped write and debug the code.
