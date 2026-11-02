# Video ball layer (draft section for the technical document)

## Why
The provided tracking loses the ball on 40% of graded frames (64,079 of 158,857), mostly in long stretches.
Our lab shows that coverage, not pinpoint accuracy, is what helps the possession model: recovering 40% of
each gap raised the lab score 0.595 -> 0.617, and even with 30% of the recovered positions deliberately
wrong (5-25 m off) the gain held (0.610). So the video layer is built to fill gaps generously but safely.

## Detector: trained on this match without human labels
- **Pseudo-labels.** A frame enters the training set only when the tracking has the ball on the ground and
  an off-the-shelf YOLO11s box lies within 15 px of the tracking ball projected into the image with the
  provided homography. The detector supplies the box; the tracking confirms it is the ball (not a head or
  the penalty spot). No person labels, selects or corrects anything.
- **Data.** 6,000 sampled frames, cut into 640x640 crops at native resolution (the ball stays a few
  pixels, as at inference), plus one ball-free negative crop per three positives. Every fifth two-minute
  block of the match is held out and never used for training.
- **Training.** YOLO11s fine-tuned for 20 epochs on CPU, seed 0, deterministic algorithms, no mixed
  precision, single-process data loading. The off-the-shelf model was too inaccurate to use directly
  (about half its detections within 25 px).
- **Inference.** Every graded frame, tiled 3x2 into 640 px tiles; top-5 candidates per frame kept,
  positions quantised to 0.5 m and confidences to 0.05 so that floating-point noise cannot change the
  output.

## Linking: from candidates to a ball track
Inside each tracking gap we walk forward from where the tracking last saw the ball and backward from where
it sees it again. A candidate is accepted only if its confidence is at least 0.3, it is on the pitch (within
1 m of the lines), it is physically reachable (at most 30 m/s x elapsed time + 2 m) and it is not a parked
spare ball (still for 2 s with no player within 8 m). A walk stops after 50 frames without a credible
candidate. Mid-gap stretches are then filled with smooth tracklets (confidence >= 0.4, length >= 15 frames).
A 5-frame median makes the recovered path as smooth as the tracking ball.

## How well it works
The broadcast camera, synchronised automatically (one offset per half), gives an independent ball
position on some frames and serves as referee:

| situation | lost-ball frames | filled | within 5 m | median error |
|---|---|---|---|---|
| all | 64,079 | 31% | 68% | 1.9 m |
| live play | 22,094 | 36% | 79% | 1.4 m |
| stoppages | 41,985 | 28% | 46% | 11.9 m |
| crowded (4+ players within 5 m) | 3,232 | 17% | 90% | 1.4 m |
| near (camera-side) touchline | 9,775 | 30% | 57% | 3.4 m |
| in a penalty box | 26,205 | 26% | 73% | 1.0 m |

91% of the balls filled are slow (< 3 m/s).

**These figures are a lower bound.** A team member watched rendered clips of the moments where our
detector and the broadcast disagreed most. In those moments our detector was on the ball and the
broadcast position was off, typically when the TV camera zoomed or panned quickly and its pitch mapping
slipped; this explains most of the large stoppage error. The review was visual only: nothing was labelled
and nothing in the pipeline was changed or tuned because of it.

## Known limitations
The detector misses balls in the air (it was trained on balls on the ground, and they are blurred), balls
seen against the advertising boards, and balls near the touchlines and corners (busy background, spare
balls, ball boys). These are misses, not false positions, so the possession model falls back on player
movement there. We judged that retraining for them (about 12-15 h of CPU) was not worth it for an expected
gain under +0.005.

## Reproducibility
`sh video/run_video_cpu.sh` runs every stage on CPU with fixed seeds and starts with a determinism check
(the same 200 frames detected twice must be identical). Each stage is skipped if its output exists, and a
model is only written when training finishes. From scratch it takes about 12 h on a laptop CPU.
