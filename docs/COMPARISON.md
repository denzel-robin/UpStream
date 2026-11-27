# Delivery modes compared end to end

`scripts/compare_modes.py` runs the live server and client over the same emulated link
for four modes and measures all four quantities. Run on 2026-11-11.

| Mode | What the controller may choose |
|---|---|
| direct | 720p only, any CRF (23 to 43) |
| x2 | 360p + UpStreamVSR x2 only, any CRF (18 to 33) |
| x4 | 180p + UpStreamVSR x4 only, any CRF (18 to 33) |
| adaptive | the full trade-off controller: any rendition and CRF (13 rungs) |

Every mode adapts its CRF to the link with the same controller, so the comparison
shows what switching rendition adds on top of ordinary bitrate adaptation. The server
encodes all 13 rungs in every mode, so encoder load is the same.

**Setup:** RTX 3050 Laptop, float16, third-design checkpoint, playout delay 150 ms,
drop-late 50 ms. Three 30 fps test clips, 30 s each (900 frames). Emulated link 2500
kbit/s, 500 from 10 s, 1500 from 20 s. One run per mode and clip.

**Quality is measured, not predicted.** The client records every reconstructed frame.
The scorer rebuilds what the viewer saw, frame slot by frame slot. A skipped or late
frame counts as the last shown frame held on screen. That is scored against the 720p
reference from the same filter graph. *Viewer PSNR* covers all 900 slots, freezes
included. *Shown PSNR* covers only fresh frames.

## Results, mean of the three clips

| | direct | x2 | x4 | adaptive |
|---|---:|---:|---:|---:|
| **Quality** | | | | |
| Viewer PSNR, dB | 30.74 | **31.75** | 28.93 | 31.64 |
| Shown PSNR, dB | 32.66 | **32.93** | 29.42 | 32.90 |
| Viewer PSNR at 2500 / 500 / 1500 kbit/s | **36.27** / 24.60 / 31.34 | 35.00 / **27.81** / **32.43** | 30.54 / 26.47 / 29.77 | 35.75 / 26.99 / 32.17 |
| **Latency** | | | | |
| Display latency p50 / p95 / max, ms | 155 / 167 / 878 | 156 / 165 / 789 | 155 / **156** / **489** | 155 / 167 / 928 |
| Frames shown / late / skipped (of 2,700) | 2388 / 73 / 239 | 2466 / 60 / 174 | **2610 / 22 / 68** | 2474 / 59 / 167 |
| Freeze events / longest freeze, ms (mean) | 16 / 1844 | 18 / 1167 | **3 / 1000** | 10 / 1533 |
| **Bandwidth** | | | | |
| Mean payload, kbit/s | 579 | 611 | **392** | 640 |
| Sent at 2500 / 500 / 1500, kbit/s | 923 / 256 / 558 | 966 / 242 / 624 | 450 / 198 / 529 | 1064 / 250 / 607 |
| **Resources** | | | | |
| Model GPU time, % of wall time | **0** | 18.3 | 28.1 | 15.6 |
| Model time per frame, ms | **0** | 6.5 | 9.6 | 7.4 |
| GPU utilisation, % (nvidia-smi) | **1.4** | 11.8 | 24.3 | 11.3 |
| GPU power, W | **12.9** | 21.2 | 25.2 | 21.3 |
| Peak GPU temperature, °C | 65 | 63 | 63 | 64 |
| Peak model memory, MiB | **9** | 59 | 94 | 94 |
| Client CPU, % of one core | 56 | 54 | 55 | 55 |
| Decode time per frame, ms | 17.5 | 5.4 | **2.0** | 7.9 |

Bold marks the best of each row.

## Per clip

Viewer PSNR in dB, with the best mode per clip in bold:

| Clip | direct | x2 | x4 | adaptive |
|---|---:|---:|---:|---:|
| NRqG3AEOVGg (scene change at 9 s) | 30.75 | **33.33** | 30.69 | 32.31 |
| LjOrDJOjSMg | 29.33 | 30.06 | 29.21 | **30.30** |
| vT7XBbyKssI | 32.13 | 31.84 | 26.89 | **32.30** |
| Worst loss against the best mode | -2.58 | -0.46 | -5.41 | **-1.02** |

## How much the adaptive controller used each option

| | direct | x2 | x4 |
|---|---:|---:|---:|
| Frames, all clips (of 2,533 received) | 630 (25%) | 1,333 (53%) | 570 (23%) |
| NRqG3AEOVGg | 252 | 247 | 330 |
| LjOrDJOjSMg | 60 | 712 | 90 |
| vT7XBbyKssI | 318 | 374 | 150 |

Rungs, all clips: x2@23 666, direct@23 471, x2@28 367, x4@28 330, x2@33 300,
direct@28 129, x4@33 120, x4@23 120, direct@38 30. Switches: 10, 8 and 12 per
30 s. Exits: the deepest exit on 95% of model frames, the first exit on 5%.

Pattern: direct at 2500 kbit/s where 720p beats any upscale; x4 and x2 during the 500
kbit/s phase; x2, with some direct, at 1500.

## What this shows

- **Adaptive is the safest choice, not the best on every clip.** It is best on two of
  three clips and never more than 1.0 dB behind the best mode. Each fixed rendition
  loses badly somewhere: x4 by 5.4 dB on a clip where 180p cannot hold the detail,
  and direct by 2.6 dB on the clip whose scene change floods the slow link.
- **x2 alone is a strong mode** on this link trace. With the model, 360p is good enough
  at 2500 kbit/s and cheap enough at 500. Its mean is within 0.1 dB of adaptive. That
  difference is smaller than run-to-run noise.
- **Where adaptive loses ground: the moment of the drop.** At 2500 kbit/s it sends
  direct@23, whose keyframes are 22 to 38 KB. When the link falls to 500, one such
  keyframe takes about half a second to cross, and a switch waits for the next keyframe.
  x2 sends smaller keyframes and recovers faster. That is why x2 beats adaptive in the
  500 kbit/s phase (27.81 against 26.99 dB).
- **x4 has the best latency and the fewest freezes** (3 freezes, p95 156 ms, max 489
  ms) and the lowest bitrate. It costs the most GPU (28% of wall time, 25 W) and the
  most quality whenever bandwidth allows more.
- **Direct needs no GPU** (13 W against 21 to 25 W) but freezes the most when the link
  drops, because 720p cannot get cheap enough.
- **Adaptive uses less GPU than either upscaling mode** (15.6% of wall time), because
  it sends direct when bandwidth allows.

## Faults found by this comparison and fixed

The first runs showed every mode skipping hundreds of frames, even on 2500 kbit/s,
and sending a fraction of what a 500 kbit/s link could carry. The causes:

1. **The client drained for the queue behind every keyframe.** P-frames sent while a
   keyframe crosses the link wait for it. That is not congestion, and skipping to the
   next keyframe only repeats it. The controller now drains only on a standing queue,
   or on queuing beyond what the last keyframe explains (docs/CONTROLLER.md).
2. **The server guard fired on a shrinking backlog.** It now fires only while the
   backlog grows.
3. **The emulated link counted a record as fully queued until its last byte left**,
   so a keyframe looked like a growing backlog for its whole transmission.
4. **The server acted on frames when they left the encoder, not at their release
   time.** During a skip it ran up to a second ahead. Switch requests then missed the
   keyframe they were meant for, and a 22 to 27 KB direct keyframe went out on the
   500 kbit/s link. The guard also read the backlog before anything had drained.

Over the four fixes, adaptive's mean viewer PSNR rose from 28.7 to 31.6 dB. Skipped
frames fell from 615 to 167. Raw results of each step are in `results/compare-*`
(not committed).

One run hit an ffmpeg crash (segfault after 14 frames, encoding 13 outputs). The script
now reruns a mode whose encoder fails; it was not needed in the final run.

## Limits

- One link trace, three clips, one run each. Differences under about 0.5 dB are within
  noise.
- One machine, emulated link without propagation delay or loss (tc netem is next).
- Viewer PSNR is dominated by freezes in the 500 kbit/s phase. PSNR does not weigh a
  freeze the way a viewer does.
- The adaptive controller predicts quality from the table's means. It does not know
  that x2 suits a given clip better than direct.

## Possible improvements suggested by these results

- **Prefer smaller keyframes when the link is uncertain.** Price each rung's keyframe
  against a possible drop, for example by requiring the keyframe to fit a fraction of
  the current capacity. Adaptive would then hold x2 instead of direct@23 when the gain
  is small.
- **Shorter keyframe interval at low rates**, so a switch takes effect sooner (changes
  the training distribution).

## Commands

```bash
python scripts/compare_modes.py data/test-clips/NRqG3AEOVGg@1440.mp4 \
    data/test-clips/LjOrDJOjSMg@1440.mp4 data/test-clips/vT7XBbyKssI@1440.mp4 --half
# results/compare/summary.json, results/compare/<clip>/table.md, per-run reports
```

## Over a real network path (tc netem, 2026-11-11)

The same comparison over real TCP. The client runs in a network namespace, joined
to the server by a veth pair (`live/netem.py`, `scripts/run_netem.sh`). netem shapes
the server-to-client direction: the same rate schedule, plus delay, jitter, loss and a
300-packet router queue. The return direction gets the same delay, jitter and loss.
Offloads are off, so netem shapes real 1500-byte packets. Both ends share one clock,
so latency is still exact.

| Profile | One-way delay | Jitter | Loss | Packets lost (measured) |
|---|---|---|---|---|
| broadband | 20 ms | 5 ms | 0.1% | 11 of 20,444 |
| mobile | 40 ms | 15 ms | 1% | 115 of 14,496 |

Mean of the three clips, all four quantities:

| | direct | x2 | x4 | adaptive |
|---|---:|---:|---:|---:|
| **Broadband** | | | | |
| Viewer PSNR, dB | 27.77 | 28.33 | 28.55 | **30.47** |
| Viewer PSNR at 2500 / 500 / 1500 | **35.92** / 18.88 / 28.51 | 34.94 / 20.33 / 29.73 | 30.54 / **25.46** / 29.64 | 35.11 / 24.48 / **31.82** |
| Latency p50 / p95 / max, ms | 150 / 159 / 426 | 156 / 163 / 409 | 156 / **157** / 420 | 155 / 167 / 694 |
| Frames shown / late / skipped | 1824 / 29 / 847 | 1993 / 45 / 662 | **2555 / 16 / 129** | 2288 / 44 / 368 |
| Longest freeze, ms | 4678 | 4633 | **967** | 2278 |
| Payload, kbit/s | 471 | 491 | **385** | 545 |
| Model GPU time, % / power, W | **0 / 12.9** | 13.9 / 20.3 | 27.6 / 24.7 | 15.3 / 21.3 |
| **Mobile** | | | | |
| Viewer PSNR, dB | 24.04 | 26.53 | 25.95 | **27.44** |
| Viewer PSNR at 2500 / 500 / 1500 | 32.04 / 16.25 / 23.83 | **33.18** / 18.64 / **27.77** | 29.51 / 23.51 / 24.84 | 31.79 / **24.70** / 25.82 |
| Latency p50 / p95 / max, ms | 151 / 377 / 1255 | 156 / 214 / 438 | 156 / **186** / **285** | 155 / 217 / 1267 |
| Frames shown / late / skipped | 1344 / 95 / 1261 | 1640 / 131 / 929 | **1900** / 74 / **726** | 1811 / 113 / 776 |
| Longest freeze, ms | 6222 | 4522 | **1389** | 2378 |
| Payload, kbit/s | 302 | 310 | **250** | 363 |
| Model GPU time, % / power, W | **0 / 12.8** | 13.6 / 19.7 | 21.7 / 22.7 | 13.6 / 20.6 |

Viewer PSNR per clip:

| Clip | Profile | direct | x2 | x4 | adaptive |
|---|---|---:|---:|---:|---:|
| NRqG3AEOVGg | broadband | 25.40 | 25.89 | 29.68 | **31.33** |
| LjOrDJOjSMg | broadband | 26.89 | 29.24 | **29.28** | 28.68 |
| vT7XBbyKssI | broadband | 31.02 | 29.86 | 26.68 | **31.41** |
| NRqG3AEOVGg | mobile | 21.89 | 24.62 | 26.36 | **27.64** |
| LjOrDJOjSMg | mobile | 22.74 | 24.67 | **26.72** | 25.66 |
| vT7XBbyKssI | mobile | 27.49 | **30.31** | 24.79 | 29.02 |

Frames per rendition chosen by adaptive, all three clips, per link phase:

| Profile | Link | direct | x2 | x4 | skipped |
|---|---|---:|---:|---:|---:|
| broadband | 2500 | 464 | 300 | 90* | 46 |
| | 500 | 17 | 145 | 451 | 287 |
| | 1500 | 25 | 720 | 120 | 35 |
| mobile | 2500 | 410 | 146 | 90* | 254 |
| | 500 | 19 | 129 | 568 | 184 |
| | 1500 | 59 | 408 | 95 | 338 |

\*The first 30 frames of every run are the starting rung, x4@28.

### What the real network shows

- **Adaptive has the best mean on both real profiles**: 1.9 dB ahead of the next mode on
  broadband, 0.9 dB on mobile. On the emulated link it tied with x2. Real TCP punishes
  the fixed high-rate modes harder.
- **Direct-only collapses when the link is below its cheapest rung.** On clip A at
  500 kbit/s, direct@43 costs 710 kbit/s, so the guard drops every group and only
  one keyframe per second gets through. That gave a 4.7 s freeze on broadband and
  6.2 s on mobile.
- **x4 is the most robust mode under loss**: the fewest late and skipped frames and the
  shortest freezes on both profiles. Its frames are small, so a retransmission delays
  less.
- **Adaptive switches to x4 when the link drops.** At 500 kbit/s it used x4 for 451
  (broadband) and 568 (mobile) of the frames it sent. It uses x2 at 1500 and direct at
  2500.
- **Loss costs latency.** With 1% loss and an 80 ms round trip, p95 latency rises
  from about 160 ms (broadband) to 186–377 ms. A lost packet stalls everything behind it
  for at least one round trip (TCP head-of-line blocking). Late frames rise two- to
  five-fold, and skips rise even at 2500 kbit/s.

### A weakness this exposed (not yet fixed)

When only keyframes get through (every group dropped), no packet finds the link idle
twice within the fit window. The client then has no capacity estimate, and the server
guard keeps the last one it was sent. On clip A, direct-only kept dropping groups from
20 s, when the link rose to 1500, until 25 s.

Proposed fix: estimate capacity from a single keyframe that found the link idle, as
size / (delay - fixed delay). Also expire the guard's capacity when the client stops
updating it.

### Commands

```bash
sudo bash scripts/run_netem.sh smoke   # 8 s check
sudo bash scripts/run_netem.sh full    # both profiles, about 25 minutes
```

## Five-minute sustained run (2026-11-11)

The success criterion: 5 minutes at 30 fps with no collapse. The earlier NanoVSR study
failed it, dropping to 20.87 fps and stopping at 80 s.

Source: ten test clips, 30 s each, joined at 30 fps and 720p
(`scripts/make_sustained_source.sh`; 9,000 frames). The emulated link changed every
30 s: 2500, 1500, 500, 1000, 3000, 700, 2000, 400, 1200, 2500 kbit/s. Two modes ran:
adaptive (the full system) and x4 only (the heaviest GPU load). FP16, RTX 3050 Laptop.

| | adaptive | x4 only |
|---|---:|---:|
| Frames released by the server on time | 9000 / 9000 | 9000 / 9000 |
| Encoder late releases | 0 | 0 |
| Frames shown / late / skipped | 8504 / 124 / 372 | 8852 / 34 / 114 |
| Display latency p50 / p95 / max, ms | 155 / 160 / 948 | 155 / 157 / 354 |
| Viewer / shown PSNR, dB | **30.87** / **31.73** | 29.16 / 29.37 |
| Mean payload, kbit/s | 638 | 492 |
| Model GPU time, % of wall time | 17.4 | 28.8 |
| GPU power, W (mean) | 20.7 | 25.3 |
| GPU temperature, °C | 59–63 | 60–65 |
| Peak model memory, MiB | 93.6 | 93.6 |
| Client CPU, % of one core | 58 | 60 |
| Rendition share (adaptive) | direct 20%, x2 54%, x4 22%, skipped 4% | |

Per 30-second window (each window is a different clip, so PSNR changes with content as
well as with the link):

| From | Link, kbit/s | Adaptive: shown / late / skipped | Latency p50 / p95, ms | Model ms | GPU °C | Adaptive used |
|---:|---:|---|---|---:|---:|---|
| 0 s | 2500 | 840 / 16 / 44 | 155 / 159 | 6.9 | 60 | x2 510, direct 316, x4 30 |
| 30 s | 1500 | 800 / 5 / 95 | 155 / 161 | 6.8 | 59 | x2 640, direct 165 |
| 60 s | 500 | 867 / 5 / 28 | 155 / 158 | 7.9 | 60 | x2 430, x4 352, direct 90 |
| 90 s | 1000 | 830 / 3 / 67 | 155 / 158 | 7.9 | 61 | x2 473, x4 360 |
| 120 s | 3000 | 871 / 6 / 23 | 155 / 156 | 6.6 | 60 | direct 817, x2 60 |
| 150 s | 700 | 839 / 21 / 40 | 156 / 159 | 8.0 | 61 | x2 523, x4 330 |
| 180 s | 2000 | 864 / 7 / 29 | 156 / 159 | 7.0 | 61 | x2 811, direct 60 |
| 210 s | 400 | 826 / 56 / 18 | 155 / 157 | 9.7 | 63 | x4 861, x2 21 |
| 240 s | 1200 | 869 / 3 / 28 | 155 / 160 | 6.9 | 62 | x2 752, direct 90 |
| 270 s | 2500 | 898 / 2 / 0 | 155 / 159 | 6.6 | 62 | x2 660, direct 240 |

**Passed.**
- The pipeline kept 30 fps for the whole 5 minutes in both modes: every frame was
  released on time, and late and skipped frames come from the link, not from the
  pipeline falling behind.
- Latency did not drift: the median stayed at 155 ms (the 150 ms playout delay plus
  transport) in every window.
- Model time, memory (93.6 MiB) and temperature (at most 65 °C) stayed flat.
- The last window, back at 2500 kbit/s, showed 898 of 900 frames.

x4 only is the steadier stream (fewer skips, lower worst-case latency) but scores
1.7 dB lower and uses 65% more GPU time.

## 18-minute long run: does the recurrent state drift? (2026-11-13)

The model never resets its state, not even at keyframes (the I-frame flag only feeds
the gate), so a slow drift would show only in long streams. `scripts/long_run.py`
feeds one continuous stream to the model with no reset: all 18 test clips, 30 s each
(9 minutes), played twice in a row (18 minutes, 32,410 frames;
`scripts/make_long_source.sh`). The x2@23 and x4@23 streams use the live x264 settings
(keyframe every 30 frames), each with its own state, float16, deepest exit. Bicubic
runs on the same frames as a reference without memory. The second pass repeats the
first frame for frame, so it shows the same content after 9 more minutes of
accumulated state.

| | x2@23 | x4@23 |
|---|---:|---:|
| PSNR, model / bicubic, dB | 33.96 / 32.74 | 29.76 / 28.42 |
| Gain over bicubic, mean (lowest 30 s window), dB | +1.22 (+0.49) | +1.33 (+0.68) |
| **Second pass minus first, same clip, mean (range over 18 clips), dB** | **−0.000 (−0.016 to +0.019)** | **+0.003 (−0.016 to +0.029)** |
| Largest state value, first / last window (overall) | 13.5 / 14.0 (18.3) | 12.1 / 9.8 (16.4) |
| Frames with NaN or inf | 0 | 0 |
| Model time per frame, median (slowest after the first two frames) | 3.9 ms (4.8) | 7.0 ms (11.4) |

**No collapse.**
- On identical content, the gain after 9 minutes of carried state matches the first
  pass to within 0.03 dB on every clip, with no sign either way.
- The state stays bounded. Its largest value tracks the content (8–18, the same in
  both passes) and does not grow.
- Model time stayed at 7.0 ms for x4 in every window. The GPU ran at 78–81 °C,
  warmer than the live run because this ran flat out, faster than real time.
- The gain moves between clips (+0.5 to +2.8 dB) with content, not with time. Its
  linear trend over 18 minutes (−0.01 to −0.02 dB per minute) repeats in both passes,
  so it comes from the order of the clips.

`python scripts/long_run.py` (about 20 minutes on the RTX 3050 Laptop; results in
results/long/: summary.json per 30 s window, frames.npz per frame).

## Re-run with the final model, no recurrent state (2026-11-18)

The final model (`runs/kaggle-7-no-state`, docs/MODEL.md) replaces the model with
recurrent state. The rate-quality table (configs/rate_quality.json) was re-measured with
it, and the emulated-link comparison and the 5-minute sustained run repeated with the
same clips, link and settings (results/final/). The 18-minute long run above tested the
recurrent state, which the final model no longer has.

Emulated link (three clips, sums or means; earlier model in brackets):

| Mode | Viewer PSNR dB | Shown / late / skipped | Latency p95 ms | Model GPU % | GPU W | kbps |
|---|---:|---:|---:|---:|---:|---:|
| direct | 30.75 (30.74) | 2391 / 71 / 238 | 168 | 0.0 | 13.1 | 582 |
| x2 | 31.84 (31.75) | 2497 / 49 / 154 | 162 | 18.4 (18.3) | 21.4 | 611 |
| x4 | 28.95 (28.93) | 2610 / 22 / 68 | 156 | **19.6 (28.1)** | **22.4 (25.2)** | 392 |
| **adaptive** | **31.49** (31.64) | 2474 / 59 / 167 | 163 | **13.9 (15.6)** | 20.7 (21.3) | 643 |

Five-minute sustained run:

| | adaptive | x4 only |
|---|---:|---:|
| Frames shown / late / skipped | 8507 / 121 / 372 | 8858 / 29 / 113 |
| Display latency p50 / p95 / max, ms | 155 / 160 / 945 | 155 / 156 / 350 |
| Viewer / shown PSNR, dB | 30.85 / 31.71 (30.87 / 31.73) | 29.10 / 29.30 (29.16 / 29.37) |
| Model time per frame, mean, ms | 6.6 (7.6) | 6.8 (9.7) |
| Model GPU time, % of wall time | 15.1 (17.4) | **20.1 (28.8)** |
| GPU power, W (mean) | 20.9 (20.7) | 22.4 (25.3) |
| Peak model memory, MiB | **69 (94)** | **69 (94)** |
| GPU temperature, max °C | 63 (63) | 63 (65) |

- **Quality is unchanged within run-to-run noise.** x4 only, where delivery is
  identical, matches to 0.02–0.06 dB. Adaptive is 0.15 dB lower here, but the earlier
  model's own adaptive runs ranged from 31.50 to 31.64 dB across the three idle-GPU sets in this
  repository (comparison, ablation, baselines); the final model's three are 31.39–31.49 dB. The difference is at
  most about 0.1 dB.
- **The cost falls.** At x4 the model uses 30% less GPU time (19.6% vs 28.1%) and 3 W
  less, and peak memory drops from 94 to 69 MiB, because no state is kept.
- **Still 30 fps for 5 minutes, latency flat** at a 155 ms median; the controller made
  the same rendition choices (x2 54%, direct 20%, x4 22%).

### tc netem, final model (2026-11-18)

`sudo bash scripts/run_netem.sh full` repeated with the final model; same clips,
profiles and rate schedule (results/netem-broadband, results/netem-mobile; the earlier
runs are kept in results/netem-*-with-state). Sums or means over three clips, earlier
model in brackets:

| Profile | Mode | Viewer PSNR dB | Shown / late / skipped | Latency p95 ms | Model GPU % |
|---|---|---:|---:|---:|---:|
| broadband | direct | 28.03 (27.77) | 1830 / 31 / 839 | 156 | 0.0 |
| broadband | x2 | 27.73 (28.33) | 1884 / 65 / 751 | 170 | 13.5 |
| broadband | x4 | 28.85 (28.55) | 2583 / 12 / 105 | 157 | 20.0 (27.6) |
| broadband | **adaptive** | **30.35** (30.47) | 2297 / 59 / 344 | 170 | 12.6 (15.3) |
| mobile | direct | 24.58 (24.04) | 1295 / 111 / 1294 | 264 | 0.0 |
| mobile | x2 | 25.72 (26.53) | 1436 / 108 / 1156 | 237 | 12.1 |
| mobile | x4 | 27.42 (25.95) | 2217 / 35 / 448 | 161 | 17.5 (21.7) |
| mobile | **adaptive** | **28.00** (27.44) | 1886 / 130 / 684 | 227 | 11.2 (13.6) |

- **Adaptive is still best on both profiles** (three clips; on 10 clips it is level
  with x4 only, see the last section): +1.5 dB over the best fixed mode on
  broadband and +0.6 dB on mobile (earlier model: +1.9 and +0.9). Per clip it is best
  in 3 of 6 runs and within 0.3 dB of the best in 2 more. It loses clearly only on
  LjOrDJOjSMg over mobile, to x4 only (27.40 vs 25.20 dB), as with the earlier model.
- **Real-network runs vary much more than the model does.** Direct delivery uses no
  model, yet moved by 0.3–0.5 dB between the two sets, and x2 and x4 by 0.3–1.5 dB:
  TCP loss recovery and jitter differ from run to run. The changes against the
  earlier model are within that spread.
- The model's GPU time falls by 18–28%, as on the emulated link.

## 18 clips, confidence intervals, VMAF and LPIPS, and an MPC baseline (2026-11-25)

The emulated-link comparison repeated on all 18 test clips, with ten modes per clip:

- the four delivery modes;
- the four published models in the adaptive controller (docs/BASELINES.md);
- two runs of a standard adaptive-bitrate controller in place of ours.

Same link (2500, 500 from 10 s, 1500 from 20 s kbit/s), 30 s per clip, final model,
float16 (results/multiclip, 180 runs).

**MPC baseline.** RobustMPC (Yin et al., SIGCOMM 2015), `MpcController` in
live/controller.py, adapted to live delivery:

- **Throughput:** the harmonic mean of the last 5 per-GOP capacity samples, divided by
  1 + the largest recent relative prediction error.
- **Planning:** exhaustive search, with pruning, over the next 5 GOPs. It maximises
  Σ PSNR − μ·stall − |ΔPSNR|. A stall is the part of the sender queue above the latency
  budget, and μ is the highest quality on the ladder.
- **Scope:** it decides once per keyframe and uses the same predicted-PSNR table.
- Two runs:
  - **mpc**, over the full 13-rung ladder;
  - **mpc-direct**, over direct rungs only. This is a conventional ABR stream with no
    upscaling.

**New metrics.** VMAF (libvmaf, vmaf_v0.6.1) and LPIPS (AlexNet; lower is better) are
computed on what the viewer saw, as viewer PSNR is: late or skipped slots count as the
held frame.

**Statistics.** The clip is the unit, with a t-based 95% interval over the 18 clips.
Differences are paired by clip: Δ = adaptive − mode, followed by the number of clips on
which adaptive is better (`scripts/summarise_clips.py`, results/multiclip/ci.md).

**Run validity.**
- Every run records:
  - the CPU load before it started;
  - the median clock of the fastest core;
  - the server's largest send lag.
- compare_modes.py waits for an idle machine and rejects runs below 2.5 GHz.
- Two earlier attempts were discarded: one under CPU contention, and one with the
  laptop's power limit stuck at 11 W (1.4 GHz, client decode 42 ms per frame).
- In the kept runs:
  - the CPU ran at 4.0–4.3 GHz;
  - the encoder kept up within 5 ms on every clip, the 1440p60 sources included.
  - 77 runs from the first session predate the health record. They ran from 13:11 to
    14:58 on 2026-11-24, before the contention began.

Mean over clips (± 95% CI):

| Mode | Viewer PSNR dB | VMAF | LPIPS | p95 ms | Late + skipped | Longest freeze ms | kbps | Model GPU % | GPU W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 30.23 ± 1.91 | 59.1 ± 6.5 | 0.177 | 170 ± 10 | 94 ± 38 | 1384 | 633 | 0.0 | 14.0 |
| x2 | 30.82 ± 1.77 | **65.7** ± 5.9 | **0.163** | 167 ± 6 | 80 ± 32 | 1280 | 686 | 17.0 | 22.3 |
| x4 | 29.21 ± 1.50 | 54.8 ± 4.5 | 0.224 | **157** ± 0 | **26** ± 7 | **809** | 452 | 18.8 | 23.4 |
| **adaptive** | **30.84** ± 1.56 | 63.0 ± 5.1 | 0.175 | 161 ± 2 | 65 ± 13 | 1278 | 654 | 13.0 | 21.4 |
| mpc (full ladder) | 29.51 ± 1.74 | 55.6 ± 7.1 | 0.215 | 194 ± 28 | 107 ± 23 | 2327 | 567 | 12.4 | 21.3 |
| mpc-direct | 29.03 ± 2.02 | 54.1 ± 7.3 | 0.211 | 201 ± 26 | 132 ± 36 | 2384 | 643 | 0.0 | 13.8 |
| adaptive, EfRLFN | 29.89 ± 1.60 | 51.1 ± 5.5 | 0.220 | 159 ± 2 | 59 ± 13 | 1290 | 600 | 23.3 | 38.4 |
| adaptive, SPAN | 30.12 ± 1.58 | 61.1 ± 5.4 | 0.179 | 158 ± 1 | 57 ± 13 | 1277 | 608 | 20.0 | 36.3 |
| adaptive, NanoVSR | 30.27 ± 1.56 | 57.8 ± 5.1 | 0.179 | 157 ± 1 | 59 ± 12 | 1264 | 598 | 18.7 | 32.3 |
| adaptive, CDA-VSR | 29.84 ± 1.85 | 57.8 ± 6.4 | 0.185 | 181 ± 11 | 108 ± 39 | 1373 | 613 | 4.1 | 17.3 |

The absolute intervals are wide because the clips differ in difficulty. The paired
differences remove that spread:

| Adaptive minus | Viewer PSNR dB | VMAF | LPIPS (negative: adaptive better) | p95 ms |
|---|---:|---:|---:|---:|
| direct | +0.61 ± 0.69 (10/18) | **+3.9 ± 2.1** (15/18) | −0.002 ± 0.019 (8/18) | −9 ± 11 (8/18) |
| x2 | +0.02 ± 0.51 (8/18) | **−2.7 ± 1.7** (3/18) | **+0.012 ± 0.010** (2/18) | **−6 ± 6** (16/18) |
| x4 | **+1.62 ± 0.70** (18/18) | **+8.2 ± 2.6** (17/18) | **−0.049 ± 0.014** (17/18) | +4 ± 2 (1/18) |
| mpc | **+1.32 ± 0.44** (16/18) | **+7.4 ± 2.5** (16/18) | **−0.039 ± 0.015** (17/18) | **−33 ± 26** (16/18) |
| mpc-direct | **+1.81 ± 0.82** (15/18) | **+8.9 ± 3.0** (16/18) | **−0.036 ± 0.023** (13/18) | **−40 ± 25** (14/18) |
| adaptive, EfRLFN | **+0.95 ± 0.25** (18/18) | **+11.9 ± 2.1** (18/18) | **−0.045 ± 0.015** (18/18) | +2 ± 2 (5/18) |
| adaptive, SPAN | **+0.72 ± 0.27** (15/18) | **+1.9 ± 0.9** (16/18) | −0.004 ± 0.006 (12/18) | +3 ± 2 (3/18) |
| adaptive, NanoVSR | **+0.57 ± 0.24** (16/18) | **+5.2 ± 1.4** (18/18) | −0.004 ± 0.007 (9/18) | +3 ± 2 (3/18) |
| adaptive, CDA-VSR | **+1.00 ± 0.66** (13/18) | **+5.3 ± 2.0** (16/18) | −0.010 ± 0.017 (10/18) | **−20 ± 12** (12/18) |

Bold: the 95% interval excludes zero.

### What the 18 clips show

- **Against MPC, the controller is better on every quality metric and on latency.**
  Over the same 13-rung ladder and quality table, the trade-off controller gains:
  - +1.32 ± 0.44 dB and +7.4 VMAF;
  - lower LPIPS on 17 of 18 clips;
  - half the longest freeze (1.3 s against 2.3 s);
  - a p95 latency 33 ms lower.
- **Why MPC loses.** Most of the loss is in the drop to 500 kbit/s: 25.4 dB against
  27.0 dB in that phase. Three things cause it:
  - The harmonic mean of past GOPs lags a sudden drop.
  - MPC commits for a whole GOP and has no backlog guard. It made 60 backlog skips over
    the 18 clips, against 26 for the controller.
  - After the drop, the large prediction error keeps its robustness discount high. In
    the last phase it uses 26% of the 1500 kbit/s link, against 42% for the controller.
- **Upscaling helps MPC too.** mpc-direct, a conventional ABR stream, is 0.48 dB below
  MPC over the full ladder, close to the 0.61 dB between direct-only and adaptive
  delivery.
- **Against the published models, the gains hold on 18 clips:** +0.57 dB (NanoVSR) to
  +1.00 dB (CDA-VSR) in viewer PSNR, and +1.9 to +11.9 VMAF.
  - Against SPAN, NanoVSR and CDA-VSR, LPIPS is within noise.
  - The others use 32–38 W of GPU power against 21 W.
  - The three-clip estimates (+0.95 to +1.33 dB) were high: those three clips favoured
    UpStreamVSR more than the average.
  - Offline, SPAN and CDA-VSR have higher VMAF than UpStreamVSR at ×4
    (docs/BASELINES.md). Live, they score lower, because neither can run ×2 in time.
- **Against fixed delivery:**
  - **×4 only:** adaptive is ahead on every quality metric, on 17–18 of 18 clips.
  - **Direct only:** adaptive is ahead in VMAF (+3.9, 15/18). The PSNR gain
    (+0.61 dB) is not significant, and LPIPS is level.
  - **×2 only** is the honest exception:
    - PSNR is tied (+0.02 ± 0.51 dB).
    - ×2 only is better perceptually: 2.7 VMAF points higher, and lower LPIPS on 16 of
      18 clips.
    - Adaptive gives lower latency (16/18), 4 points less model GPU time, 0.9 W less
      power and 5% fewer bits.
    - The likely cause is the controller's switching. It maximises predicted PSNR, and
      at mid rates direct and ×2 are close in PSNR. So it moves to direct when the link
      allows (28% of frames). ×2 + UpStreamVSR scores higher than direct in VMAF and
      LPIPS, as the offline table shows. A quality table in VMAF, or a switching
      penalty, would address this; neither has been tried.
- **The three-clip ranking of ×2 only vs adaptive was clip luck.** On three clips ×2
  only led by 0.35 dB. On 18 clips, PSNR is tied, and in latency and cost the
  controller is better.

### Commands

```
python scripts/compare_modes.py data/test-clips/*.mp4 --half --resume --out results/multiclip \
    --modes direct x2 x4 adaptive mpc mpc-direct adaptive-efrlfn adaptive-span adaptive-nanovsr adaptive-cdavsr
python scripts/summarise_clips.py results/multiclip/summary.json --json results/multiclip/ci.json --markdown results/multiclip/ci.md
```

## tc netem on 10 clips, with confidence intervals and RobustMPC (2026-11-25)

> **Superseded** by the re-run of 2026-11-27 at the end of this file. These runs were
> distorted by a backlog-guard bug; the numbers below are kept for the record.

`sudo bash scripts/run_netem.sh multi`. The first 10 test clips run over real TCP
shaped with tc netem. Same profiles and rate schedule as above; modes direct, x2, x4,
adaptive, mpc and mpc-direct; final model (results/multiclip-netem-broadband,
results/multiclip-netem-mobile; 120 runs).

**Run validity:**
- **One run redone.** Broadband p-kAI-qOeJA mpc-direct ran while other work loaded the
  CPU: its 720p decode was 30% slower than in the clean emulated run, and the clock was
  4.02 GHz. It was redone (results/multiclip-netem-broadband).
- **All others are clean:**
  - CPU at 4.0–4.3 GHz;
  - decode times at or below the clean runs of the same clips.
- **Two mobile MPC runs** (LjOrDJOjSMg mpc, NRqG3AEOVGg mpc-direct) show up to 0.9 s
  of server release lag. This lag comes from the network, not the machine:
  - The encoder stayed 33 ms ahead and the CPU clock was normal.
  - The sender's blocking write stalled because MPC sent more than the lossy link
    carried, with no backlog guard.
  - That is part of what this test measures, so the runs are kept.

Adaptive minus each mode, paired by clip, ± 95% CI (clips adaptive wins). Bold: the
interval excludes zero.

| Profile | vs | Viewer PSNR dB | VMAF | LPIPS | p95 ms | Late + skipped | Longest freeze ms |
|---|---|---:|---:|---:|---:|---:|---:|
| broadband | direct | **+2.14 ± 1.61** (8/10) | **+6.7 ± 4.8** | **−0.055 ± 0.049** | −3 ± 21 | **−116 ± 103** | −2227 ± 2940 |
| broadband | x2 | +1.76 ± 2.27 (6/10) | +2.6 ± 8.0 | −0.042 ± 0.065 | −3 ± 14 | −81 ± 123 | −1786 ± 2629 |
| broadband | x4 | +0.86 ± 1.04 (7/10) | **+7.1 ± 4.0** | **−0.022 ± 0.020** | **+12 ± 9** | **+96 ± 33** | **+891 ± 473** |
| broadband | mpc | **+1.83 ± 0.77** (9/10) | **+10.0 ± 3.7** | **−0.052 ± 0.035** | +4 ± 9 | −42 ± 59 | **−2538 ± 2075** |
| broadband | mpc-direct | **+3.77 ± 2.28** (9/10) | **+13.9 ± 8.0** | **−0.092 ± 0.067** | −38 ± 54 | **−148 ± 103** | **−4330 ± 3781** |
| mobile | direct | **+2.97 ± 2.06** (7/10) | **+6.4 ± 5.6** | **−0.063 ± 0.053** | +7 ± 39 | **−172 ± 107** | **−2806 ± 2410** |
| mobile | x2 | +1.56 ± 2.03 (6/10) | −1.3 ± 5.4 | −0.025 ± 0.050 | +6 ± 46 | −80 ± 107 | −2078 ± 2316 |
| mobile | x4 | −0.04 ± 0.60 (5/10) | +1.4 ± 3.1 | −0.002 ± 0.020 | +39 ± 40 | **+87 ± 64** | +520 ± 655 |
| mobile | mpc | **+1.44 ± 0.86** (9/10) | **+5.8 ± 5.4** | **−0.029 ± 0.026** | +13 ± 33 | −46 ± 78 | **−1704 ± 1191** |
| mobile | mpc-direct | **+2.92 ± 2.27** (8/10) | +6.5 ± 7.8 | −0.066 ± 0.065 | **−51 ± 48** | **−140 ± 125** | **−4324 ± 3388** |

Adaptive itself: 30.07 ± 2.56 dB on broadband and 27.19 ± 2.33 dB on mobile.

### What the 10 clips show

- **Against RobustMPC, the controller is ahead on both profiles:**
  - +1.83 ± 0.77 dB (broadband) and +1.44 ± 0.86 dB (mobile), on 9 of 10 clips each;
  - higher VMAF and lower LPIPS;
  - a longest freeze 1.7–2.5 s shorter.
  The gain is larger than on the emulated link (+1.32 dB). Over real TCP, MPC's lagging
  estimate and lack of a backlog guard cost more.
- **Against 720p delivery, with or without MPC, the gain is large and significant:**
  +2.1 to +3.8 dB.
- **×4 only is not beaten.**
  - On broadband, adaptive is +0.86 ± 1.04 dB in PSNR (not significant). It is
    significantly ahead in VMAF (+7.1) and LPIPS.
  - On mobile it ties: −0.04 ± 0.60 dB, with VMAF and LPIPS level.
  - ×4 only has the lowest bitrate, so it has the fewest late and skipped frames, the
    shortest freezes and the lowest p95 latency on both profiles.
  - On mobile, adaptive's late and skipped frames are 87 higher, and its p95 latency
    39 ± 40 ms higher.
- **×2 only:** adaptive is ahead on average, by 1.6–1.8 dB, but the interval includes
  zero. ×2 only fails badly on some clips and holds level on others.
- **This revises the three-clip result.** On three clips, adaptive beat every fixed
  mode, by +1.5 dB (broadband) and +0.6 dB (mobile). On 10 clips:
  - it clearly beats direct-only delivery and RobustMPC;
  - it is level with ×4 only, which has better latency and freezes.
  On a lossy, high-delay path, the safest fixed choice (always the smallest stream)
  is hard to beat. The controller still pays for its moves to larger streams in late
  frames.
- **Possible improvements:**
  - make the capacity estimate account for loss and RTT, so the controller reacts to
    TCP's effective rate;
  - lower the latency budget on lossy paths.

## tc netem re-run after the guard fix, with a loss-aware controller (2026-11-27)

This section replaces the real-TCP numbers of 2026-11-25 above. Those runs were
distorted by a bug, described below. The emulated-link results are not affected: the
bug needs a lossy path to trigger, and the emulated runs are essentially free of it.

### The bug: the backlog guard could lock on a stale estimate

- **The client reported capacity only when it changed** (a C record when the estimate
  moved by more than 10%).
- **A low estimate could therefore stay in force indefinitely.** After a loss burst, the
  server's backlog guard compared the send backlog with that stale low rate and skipped
  every frame of each GOP except its keyframe.
- **One keyframe per second measures nothing**, so the estimate never recovered, and the
  picture froze for 5–24 s although the link had room again.
- **Seen in 49 of the 160 earlier netem runs**, in every mode that uses the guard,
  including adaptive.
- **The fix (commit 75fb9f4):** the controller records when its estimate was last
  measured. If that is more than 3 s ago, the client withdraws the estimate (sends
  `kbps: null`), and the guard stands down until a fresh measurement arrives. The
  withdrawals are logged per run (`capacity_withdrawn` in client.json).
- Checked first on E9Bmp_Ce8zI, direct, mobile. The longest run of guard skips fell
  from 15 GOPs to 4.

### The loss-aware controller (`adaptive-loss`)

Over lossy TCP the link capacity is not what TCP can carry: every loss halves the
congestion window. The controller now reads the sender's TCP counters (TCP_INFO: RTT,
MSS, retransmitted and sent segments; `live/backlog.py:tcp_path`), which the server
attaches to each per-frame report.

- **Loss rate:** retransmitted / sent segments over the last 10 reports, once at
  least 200 segments have been sent.
- **TCP ceiling** (Mathis et al.): MSS · 8 · 1.22 / (RTT · √p).
- **Usable capacity** = min(link estimate, share × ceiling), with share = 0.5. It
  replaces the link estimate in the controller's capacity check and hysteresis.
- With no loss, the cap is off, and the controller is exactly the old one. The
  emulated-link results are unchanged.

**Tuning used only the 8 test clips outside the evaluation set** (qZlk, QZVq, rPDr,
UkR0, vT7X, W-8c, X1y2, xxAE; results/tune-netem-*). These runs predate the guard fix,
so they are noisier than the final runs.

| Setting (mobile unless noted) | PSNR gain over adaptive, dB |
|---|---:|
| share 0.5, no stall price | +1.59 ± 1.76; repeated: +0.58 ± 2.45 |
| share 0.5, broadband | +0.47 ± 0.60 |
| share 0.35 | +0.85 ± 1.00 (mobile), +0.38 ± 0.83 (broadband) |
| share 1.0 (the raw Mathis rate), stall price 10 | −0.34 ± 1.67 |
| share 0.5, stall price 10 / 25 | +0.59 / +1.09 |

- **Share 0.5 was kept.** A smaller share was no better, and the raw Mathis rate did not
  help: TCP needs headroom over its average rate to meet a deadline.
- **A stall price was tried and dropped.** It subtracts the expected cost of a
  retransmission timeout from each rung's predicted quality; it gave no gain over the cap
  alone (`--loss-weight`, default 0).
- **Two identical runs differed by 1 dB** on mobile, which sets the noise level here.

### The VMAF-driven controller (`--quality-metric vmaf`): a negative result

The controller's quality table (configs/rate_quality.json) now also has a VMAF column
per rung (`scripts/measure_rate_quality.py`). With `--quality-metric vmaf` it maximises
predicted VMAF; its thresholds are scaled by 6 VMAF points per dB.

- **Emulated link, 18 clips** (adaptive-vmaf-loss; the loss cap is inactive there):
  30.65 dB / 62.6 VMAF against adaptive's 30.84 dB / 63.0 VMAF. Neither difference is
  significant (Δ +0.19 ± 0.33 dB, +0.4 ± 1.2 VMAF).
  - It uses 6% more bits and 2.2 points more model GPU time.
  - It moves direct from 28% of frames to 13%, and ×4 from 26% to 36%.
  - It does not reach ×2 only's VMAF (65.7). The 2.7-point gap to ×2 only stays.
- **tc netem, 10 clips:** it is behind adaptive-loss on both profiles (−0.50 and
  −0.84 dB, −2.4 and −3.5 VMAF, not significant), even on VMAF itself.
- **Why it fails:** the table's VMAF predicts a rung's quality when its frames arrive.
  The viewer's VMAF is set by held frames when they do not, and that cost is the same
  under either metric. Choosing the metric changes too little to show.

### Setup and validity

- **Runs:** 8 modes × 10 clips × 2 profiles = 160 runs. The modes are direct, x2, x4,
  adaptive, mpc, mpc-direct, adaptive-loss and adaptive-vmaf-loss. Same clips, profiles
  and rate schedule as on 2026-11-25. Results are in
  results/multiclip-netem-v2-broadband and results/multiclip-netem-v2-mobile.
- **No sudo:** the runs used `scripts/netem_unprivileged.sh`, which starts the harness
  in its own user, mount and network namespaces. The kernel TCP stack and tc netem are
  the same.
  - Spot checks against the sudo runs agreed within the run-to-run noise (1–2 dB).
- **All 160 runs are healthy:**
  - CPU at 3.5 GHz or more;
  - load below 8 before each run;
  - encoder lag below 0.5 s;
  - no client errors.
- **Withdrawals:** the client withdrew its estimate 38 times over the 80 broadband runs
  and 102 times over the 80 mobile runs.
- **Long freezes remain only where expected:**
  - No adaptive variant froze for more than 5.2 s.
  - Freezes of more than 5 s were in direct, mpc-direct, mpc and x2 runs, mostly within
    the 10 s at 500 kbit/s. A 720p rung cannot fit there.

### Results: loss-aware adaptive against every mode

adaptive-loss minus each mode, paired by clip, ± 95% CI. The count in brackets is the
clips adaptive-loss wins. Bold: the interval excludes zero.

| Profile | vs | Viewer PSNR dB | VMAF | LPIPS | Late + skipped |
|---|---|---:|---:|---:|---:|
| broadband | direct | **+1.77 ± 1.49** (8/10) | **+6.9 ± 4.8** | **−0.048 ± 0.039** | **−95 ± 86** |
| broadband | x2 | +0.80 ± 0.96 (7/10) | −0.5 ± 3.4 | −0.012 ± 0.020 | −44 ± 77 |
| broadband | x4 | **+1.01 ± 0.82** (8/10) | **+8.2 ± 3.8** | **−0.026 ± 0.018** | **+84 ± 36** |
| broadband | adaptive | **+0.42 ± 0.41** (7/10) | **+1.7 ± 1.5** | −0.011 ± 0.011 | −9 ± 28 |
| broadband | mpc | **+1.75 ± 0.99** (9/10) | **+9.2 ± 5.3** | **−0.048 ± 0.025** | −50 ± 63 |
| broadband | mpc-direct | **+3.55 ± 1.84** (9/10) | **+14.5 ± 6.9** | **−0.098 ± 0.055** | **−141 ± 89** |
| broadband | adaptive-vmaf-loss | +0.50 ± 0.63 (9/10) | +2.4 ± 3.2 | −0.015 ± 0.016 | −13 ± 38 |
| mobile | direct | **+2.04 ± 1.59** (7/10) | +5.3 ± 6.1 | −0.041 ± 0.043 | **−134 ± 87** |
| mobile | x2 | **+1.34 ± 1.24** (7/10) | −1.1 ± 4.5 | −0.004 ± 0.033 | **−136 ± 72** |
| mobile | x4 | −0.28 ± 1.07 (4/10) | +1.4 ± 4.1 | +0.006 ± 0.024 | +55 ± 61 |
| mobile | adaptive | **+0.86 ± 0.74** (7/10) | +3.2 ± 3.3 | −0.018 ± 0.023 | −48 ± 51 |
| mobile | mpc | +0.75 ± 1.30 (7/10) | +2.8 ± 3.9 | −0.006 ± 0.034 | −33 ± 65 |
| mobile | mpc-direct | **+1.98 ± 1.32** (8/10) | +3.5 ± 5.2 | −0.033 ± 0.044 | **−120 ± 64** |
| mobile | adaptive-vmaf-loss | +0.84 ± 1.45 (7/10) | +3.5 ± 6.0 | −0.016 ± 0.035 | −51 ± 81 |

adaptive-loss itself: 30.26 ± 2.57 dB / 62.3 VMAF on broadband and 27.42 ± 2.57 dB /
46.9 VMAF on mobile.

The old controller (adaptive) minus each mode, for comparison with 2026-11-25
(results/multiclip-netem-v2-*/ci.md):

| Profile | vs | Viewer PSNR dB | VMAF |
|---|---|---:|---:|
| broadband | direct | +1.35 ± 1.54 (7/10) | **+5.2 ± 4.4** |
| broadband | x2 | +0.38 ± 1.09 (5/10) | −2.2 ± 3.8 |
| broadband | x4 | +0.59 ± 0.83 (7/10) | **+6.5 ± 3.8** |
| broadband | mpc | **+1.33 ± 0.94** (9/10) | **+7.5 ± 4.5** |
| mobile | direct | +1.18 ± 1.51 (7/10) | +2.1 ± 4.2 |
| mobile | x2 | +0.48 ± 1.14 (6/10) | **−4.3 ± 3.5** |
| mobile | x4 | **−1.14 ± 0.98** (1/10) | −1.8 ± 5.0 |
| mobile | mpc | −0.11 ± 1.20 (5/10) | −0.4 ± 2.8 |

### What the re-run shows

- **The loss-aware cap is what makes the controller work over lossy TCP.** It beats the
  old controller on both profiles: +0.42 ± 0.41 dB on broadband and +0.86 ± 0.74 dB on
  mobile. On broadband it also leads in VMAF, by 1.7 ± 1.5.
- **On broadband it is the best mode.** It is significantly ahead of everything except
  ×2 only (+0.80 ± 0.96 dB, not significant), including ×4 only (+1.01 dB, +8.2 VMAF)
  and RobustMPC (+1.75 dB, +9.2 VMAF).
- **On mobile it ties ×4 only** (−0.28 ± 1.07 dB, VMAF +1.4, not significant). ×4 only
  still has fewer late and skipped frames. It is clearly ahead of 720p delivery, with or
  without MPC, and of ×2 only. Its lead over RobustMPC (+0.75 ± 1.30 dB) is not
  significant.
- **The old controller is worse than these runs first suggested.**
  - On mobile it is now 1.14 ± 0.98 dB behind ×4 only, where 2026-11-25 had a tie.
  - Its lead over RobustMPC is +1.33 dB on broadband (was +1.83) and gone on mobile
    (was +1.44).
  - The fix helped the fixed modes and MPC more than adaptive, because their larger
    streams ran into the stale-estimate lock more often. Per mode, v2 minus v1 is −0.6
    to +1.6 dB, mostly inside the noise. Direct, mobile gained most: +1.15 ± 1.13 dB.
- **What remains:**
  - On the mobile path (80 ms RTT, 1% loss), always sending the smallest stream is still
    as good as any controller tested.
  - The loss-aware cap closes the gap to ×4 only in quality, but not in late frames.

### Reproduce

```bash
# without sudo; with sudo, drop the wrapper
MODES="direct x2 x4 adaptive mpc mpc-direct adaptive-loss adaptive-vmaf-loss"
bash scripts/netem_unprivileged.sh "$PYTHON" scripts/compare_modes.py <10 eval clips> \
    --modes $MODES --half --netem --resume \
    --path delay=20,jitter=5,loss=0.1,limit=300 --out results/multiclip-netem-v2-broadband
bash scripts/netem_unprivileged.sh "$PYTHON" scripts/compare_modes.py <10 eval clips> \
    --modes $MODES --half --netem --resume \
    --path delay=40,jitter=15,loss=1,limit=300 --out results/multiclip-netem-v2-mobile
for p in broadband mobile; do
  python scripts/summarise_clips.py results/multiclip-netem-v2-$p/summary.json \
      --json results/multiclip-netem-v2-$p/ci.json --markdown results/multiclip-netem-v2-$p/ci.md
done
```
