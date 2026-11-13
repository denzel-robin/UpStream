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
| Model time per frame | 4.5 ms | 7.0 ms, every window |

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
