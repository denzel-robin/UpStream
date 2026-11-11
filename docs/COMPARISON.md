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
