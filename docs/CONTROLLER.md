# Trade-off controller

Milestone 4. The live client chooses what the server sends, frame by frame, to balance
latency, client resources, quality and bandwidth. The code is `live/controller.py`; the
server side is in `live/server.py`.

## What it chooses between

The server encodes a **ladder** of rungs at once from one filter graph. A rung is a
rendition at a CRF:

| Rendition | Sent | Shown | Client work |
|---|---|---|---|
| direct | 720p | as decoded | decode only |
| x2 | 360p | UpStreamVSR x2 to 720p | decode + model |
| x4 | 180p | UpStreamVSR x4 to 720p | decode + model, more per frame than x2 |

Default ladder: direct at CRF 23, 28, 33, 38 and 43; x2 and x4 at CRF 18, 23, 28 and 33
(the CRFs the model was trained on). Only the chosen rung goes on the wire. Every rung
has a keyframe every 30 frames at the same frames, so a switch takes effect at the next
keyframe, at most 1 s later.

## The four quantities

| Quantity | Measured from | Used as |
|---|---|---|
| **Bandwidth** | Each rung's actual bitrate on this content (the server reports every rung's rate and keyframe size once a second), and the link capacity estimated by the client | A rung must fit inside 85% of capacity to be chosen (95% to stay on it) |
| **Latency** | Queuing delay of every packet | A growing queue forces a step down; a rung whose keyframe would take longer than the latency budget to cross the link is excluded |
| **Resources** | The model's measured time per scale and exit (the depth policy's running averages) | x2 or x4 is usable only if some exit fits the GPU share of the frame interval (60% by default); the deepest exit that fits sets its quality |
| **Quality** | PSNR per rung and exit from `scripts/measure_rate_quality.py` | The thing maximised |

### Estimating the link

The client knows when each packet was sent (the server stamps it) and when it arrived.

- **Capacity from keyframes.** A packet that enters an idle link takes
  `fixed + size / capacity` to arrive. Keyframes are several times larger than other
  frames, so a straight line through recent (size, delay) pairs of packets that found
  the link idle gives the capacity (slope) and the fixed delay (intercept), which
  includes propagation and any clock offset between the machines. Every keyframe is a
  free probe; no extra traffic is sent.
- **Capacity under congestion.** When the link is slower than the stream, no packet
  finds it idle, but a packet queued behind another arrives one transmission time
  after it, so bytes over arrival gaps of queued packets is the capacity (packet
  dispersion). This holds even when the stream is intermittent, for example while
  frames are being dropped; the plain delivery rate does not, and read far too low.
- **A fit expires** after 3 s without a new one, so a capacity measured before a
  change in the link is not trusted indefinitely.
- **Queuing delay** is a packet's delay less the fixed delay and its own transmission
  time. The **standing queue** is the smallest over the last 0.5 s: a keyframe burst
  raises the delay for a few frames, a link slower than the stream raises it for good.

### The decision

On every packet:

1. Drop rungs the client cannot run in time (resources) and rungs whose keyframe would
   miss the latency budget.
2. Set the bandwidth allowance:
   - **drain**, when the queue is longer than the latency budget (playout delay less
     25 ms for decoding and reconstruction): half the capacity estimate, and ask the
     server to skip the rest of the group of pictures so the queue empties before the
     switch. The screen holds the last frame meanwhile; pictures are never corrupted,
     because the next frame sent is a keyframe.
   - **capacity**, when capacity is known: 85% of it, or 95% for the current rung, so
     an estimate near the edge does not flip rungs.
   - **probe**, only when no capacity estimate exists yet: after 4 quiet seconds, the
     next better rung.
3. Among the remaining rungs within the allowance, take the best predicted quality;
   among those within 0.05 dB of it, the cheapest.
4. Step down at once. Step up only for at least 0.1 dB, with a short queue, and 2 s
   after the last step down.

Rungs are priced at their peak rate over the last 3 s, because content changes the rate
faster than a switch can follow.

### The server's backlog guard

The client learns of congestion only when delayed packets arrive, so its signal is
stuck in the same queue: in the first live run a scene change raised the direct@23
rate from about 1 to over 5 Mbit/s, and by the time the client saw it, 9 s of video
was queued. Skipping on request could not help, because the data was already sent.

The server knows at once. It tracks what it has handed to the network but not had
delivered (`live/backlog.py`: the kernel's send queue, SIOCOUTQ, or the emulated
link's queue). Before each frame, if that backlog would take longer to drain at the
client's capacity estimate than the client can wait for a frame (playout delay plus
drop-late less processing, sent by the client with the estimate in a C record), it
drops frames up to the next keyframe and sends a G record. It drops only frames that
would have arrived too late to be shown.

Two variants failed and were removed:

- **Measuring the drain rate on the server** from the backlog. The backlog is seen
  only once per frame and often empties in between, so the rate came out at a few
  percent of the link (118 kbit/s on 2500), and the guard then dropped nearly
  everything.
- **Guarding at the controller's planning budget** (playout delay less processing)
  rather than at the display deadline. A single keyframe on a slow link exceeded it,
  so the guard dropped the group after almost every keyframe (38% of frames).

### Why each rendition gets chosen

- **x4** when bandwidth is tight: the model recovers the most per bit at x4, and 180p
  costs least.
- **x2** in the middle range: better quality than x4 at the same rate once there is
  enough bandwidth for 360p at a good CRF.
- **direct** when bandwidth is plentiful (720p at a low CRF beats any upscale), when the
  GPU is busy or slow (no exit fits the GPU share), or when the client has no model.

## Measured rate and quality

`scripts/measure_rate_quality.py` on the 18 test clips (4 s each, full 720p frames,
causal, third design, 32 channels), against the 720p reference. Table in
`configs/rate_quality.json`; the controller reads it.

| Rung | kbit/s | PSNR | | Rung | kbit/s | PSNR (exit 1 / 3) |
|---|---:|---:|---|---|---:|---:|
| direct@18 | 5088 | 41.16 | | x2@18 | 1743 | 36.86 / 36.95 |
| direct@23 | 2614 | 39.53 | | x2@23 | 884 | 35.62 / 35.76 |
| direct@28 | 1340 | 37.49 | | x2@28 | 440 | 34.01 / 34.17 |
| direct@33 | 710 | 35.24 | | x2@33 | 227 | 32.11 / 32.26 |
| direct@38 | 402 | 32.78 | | x4@18 | 582 | 32.94 / 33.11 |
| direct@43 | 246 | 30.22 | | x4@23 | 305 | 32.06 / 32.28 |
| | | | | x4@28 | 155 | 30.80 / 30.96 |
| | | | | x4@33 | 80 | 29.26 / 29.36 |

BD-rate against direct delivery (negative: fewer bits for the same PSNR), mean of the
per-clip values:

| | UpStreamVSR, deepest exit | UpStreamVSR, first exit | Bicubic |
|---|---:|---:|---:|
| x2 | **-4.4%** (18 clips) | -0.1% | +18.7% |
| x4 | **-31.3%** (16 clips) | -28.4% | -21.4% (12 clips) |

Clips are left out where the curves do not overlap in PSNR. On the mean curves: x2
-10.0%, x4 -29.6%, bicubic x2 +9.7%, bicubic x4 +7.6%.

What this means for the ladder:

- **Below about 1 Mbit/s an upscaled rung wins** at every rate: x4@33 at 80 kbit/s,
  x4@28, then x2@33 (32.26 dB at 227 kbit/s, against direct@43's 30.22 at 246), x2@28
  (34.17 at 440 against direct@38's 32.78 at 402) and x2@23.
- **Above about 1 Mbit/s direct wins**: direct@28 (37.49 at 1340) beats x2@18 (36.95
  at 1743). x2 and x4 are capped by what 360p and 180p can hold.
- The model is what makes x2 worth having: bicubic x2 needs 19% more bits than direct,
  UpStreamVSR x2 4% fewer. At x4 the model adds 10 points of BD-rate over bicubic.
- x2@18 and x4@18 are never on the efficient frontier; the ladder could drop them.

These are averages: per clip, x2's BD-rate ranges from about -77% to +106%, so which
rendition wins depends on the content. The live R records price rungs on the actual
content; the quality side still uses the table's means.

## Results

### Simulated link (`live/simulate.py`, measured table, no server guard)

| Scenario | What the controller did | Late | Skipped | Max delay |
|---|---|---:|---:|---:|
| 3000, 700 at 20 s, 1800 at 40 s | direct@28; at the drop drained to x2@33 then x2@28; back to direct@28 | 9 | 21 | 429 ms |
| 400, 150 at 20 s, 1000 at 40 s | x2@33; x4@33 at 150; x2@23 at 1000 | 7 | 23 | 259 ms |
| 800, GPU 5x slower at 20 s | x2@28; direct@38 when no exit fits the GPU share | 0 | 0 | 67 ms |

Late and skipped are out of 1,800 frames (60 s). Without the GOP skip, the first
scenario reached 2.9 s of delay and 182 late frames.

### Live, one machine (2026-11-11)

RTX 3050 Laptop, float16, 30 s of a test clip, full ladder (13 rungs encoded at once,
no late releases), emulated link 2500, 500 at 10 s, 1500 at 20 s, playout delay 150 ms.

| | |
|---|---:|
| Frames shown / dropped late / skipped | 762 / 12 / 138 of 900 |
| Latency p50 / p95 / max | 150 / 156 / 430 ms |
| Mean payload | 456 kbit/s |
| Rungs used | direct@23, direct@28 on 2500; x4@28, x4@33 on 500; direct@38, x2@23, x2@28 on 1500 |

This clip is hard: a scene change at 9 s makes direct@23 frames 25-75 KB (over
5 Mbit/s) and keyframes stay large, so most skips are at the scene change and the link
drop together. A keyframe of 75 KB takes 1.2 s on 500 kbit/s and cannot be split; that
is the worst case of switching on keyframes.

## Limits

- One machine, emulated link without propagation delay or loss; tc netem in milestone 5.
- Quality is predicted from the table's means, not measured live (the client has no
  reference). Milestone 5 measures it from recorded output.
- The keyframe interval (1 s) bounds how fast a switch takes effect; a shorter one
  would change the training distribution.

## Testing it without a network

`live/simulate.py` runs the controller against a simulated first-in first-out link with
a rate schedule and an optional GPU load schedule, using the rate-quality table, with no
video or GPU. `--link` on the server puts the same kind of bottleneck
(`live/link.py`) in front of the real stream. Real network shaping (tc netem, with
propagation delay and loss) is milestone 5.

```bash
python scripts/measure_rate_quality.py --checkpoint runs/kaggle-6-v3/run/best-calibrated.pt
python live/simulate.py --link 0:3000,20:700,40:1800 --seconds 60
python live/server.py data/test-clips/NRqG3AEOVGg@1440.mp4 --seconds 60 --start x4@28 \
    --crfs 18 23 28 33 --direct-crfs 23 28 33 38 43 --link 0:3000,20:700,40:1800
python live/client.py --checkpoint runs/kaggle-6-v3/run/best-calibrated.pt --controller \
    --playout-delay 0.15 --half --report results/controller.json
```
