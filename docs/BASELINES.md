# UpStreamVSR against published models

Four current super-resolution models, run with their published weights in place of
UpStreamVSR (`upstream_vsr/baselines.py`):

| Model | Venue | Kind | Scales | Trained on | Parameters |
|---|---|---|---|---|---:|
| **UpStreamVSR** (ours, final) | | per frame, CRF-conditioned, anytime exits | x2, x4 | StreamSR | **217 k** |
| [EfRLFN](https://github.com/EvgeneyBogatyrev/EfRLFN) | ICLR 2026 | single image, real-time | x2, x4 | StreamSR (the dataset authors' model) | 504 k |
| [SPAN](https://github.com/hongyuanyu/SPAN) (48 ch) | CVPR 2024 W, NTIRE 2024 ESR winner | single image, real-time | x2, x4 | DIV2K / LSDIR, clean | 426 k |
| [NanoVSR-226k](https://github.com/filippawlicki/nanovsr) | ECCV 2026 | bidirectional recurrent video | x4 | Vimeo-90K, REDS | 226 k |
| CDA-VSR | CVPR 2026 | online video, codec motion, residuals, frame types | x4 | REDS, H.264 CRF 18–28 | 3.1 M |

The re-implementations keep the authors' parameter names, so the checkpoints load
unchanged, and their outputs were checked against the authors' code: EfRLFN identical,
SPAN within 0.002 and NanoVSR within 0.0003 (on a [0, 1] scale). To run them live:

- NanoVSR's backward branch sees only the current frame, as for the last frame of each
  of its training sequences.
- CDA-VSR uses torchvision's modulated deformable convolution in place of mmcv's (same
  operation and weights). The residual map is the motion-compensation error, because the
  client has no decoder residuals. Of the two motion-vector signs, and no motion, the
  codec's own convention (frame to reference) scored best on three clips, so it is used.

## Offline: every rung, 18 test clips

`python scripts/benchmark_baselines.py --half` (results/baselines/offline.json; the UpStreamVSR
column is the final model without state, from results/no-state/offline.json). Each clip
is encoded with the live ladder (4 s), decoded with motion vectors and upscaled causally,
one frame at a time, in float16. Time per frame is on the RTX 3050 Laptop GPU with a
sync after every frame, as the live client measures it.

PSNR (dB) against the 720p reference, and median ms per frame:

| Rung | kbps | **UpStreamVSR** | bicubic | EfRLFN | SPAN | NanoVSR | CDA-VSR |
|---|---:|---:|---:|---:|---:|---:|---:|
| x2@18 | 1743 | **36.90** · 4.6 | 35.67 · 2.2 | 34.55 · 63.6 | 35.23 · 44.3 | | |
| x2@23 | 884 | **35.72** · 4.6 | 34.72 | 33.77 | 34.30 | | |
| x2@28 | 439 | **34.14** · 4.6 | 33.39 | 32.62 | 32.94 | | |
| x2@33 | 227 | **32.25** · 4.6 | 31.76 | 31.13 | 31.29 | | |
| x4@18 | 582 | **33.09** · 5.0 | 31.61 · 2.1 | 30.33 · 17.3 | 31.33 · 12.1 | 32.08 · 12.4 | 32.62 · 41.8 |
| x4@23 | 305 | **32.23** | 30.96 | 29.86 | 30.64 | 31.29 | 31.89 |
| x4@28 | 155 | **30.95** | 30.02 | 29.11 | 29.64 | 30.18 | 30.76 |
| x4@33 | 80 | **29.34** | 28.80 | 28.06 | 28.34 | 28.80 | 29.18 |

- UpStreamVSR is best at every rung, by 0.2–0.5 dB over CDA-VSR and 0.5–1.5 dB over
  bicubic. It is also the fastest learned model: 5.0 ms at x4, an eighth of CDA-VSR's
  time and 60% less than NanoVSR's. (The earlier model with recurrent state scored
  0.02–0.06 dB higher in 8.3 ms at x4; docs/ABLATION.md.)
- Only UpStreamVSR runs x2 in real time: EfRLFN (64 ms) and SPAN (44 ms) work at full
  360p resolution and exceed the 33 ms frame interval.
- CDA-VSR, the closest in quality, needs 42 ms per frame at x4, so it cannot keep up
  with 30 fps on this GPU.
- EfRLFN and SPAN score below bicubic in PSNR on these compressed streams. EfRLFN's
  authors rank by subjective quality; SPAN was trained on clean images.
- Temporal PSNR (steadiness): UpStreamVSR is steadiest at every rung, tied with EfRLFN
  at x4@33 (35.58 dB). The earlier model with recurrent state flickered more than
  bicubic and EfRLFN at x4@28–33 (34.76 vs 35.58 dB at x4@33); removing the state fixed it.

### Perceptual metrics: VMAF and LPIPS

The same run repeated with VMAF (FFmpeg libvmaf, model vmaf_v0.6.1, every frame) and
LPIPS (AlexNet, every 4th frame) added, `upstream_vsr/metrics.py`
(results/final/offline.json; its PSNR matches the table above to 0.01 dB). VMAF is out
of 100, higher is better; LPIPS is a distance, lower is better. Bold is the best value.

| Rung | **UpStreamVSR** | bicubic | EfRLFN | SPAN | NanoVSR | CDA-VSR |
|---|---:|---:|---:|---:|---:|---:|
| x2@18 | 88.7 · 0.065 | 84.0 · 0.085 | 77.7 · 0.101 | **89.9 · 0.057** | | |
| x2@23 | 83.5 · 0.089 | 78.3 · 0.106 | 72.7 · 0.121 | **83.7 · 0.080** | | |
| x2@28 | **73.6** · 0.126 | 68.3 · 0.142 | 63.7 · 0.155 | 72.8 · **0.120** | | |
| x2@33 | **59.1** · 0.176 | 54.4 · 0.190 | 50.9 · 0.202 | 57.8 · **0.171** | | |
| x4@18 | 68.9 · 0.166 | 52.9 · 0.227 | 44.6 · 0.231 | 73.4 · 0.153 | 64.1 · 0.150 | **77.0 · 0.134** |
| x4@23 | 61.6 · 0.192 | 46.2 · 0.250 | 39.4 · 0.254 | 64.4 · 0.182 | 56.5 · 0.178 | **69.4 · 0.158** |
| x4@28 | 49.9 · 0.231 | 36.5 · 0.284 | 31.5 · 0.288 | 51.2 · 0.227 | 44.8 · 0.221 | **57.7 · 0.197** |
| x4@33 | 35.9 · 0.276 | 25.6 · 0.323 | 22.3 · 0.329 | 36.6 · 0.276 | 31.7 · 0.271 | **41.7 · 0.249** |

- **UpStreamVSR does not lead on the perceptual metrics.** At x4, CDA-VSR is best on
  both metrics at every rung (VMAF +5.8 to +8.1 over UpStreamVSR), and SPAN scores 0.7–4.5
  VMAF points higher. At x2, SPAN has the lower LPIPS everywhere and the higher VMAF at
  CRF 18–23; UpStreamVSR leads VMAF at CRF 28–33.
- UpStreamVSR beats bicubic and EfRLFN on both metrics everywhere, and NanoVSR on VMAF
  everywhere (NanoVSR has 0.005–0.016 lower LPIPS at x4).
- The likely cause is the training loss: Charbonnier plus edge loss rewards pixel
  accuracy, which raises PSNR but leaves outputs smoother than the models trained for
  detail. A perceptual term (LPIPS or a feature loss) in fine-tuning is the obvious
  next step; it was not tried.
- Speed still decides the live system: CDA-VSR (41.7 ms) cannot run x4 at 30 fps, and
  SPAN (44 ms at x2, 12 ms at x4) uses 2.4× UpStreamVSR's time at x4 and cannot run x2
  live. The live comparison below and docs/COMPARISON.md score what the viewer saw,
  including frames lost to slowness.

## Live: the full pipeline, three clips

On all 18 clips, with confidence intervals and VMAF and LPIPS, see the last section of
docs/COMPARISON.md. The gains there are smaller: +0.57 to +1.00 dB viewer PSNR and
+1.9 to +11.9 VMAF, all significant, with LPIPS level against SPAN, NanoVSR and CDA-VSR.

`python scripts/compare_modes.py <3 clips> --modes adaptive adaptive-efrlfn ... x4-cdavsr --half`
(results/baselines/live). Same emulated link as docs/COMPARISON.md (2500, 500 from 10 s,
1500 from 20 s kbit/s), 30 s per clip. The controller is unchanged and times each
model, so it keeps each one to the rungs it can run within the GPU budget (0.6 of the
frame interval): no x2 for EfRLFN and SPAN, almost no x4 for CDA-VSR. Viewer PSNR counts
every frame slot; a late or skipped frame counts as the previous frame held.

| Mode | Viewer PSNR dB | Shown / late / skipped | p95 ms | Model GPU % | GPU W | kbps | Frames direct / x2 / x4 |
|---|---:|---:|---:|---:|---:|---:|---|
| **adaptive, UpStreamVSR** | **31.39** | 2449 / 58 / 193 | 163 | 13.7 | 20.5 | 640 | 631 / 1331 / 545 |
| adaptive, EfRLFN | 30.06 | 2469 / 64 / 167 | 160 | 22.2 | 37.4 | 574 | 1412 / 0 / 1121 |
| adaptive, SPAN | 30.20 | 2457 / 74 / 169 | 161 | 18.9 | 35.3 | 574 | 1416 / 0 / 1115 |
| adaptive, NanoVSR | 30.44 | 2440 / 67 / 193 | 159 | 17.1 | 32.9 | 572 | 1416 / 0 / 1091 |
| adaptive, CDA-VSR | 30.44 | 2356 / 107 / 237 | 186 | 4.1 | 17.2 | 573 | 2373 / 0 / 90 |
| **x4 only, UpStreamVSR** | **28.95** | 2610 / 22 / 68 | 157 | 19.7 | 22.2 | 392 | |
| x4 only, bicubic | 27.63 | 2610 / 22 / 68 | 156 | 8.0 | 19.6 | 392 | |
| x4 only, EfRLFN | 26.74 | 2609 / 23 / 68 | 157 | 52.7 | 58.1 | 392 | |
| x4 only, SPAN | 27.21 | 2609 / 23 / 68 | 157 | 45.2 | 56.0 | 392 | |
| x4 only, NanoVSR | 27.81 | 2609 / 23 / 68 | 157 | 41.6 | 52.2 | 392 | |
| x4 only, CDA-VSR (forced) | 14.06 | 153 / 1999 / 548 | 1001 | 101 | 68.4 | 193 | |

- **Adaptive:** with UpStreamVSR the system gains 0.95–1.3 dB over every published model
  in the same controller, at 55–62% of their GPU power (CDA-VSR, which barely runs,
  aside). Most of the gain
  comes from x2: UpStreamVSR is the only model that can run it live, and x2 carries over
  half of its frames. The others fall back to direct, which needs more bandwidth for the
  same quality.
- **CDA-VSR** is too slow for x4 on this GPU (42 ms against a 20 ms budget), so the
  controller sends it direct 94% of the time. Direct streams have larger keyframes, so
  late and skipped frames rise. Forced onto x4, it shows only 153 of 2700 frames on time.
- **x4 only** (same delivery for all): UpStreamVSR leads by 1.1 dB over NanoVSR, 1.3 dB over
  bicubic, and 1.7–2.2 dB over SPAN and EfRLFN. It uses under half their GPU time and power.

The UpStreamVSR rows are the final model without state (2026-11-18, results/final/baselines-live).
The published models' rows are from the earlier run with the same link and clips; their
controller table came from the earlier model, whose x4 predictions differ by 0.05 dB.
