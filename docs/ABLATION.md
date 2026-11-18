# Ablation: what each part of the live system contributes

Each run is the full adaptive pipeline with one part taken away, over the same emulated
link (2500 kbit/s, 500 from 10 s, 1500 from 20 s), three 30 s clips, playout delay
150 ms, float16. Viewer PSNR counts every frame slot against the 720p reference, a late
or skipped frame scored as the held previous frame (as in docs/COMPARISON.md).

```
python scripts/compare_modes.py <3 clips> --modes adaptive adaptive-bicubic adaptive-no-guard \
    --half --out results/ablation-idle
python scripts/compare_modes.py <3 clips> --modes adaptive adaptive-fixed-depth adaptive-no-gpu-awareness \
    --half --gpu-load 8192,200,5 --out results/ablation-loaded
```

| Removed | Switch | GPU |
|---|---|---|
| UpStreamVSR (bicubic upscales the x2/x4 frames; rung choices unchanged) | client `--upscaler bicubic` | idle |
| Sender backlog guard | server `--no-guard` | idle |
| Anytime exits (always the deepest exit) | client `--fixed-exit -1` | loaded |
| GPU awareness in the controller (every rung credited with its deepest exit) | client `--no-gpu-awareness` | loaded |

"Loaded" runs `scripts/gpu_load.py` (8192² float16 matrix products, 200 ms busy, 5 ms
idle) for the whole stream, as another application on the viewer's GPU would. It holds
the GPU at 98% utilisation and slows the deepest x4 exit from 7.9 ms to about 21 ms.

## Results (sums or means over the three clips)

| Variant | Viewer PSNR dB | Shown / late / skipped | Latency p95 ms | Model GPU % | GPU power W |
|---|---:|---:|---:|---:|---:|
| **adaptive, idle GPU** | **31.51** | 2473 / 60 / 167 | 164 | 15.6 | 21.1 |
| bicubic in place of the model | 30.73 (−0.78) | 2472 / 61 / 167 | 163 | 6.3 | 18.0 |
| no backlog guard | 30.62 (−0.89) | 2353 / 134 / 213 | 326 | 15.0 | 20.7 |
| **adaptive, loaded GPU** | **31.46** | 2454 / 80 / 166 | 178 | 36.6 | 74.1 |
| fixed depth (deepest exit) | 31.35 (−0.11) | 2432 / 99 / 169 | 188 | 37.1 | 74.2 |
| no GPU awareness | 31.56 (+0.10) | 2455 / 73 / 172 | 180 | 37.7 | 73.8 |

## What each result says

- **The model.** With delivery identical (the same frames shown, late and skipped),
  replacing UpStreamVSR with bicubic loses 0.78 dB over all frames. About 70% of the frames
  are x2/x4, so the model adds about 1.1 dB on each upscaled frame, for 9 percentage
  points of GPU time and 3 W. This is the model's live contribution, separate from the
  controller's.
- **The backlog guard** is the largest single contribution: without it, late frames more
  than double (60 → 134), p95 latency doubles (164 → 326 ms), the longest freeze
  grows, and viewer PSNR falls 0.89 dB. Frames queued behind a link drop arrive too late to
  show, and the screen holds stale frames longer than the guard's deliberate skip.
- **Anytime exits** help a little under load: 19 fewer late frames (80 vs 99) and
  +0.11 dB. The policy took a shallow exit on about 6% of upscaled frames, when the time
  left before the deadline was short. The benefit is small on this GPU because even the
  deepest exit usually fits; it grows on slower hardware.
- **GPU awareness** made no measurable difference here (+0.10 dB without it, within
  run-to-run noise): even with the GPU saturated by another process, the deepest x2 exit
  and the shallow x4 exit fit the 20 ms model budget, so the controller never had a
  rung to exclude. It is a safeguard for weaker GPUs. On those it moves the stream to direct
  rungs instead of running a model that cannot keep up; this is covered by
  `tests/test_controller.py::test_controller_leaves_the_model_when_the_gpu_is_too_slow`
  but not shown on this laptop.

## The recurrent state (2026-11-18)

The same recipe as the final model (32 channels, flicker loss 0.05, 70k iterations,
the same data; `kaggle/2b_train_no_state.ipynb`, Kaggle run upstream-7-no-state) with
the state cleared before every frame, in training and at inference. Memory is used at
x4 only, so the x2 rows compare two models that differ only by training run: their
gap (0.03–0.06 dB) is the noise floor between runs.

Offline, 18 test clips, every rung (`python scripts/benchmark_baselines.py --half
--models upstream no-state bicubic`, results/no-state/offline.json). PSNR dB /
temporal PSNR dB (higher is steadier) / median ms per frame on the RTX 3050 Laptop:

| Rung | With state (final) | No state | Bicubic |
|---|---|---|---|
| x2@18 | 36.97 / 41.60 / 4.6 | 36.90 / 41.60 / 4.6 | 35.67 / 40.96 |
| x2@33 | 32.28 / 37.48 / 4.6 | 32.25 / 37.47 / 4.6 | 31.76 / 37.31 |
| x4@18 | 33.11 / 38.69 / 8.2 | 33.09 / **38.83** / **5.0** | 31.61 / 38.08 |
| x4@23 | 32.27 / 37.57 / 8.3 | 32.23 / **37.92** / **5.0** | 30.96 / 37.37 |
| x4@28 | 31.00 / 36.36 / 8.2 | 30.95 / **36.80** / **5.0** | 30.02 / 36.50 |
| x4@33 | 29.39 / 34.76 / 8.3 | 29.34 / **35.58** / **5.0** | 28.80 / 35.51 |

Kaggle's own validation and test windows (12 frames) agree: at x4 the state adds
0.04 dB (test 29.133 vs 29.095) and the model without it is steadier (35.52 vs 35.29).

- **The state does not earn its cost.** At x4 it adds 0.02–0.05 dB, below the noise
  floor between training runs. Per clip it ranges from −0.39 to +0.21 dB at x4@18.
- **It costs 65% more x4 time** (8.2 vs 5.0 ms): warping, motion refinement and fusion.
- **It is the source of the flicker.** Without it, the model is steadier at every x4
  rung, and at x4@28–33 steadier than bicubic too. That removes the one weakness found
  against the published models (docs/BASELINES.md).

Results: `results/ablation-idle/`, `results/ablation-loaded/` (summary.json, per-clip
reports), `results/no-state/`.
