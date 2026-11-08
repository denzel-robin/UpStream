# UpStreamVSR

A causal video super-resolution model for live streaming. It combines ideas from
NanoVSR and CDA-VSR and adds anytime depth and resolution-switchable state.
Code: `upstream_vsr/model.py`.

**Status: not yet trained on the full dataset. No quality claim can be made yet.**

## What comes from where

| Part | Source | Why |
|---|---|---|
| Reparameterizable blocks that fuse to plain 3x3 convs | NanoVSR | Fast at inference, no custom operators |
| Additive state propagation | NanoVSR | Cheaper than concatenating features |
| Warping recurrent state with codec motion vectors | CDA-VSR | Motion is already in the bitstream, so no flow network is needed |
| Gating temporal fusion by residual-like error | CDA-VSR | Limits ghosting where motion compensation fails |
| Frame-type input | CDA-VSR | I-frames carry no motion and reset compression artifacts |
| Pixel-shuffle head at the low-resolution grid | CDA-VSR | Far cheaper than convolving at high resolution |
| Anytime depth | This project | Depth can drop per frame to meet a deadline |
| Shared grid for x2 and x4 | This project | Similar cost in both modes; state survives a resolution switch |

## What was left out, and why

- **NanoVSR's backward pass:** it needs future frames, which adds latency.
- **CDA-VSR's deformable convolution:** it needs a custom operator (mmcv).
- **CDA-VSR's true codec residual:** FFmpeg does not export it without a patched
  decoder. The model uses `|frame - warped previous frame|` on the grid instead.
  This is an approximation, and it also reacts to bad motion vectors and occlusions.

## How one frame is processed

1. Map the input onto the grid (output size / 4): x4 uses the frame directly, x2
   pixel-unshuffles it.
2. Warp the previous hidden state and previous grid image with codec motion vectors.
3. Compute a gate from the warping error and the I-frame flag.
4. Compute `features + gate * warped_hidden`.
5. Run the blocks up to the chosen exit.
6. Add the pixel-shuffled head output to a bilinear upscale of the input.

The head starts at zero, so an untrained model outputs exactly bilinear. With random
weights, early training was spent unlearning the noise the head added.

## Training data

StreamSR provides H.264 clips as `<id>@360.mp4`, `@720.mp4` and `@1440.mp4`.
`scripts/prepare_streamsr.py` uses only `@1440` as source material:

- Target: downscaled to 720p.
- Inputs: re-encoded to 180p and 360p with the live pipeline's x264 settings and a
  random CRF from 18, 23, 28 or 33.
- Motion vectors: read from those re-encodes.
- Filtering: non-16:9 clips, clips under 23 fps, and blank or frozen windows are dropped.

This differs from the StreamSR paper, which pairs YouTube's native renditions. Our
re-encode matches the live stream's artifacts and its I/P-only structure, while
YouTube's renditions contain B-frames. The cost is that the model never trains on
YouTube's own compression. The 720p target also inherits the source's compression.

## Measurements so far

Forward time on an RTX 3050 Laptop GPU, measured with `scripts/benchmark_model.py`.
This covers untrained, fused weights on synthetic input with zero motion. It excludes
decoding, motion vector extraction, GPU transfer and display, so it is not end-to-end latency.

| Model | Exit | Input | Mean ms/frame | Peak MiB |
|---|---|---|---:|---:|
| UpStreamVSR (119k params) | 3 blocks | 180p | 3.5 | 80 |
| UpStreamVSR | 6 blocks | 180p | 4.7 | 80 |
| UpStreamVSR | 9 blocks | 180p | 5.9 | 80 |
| UpStreamVSR | 3 blocks | 360p | 3.7 | 83 |
| UpStreamVSR | 6 blocks | 360p | 4.9 | 83 |
| UpStreamVSR | 9 blocks | 360p | 6.1 | 83 |
| NanoVSR-226k, fused, 3-frame chunks | 8 blocks | 180p | 20.4 | |

Overfitting test: x4, four windows from two clips, 300 steps. This only shows that
the training path works. It says nothing about generalization.

| Variant | PSNR on those frames, eval mode |
|---|---:|
| Bilinear | 26.05 |
| Bicubic | 26.69 |
| No recurrent state, deepest exit | 28.77 |
| Recurrent state, deepest exit | 27.84 |
| Recurrent state, trained on random exits | 28.19 |

Held-out test: 3,000 iterations on 16 windows from 5 clips, scored on 3 windows from
1 unseen clip. This is far too little data to judge final quality. It only shows that
the model can beat bicubic on video it has not seen.

| Method | x2 PSNR | x4 PSNR |
|---|---:|---:|
| Bilinear | 24.17 | 21.55 |
| Bicubic | 24.96 | 21.84 |
| 3 blocks | 25.29 | 22.01 |
| 6 blocks | 25.25 | 21.90 |
| 9 blocks | 25.20 | 21.94 |
| 9 blocks, state reset every frame | 25.18 | 22.00 |

First full Kaggle run (2026-11-08): 1,940 training windows from about 1,230 clips,
100,000 iterations at batch 8 on a T4, about 4.1 it/s. `best.pt` is from iteration
68,000, chosen on val. Val and test clips are never trained on; test is the 18-clip
StreamSR subjective set. Model PSNR is RGB against the 720p ground truth.

| Method | val x2 | val x4 | test x2 | test x4 |
|---|---:|---:|---:|---:|
| Bilinear | 30.14 | 26.75 | 30.25 | 26.99 |
| Bicubic | 30.61 | 27.13 | 30.63 | 27.34 |
| 3 blocks | 31.20 | 27.73 | 31.59 | 28.19 |
| 6 blocks | 31.21 | 27.78 | 31.65 | 28.26 |
| 9 blocks | 31.27 | 27.84 | 31.78 | 28.37 |
| 9 blocks, state reset every frame | 31.27 | 27.83 | 31.77 | 28.36 |

Findings:

- On test it beats bicubic by 1.15 dB at x2 and 1.02 dB at x4.
- Depth helps a little: 9 blocks is 0.11 to 0.19 dB above 3 blocks. 3 blocks keeps
  about 85% of the gain at about 60% of the time.
- Recurrent state adds nothing (at most 0.01 dB). The learned gate passes only 2 to 6%
  of the warped state even when the warp is perfect, so the model has learned to ignore it.
- Validation scores swing by up to 2.9 dB between checks, and x2 and x4 move in
  opposite directions (correlation -0.83 over 50 checks). Cause: all blocks shared one
  set of BatchNorm statistics, which track whichever scale the last batches used, since
  they update once per frame. The checkpoint picked as best is partly luck. Fixed by giving
  each scale its own BatchNorm (convolutions stay shared) and by
  `scripts/calibrate_bn.py`, which recomputes per-scale statistics for existing checkpoints.

## Open risks

1. **Recurrent state does not help.** After full training the gate is almost closed.
   Possible causes: raw deep features are added straight onto entry features, motion
   vectors cover 16-pixel cells, and the shared BatchNorm problem above. The design of the
   temporal path needs to change, or the report has to say it gives no benefit.
2. **Depth gives a small range** (0.1 to 0.2 dB). That is enough for the controller to
   trade speed for quality, but only just.
3. **Direct delivery may still win.** The earlier study found it beat NanoVSR at matched
   rates at 360p playback. The real test is rate-quality against direct delivery at 720p.
4. **Per-scale BatchNorm is untested on real data.** Recalibration should reach the
   upper values seen during training (val x2 about 31.8, x4 about 28.3) for both scales at
   once; `kaggle/3_calibrate.ipynb` checks this.

## Training on Kaggle

The notebooks in `kaggle/` call `scripts/kaggle_run.py`. Upload them to Kaggle and
run each with **Save Version > Save & Run All**, because an interactive session loses its
output when it stops. Internet must be on, which requires a phone-verified Kaggle account.

1. **`kaggle/1_prepare_data.ipynb`**, Accelerator **None**. Downloads StreamSR (links from
   the [EfRLFN repository](https://github.com/EvgeneyBogatyrev/EfRLFN)), processes up to
   2,000 clips and writes `streamsr-windows.tar`. It stops by itself after 8 hours and tars
   whatever is done.
2. **`kaggle/2_train.ipynb`**, Accelerator **GPU T4 x2**, with notebook 1's output attached
   under **Add Input**. It trains for up to 8 hours, then writes `run/` with checkpoints
   and `evaluation.json`. To continue, also attach its previous output and run it again.
3. **`kaggle/3_calibrate.ipynb`**, Accelerator **GPU T4 x2**, with the outputs of notebooks
   1 and 2 attached. Recomputes per-scale BatchNorm statistics and scores each checkpoint
   before and after.

Tested locally with stand-in archives and a fake Kaggle directory, including resuming and
stopping at the time limit. Notebooks 1 and 2 have run successfully on Kaggle. The
paper describes the videos as CC BY 4.0; do not commit them here.

To train elsewhere, use `scripts/prepare_streamsr.py` and `scripts/train_upstream.py`
directly. Score a checkpoint with `scripts/evaluate_upstream.py`.

## Tests

```bash
python -m pytest tests
```
