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

Per-scale BatchNorm recalibration (`kaggle/3_calibrate.ipynb`, 800 training windows
per scale, weights unchanged). `latest.pt` (iteration 100,000) after recalibration is
now the best checkpoint on val, so it is the one to use.

| Method | val x2 | val x4 | test x2 | test x4 |
|---|---:|---:|---:|---:|
| Bicubic | 30.61 | 27.13 | 30.63 | 27.34 |
| best.pt before | 31.27 | 27.84 | 31.78 | 28.37 |
| latest.pt before | 31.50 | 27.28 | 32.06 | 27.56 |
| latest.pt after, 3 blocks | 31.80 | 28.19 | 32.31 | 28.72 |
| latest.pt after, 6 blocks | 31.77 | 28.21 | 32.28 | 28.75 |
| latest.pt after, 9 blocks | **31.88** | **28.29** | **32.38** | **28.84** |
| latest.pt after, 9 blocks, state reset | 31.87 | 28.28 | 32.38 | 28.83 |

- Recalibration added 0.6 dB (x2) and 0.5 dB (x4) on test over the earlier best, with
  no retraining. On test the model now beats bicubic by 1.75 dB at x2 and 1.50 dB at x4.
- The shared statistics had left `latest.pt` tuned for x2: before recalibration it was
  only 0.2 dB above bicubic at x4.
- Depth range is now smaller: 3 blocks reaches about 95% of the 9-block gain, and
  6 blocks is no better than 3. The anytime exits currently buy speed, not quality.
- Recurrent state still adds at most 0.01 dB.

Speed of the trained, fused model on the RTX 3050 Laptop: same as the table above
(3.5 / 4.8 / 6.0 ms at 180p, 3.7 / 4.9 / 6.2 ms at 360p). Deployed with one fused trunk
per scale, 202k parameters in total, of which each frame uses half.

## Second design (2026-11-08)

Changes after the first run, where the gate closed and depth gave little:

- **Temporal fusion:** current features, the warped hidden state and the warped previous
  output frame (720p, moved by the motion vectors scaled to output pixels, then
  space-to-depth onto the grid) are mixed by one 1x1 conv, instead of adding raw hidden
  features. The previous output is at output resolution, so it survives a scale switch.
- **Gate starts open** (sigmoid bias +2, about 0.88), so it has to learn when to close.
- **All exits scored on every frame**, instead of one random exit per frame. The state
  passed on still comes from a random exit, so it works whichever exit runs live.
- **Per-scale BatchNorm from the start**, plus recalibration at the end of training.
- **Ablation run** (`kaggle/2b_train_no_state.ipynb`): the same recipe with the state reset
  on every frame.

Cost: training is about 0.56x as fast (laptop measurement), so the run is 70,000 iterations.
Inference adds one warp at output resolution and a 1x1 conv; benchmark again after training.

Second-design results (`upstream-4-train`, 70,000 iterations, 7.4 h on a T4 at 2.76 it/s,
then per-scale recalibration; `latest`). First design shown for comparison (recalibrated
`latest` from the first run, 100,000 iterations).

| Test PSNR | x2 3 blocks | x2 9 blocks | x4 3 blocks | x4 9 blocks | x4 9 blocks, state reset |
|---|---:|---:|---:|---:|---:|
| Bicubic | 30.63 | 30.63 | 27.34 | 27.34 | 27.34 |
| First design | 32.31 | 32.38 | 28.72 | 28.84 | 28.83 |
| Second design | 32.23 | 32.36 | 28.67 | 28.85 | 28.70 |

| Val PSNR | x2 3 blocks | x2 9 blocks | x4 3 blocks | x4 9 blocks |
|---|---:|---:|---:|---:|
| First design | 31.80 | 31.88 | 28.19 | 28.29 |
| Second design | 31.70 | 31.85 | 28.15 | 28.31 |

- Peak quality is unchanged (within 0.03 dB).
- The temporal path now does something: resetting state costs 0.15 dB at x4 (was 0.01).
  At x2 it is still only 0.01 dB. These are the same weights with state reset, not a
  separately trained model; the proper ablation is postponed.
- Depth range grew a little: 9 blocks over 3 is 0.13 dB at x2 and 0.18 dB at x4 on test.
- Training is now stable: x2 and x4 validation move together (correlation +0.93, was -0.83).
- Cost on the RTX 3050 Laptop, fused, synthetic input with zero motion: 5.4 / 6.6 / 7.9 ms
  at 180p and 5.5 / 6.8 / 8.0 ms at 360p for 3 / 6 / 9 blocks, about 1.9 ms more than the
  first design (the extra warp at output resolution). Peak 123 to 127 MiB.

Why the state helps so little (`scripts/diagnose_temporal.py`, second design, the 64 test
windows rebuilt locally, P-frames only):

| Measure | x2 | x4 |
|---|---:|---:|
| Mean abs error to previous frame, no motion | 0.032 | 0.031 |
| ... warped with codec MVs | **0.018** | **0.017** |
| ... warped with MVs negated | 0.042 | 0.040 |
| Input pixels identical to the previous frame | 48% | 48% |
| GT frame t-1 warped onto t by MVs (PSNR) | 36.0 | 35.0 |
| Previous model output warped onto t (PSNR) | 28.5 | 26.4 |
| Current frame alone, no state (PSNR) | 31.6 | 28.3 |
| Mean gate value | 0.87 | 0.88 |
| PSNR gain from state, frames 1 to 15 | 0.00 to 0.09 | 0.16 to 0.27 |

- Motion vectors are correct: warping halves the error, and negating them makes it worse.
- The gate stays open, so the model does use the state.
- Half of every frame is copied unchanged from the previous one (skipped blocks), so there
  is no new information there to accumulate.
- The warped previous output is 3 dB worse than processing the current frame alone. Block
  motion vectors (one per 4x4 input pixels) and bilinear resampling misalign and blur it, so
  the model can only take small corrections from it.
- The gain appears at frame 1 and does not grow, so longer windows would not help.

Reusing unchanged regions (`scripts/simulate_skip.py --clips`, whole frames of the 18 test
clips, 6 s each at CRF 23 and 33, second design, 9 blocks). A region is reused from the
previous output only on P-frames where the input pixels and the motion vectors under it,
including an 11-grid-pixel halo (the receptive field), are unchanged.

| | x2 | x4 |
|---|---:|---:|
| Grid pixels changed | 49% | 48% |
| Grid pixels whose output can change (changed, dilated by the halo) | 75% | 69% |
| 16-px tiles to recompute | 81% | 76% |
| Work left with each tile's halo (upper bound) | 96% | 95% |
| PSNR, full / reused | 32.75 / 32.74 | 29.71 / 29.70 |

- Reuse does not hurt quality (output within 80 to 90 dB of full recomputation).
- On average it saves little: the receptive field spreads every change by 11 grid
  pixels, so at best about 25 to 30% of work could be skipped with ideal sparse compute,
  and about 5% with tiles on a GPU.
- It is strongly content-dependent. Changed area per P-frame ranges from 2% to 88% by
  clip (mean 46%); 11% of P-frames change under 10%, 4% under 1%. Mostly static streams
  would save nearly everything.

## Colour mismatch in the Kaggle data (2026-11-09)

**All results above were measured on training, validation and test inputs whose colours
were wrong.** They are not valid for the live pipeline.

`prepare_streamsr.py` scales the source to 720p (tagged BT.709) and then downscales that
again for the x2 and x4 inputs. Kaggle's ffmpeg does not carry the BT.709 matrix into the
second scale and assumes BT.601, so the inputs' colours and brightness are shifted while
the ground truth is correct. The local ffmpeg (9.0.2) carries the matrix forward.

How it was found and confirmed:

- On test windows rebuilt locally, the second design is 0.13 dB *below* bicubic at x2
  (31.73 against 31.86) and only 0.63 dB above it at x4 (28.61 against 27.98). The same
  code on Kaggle's copy of the test set gives +1.72 and +1.51 dB.
- Ground truth is bit-identical between the two copies; only the inputs differ. The
  Kaggle inputs are darker by 1 to 13 levels depending on content, and luma PSNR of the
  input against the downscaled ground truth is 32.0 dB on Kaggle against 35.7 dB locally.
- Rebuilding one test clip locally with `in_color_matrix=bt601` forced on the second
  scale reproduces Kaggle's inputs **bit for bit**.

So the models learned partly to undo a colour error that live inputs do not have. On
correct inputs that correction is wrong, which is why the gain over bicubic shrinks or
reverses.

Fix: the second scale now states `in_color_matrix=bt709:in_range=tv` explicitly, which
gives bit-identical windows with the local ffmpeg. `check_colour()` runs a saturated
solid colour through the real filter graph before any data is prepared and stops if the
renditions disagree by more than 5 levels (a correct graph gives 2 locally and 3 on
Kaggle, from chroma quantisation; the old behaviour gives 17). The data has to be rebuilt and the model
retrained.

## Third design (2026-11-09)

Changes, all from the measurements above:

| Change | Why | Cost |
|---|---|---|
| **Motion-vector refinement**: a small net (conv 3x3 70 to 16, conv 3x3 16 to 2, tanh) sees current and warped features and pixels and corrects the codec vector by up to 2 grid pixels per grid pixel; the 720p flow is upsampled bilinearly instead of in 4x4 steps. Zero-initialised | The warped previous output was 3 dB worse than the current frame alone | about 10k parameters |
| **Memory only at x4** (`memory_scales`). x2 frames still pass state on, so a switch to x4 can use it | State gave 0.01 dB at x2 for 1.9 ms | saves time at x2 |
| **CRF conditioning**: the stream's CRF, which the server knows, sets a per-channel scale and shift of the entry features (FiLM). Zero-initialised | Tells the model how much compression to undo, which matters most at the low bitrates where x2 and x4 have to beat direct 720p | about 1k parameters |
| **Flicker loss** (`--temporal-weight 0.05`): Charbonnier on the change between consecutive outputs against the change between consecutive ground-truth frames | With state, the second design flickered slightly more than with state reset (below) | none at inference |
| **Width test**: a second run with 48 channels and exits at 2/4/6 | Depth added only 0.13 to 0.18 dB, so width may be the limit | slower, see below |

Sampling positions in `warp` are always float32, so half-precision inference does not
misplace samples (in float16 a position near the right edge of 1280 pixels is only
accurate to about 0.3 pixels).

Flicker, second design on the locally rebuilt test windows (PSNR of frame-to-frame
change against the ground truth's change; higher is steadier):

| | Bicubic | 9 blocks | 9 blocks, state reset |
|---|---:|---:|---:|
| x2 | 37.32 | 37.51 | 37.61 |
| x4 | 35.00 | 35.12 | 35.44 |

Cost on the RTX 3050 Laptop (untrained weights, fused, synthetic input with zero motion;
ms per frame for the three exits):

| | x2 | x4 | Deployed parameters | Peak MiB |
|---|---|---|---:|---:|
| Second design | 5.5 / 6.8 / 8.0 | 5.4 / 6.6 / 7.9 | 205,601 | 127 |
| Third, 32 channels, exits 3/6/9 | 3.3 / 4.5 / 5.8 | 6.7 / 7.9 / 9.1 | 217,107 | 148 |
| Third, 48 channels, exits 2/4/6 | 5.4 / 7.9 / 10.3 | 9.4 / 11.9 / 14.3 | 328,451 | 182 |

Training notebooks: `kaggle/4_train_v3.ipynb` and `kaggle/4b_train_v3_wide.ipynb`, after
the data is rebuilt with notebook 1.

## Open risks

1. **Recurrent state does not help.** After full training the gate is almost closed.
   Possible causes: raw deep features are added straight onto entry features, motion
   vectors cover 16-pixel cells, and the shared BatchNorm problem above. The design of the
   temporal path needs to change, or the report has to say it gives no benefit.
2. **Depth gives a small range** (about 0.1 dB after recalibration). Three blocks is
   nearly as good as nine, so the controller can always afford good quality, but the
   anytime claim is weak unless deeper exits gain more after the next training run.
3. **Direct delivery may still win.** The earlier study found it beat NanoVSR at matched
   rates at 360p playback. The real test is rate-quality against direct delivery at 720p.
4. **Per-scale BatchNorm during training is untested.** Recalibration confirmed the
   cause; the next training run will be the first with per-scale BatchNorm from the start.

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
