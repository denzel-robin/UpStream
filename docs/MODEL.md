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

## Open risks

1. **Recurrent state is not yet shown to help.** In the test above it scored about
   0.9 dB below no state. An overfitting test favours memorizing single frames, so
   this is not conclusive, but the temporal path has to beat no-state on held-out
   video or the design needs to change. This is the first thing to check after training.
2. **Held-out quality is unknown.** The only held-out run predates the head fix and
   stayed below bicubic.
3. **Direct delivery may still win.** The earlier study found it beat NanoVSR at matched
   rates at 360p playback. The real test is rate-quality against direct delivery at 720p.
4. **BatchNorm:** eval-mode scores sit about 1 dB below train-mode in this tiny test.
   Recheck at full batch sizes.

## Training in the cloud

Dataset links are in the [EfRLFN repository](https://github.com/EvgeneyBogatyrev/EfRLFN)
(`youtube-train.tar.gz`, 12 GB; `subjective-set.tar.gz`, 437 MB). The paper describes
the videos as CC BY 4.0; do not commit them here. The commands below have only been
run on a seven-clip sample, not the full archive.

```bash
pip install -r requirements-train.txt
tar xzf youtube-train.tar.gz && tar xzf subjective-set.tar.gz
python scripts/prepare_streamsr.py youtube-GPT-queries data/streamsr
python scripts/prepare_streamsr.py subjective-set data/streamsr --split test
python scripts/train_upstream.py data/streamsr runs/main
```

Rerunning the same `train_upstream.py` command resumes from `runs/main/latest.pt`, and
`prepare_streamsr.py` skips clips already listed in its manifest. Keep `runs/` on
persistent storage, because Colab and Kaggle sessions end without warning.

## Tests

```bash
python -m pytest tests
```
