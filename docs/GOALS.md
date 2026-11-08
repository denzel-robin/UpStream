# Project goals

Recorded 2026-11-07. Update this file when goals change.

## Objective

Complete a live video upscaling pipeline that balances four things:

1. **Latency**: network delay plus upscaling time.
2. **Resources**: GPU time, memory and heat on the client.
3. **Quality**: how close the output is to the original, and how stable it is over time.
4. **Bandwidth**: bits actually sent.

## Deliverables

- A new VSR model, **UpStreamVSR**, trained on the StreamSR dataset, that combines
  the strengths of NanoVSR and CDA-VSR.
- Our own novel techniques in that model and pipeline:
  - **Anytime depth**: the model can use less depth per frame without reloading,
    so a frame at risk of missing its deadline costs less.
  - **Trade-off controller**: a live feedback loop that picks resolution, bitrate
    and model depth from network and GPU conditions.
- Both x2 and x4 input modes, producing the same output size, so the pipeline can
  switch between 360p and 180p input as bandwidth changes.
- The trained model running in the live pipeline, evaluated under realistic network
  conditions against direct delivery, bicubic and NanoVSR.

## Constraints

- **Timeline:** due in about three months (around February 2027); finish sooner if possible.
- **Training:** on Kaggle, so training must resume across sessions.
- **Client hardware:** RTX 3050 Laptop, 4 GB VRAM.
- **Live use:** no lookahead frames.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | Model, codec motion extraction, data preparation, training and benchmark scripts | Done |
| 2 | Prepare the full StreamSR data and train on Kaggle | In progress: data prepared on Kaggle; training next |
| 3 | Live integration: motion vectors from the decoder, per-frame depth, separate reconstruction and display threads | Not started |
| 4 | Trade-off controller with mid-stream resolution and bitrate switching | Not started |
| 5 | Evaluation with shaped network conditions, ablation of each technique, sustained run | Not started |
| 6 | Report and final demo | Not started |

## Success criteria

Proposed targets, to be confirmed once a trained model exists:

- **Real time:** a 5-minute sustained run at 30 fps with no collapse. NanoVSR failed
  this earlier, dropping to 20.87 fps.
- **Quality versus bandwidth:** report BD-rate against direct delivery across a full
  rate-quality curve. The earlier study found direct delivery beat NanoVSR at matched
  rates (at 360p playback), so beating direct is the real test, not beating bicubic.
- **Ablation:** each technique must show a measured benefit when removed and compared,
  or be reported honestly as not helping.
