# Progress presentation assets

> **Superseded (October 2026 progress presentation).** This folder documents the
> earlier study: off-the-shelf NanoVSR, bicubic and direct delivery over the first
> MPEG-TS pipeline (`server/`, `client/`). Its numbers do not describe the final
> system. For the final model, live pipeline and results, see the top-level
> [README](../README.md) and [docs/REPORT.md](../docs/REPORT.md).

Start with [the handoff](../logs/progress-report-handoff.md) and [launch commands](REPRODUCE.md).

- `data/comparison.csv`: concise 26-column table, one row per clip/mode/startup/buffer configuration; actual TS bytes/kbps, whole-stream PSNR/SSIM, three-run ranges and means, processing/submission rates, latency, drops and GPU allocation. Units appear in column names. No confidence intervals.
- `data/comparison-detailed.csv`: full-precision means/minima/maxima and resolution metadata for each configuration.
- `data/valid-repetitions.csv`: all 39 valid individual timing runs, with original evidence paths. `quality-all-points.csv`: all 31 offline rate-quality points. Quality and playback completeness are distinct measurements.
- `data/run-inventory.csv`: completed, excluded, verification and failed attempts; `inventory-summary.json`: missing repetitions and replacement evidence.
- `figures/`: PNG for slides and SVG for editable/vector export; three rate-quality plots, repeated-run latency/completeness, architecture, sustained trends.
- `crops/`: fixed aligned frame150 crops, including positive road/waterfall and negative flowers; labels include reference/direct/bicubic/NanoVSR. Twofold nearest enlargement is display-only.
- `demo/OFFLINE-road-comparison.mp4`: explicitly offline comparison backup; never use as live-performance evidence.
- `evidence/`: fresh verification, stability run, GPU telemetry and visible-demo snapshots. Screen captures occur separately from timed stability and original timing runs.
- `ATTRIBUTION.md`, `data/provenance.json`: source licenses, authors, URLs, hashes and preprocessing.

Rate includes MPEG-TS overhead, excludes TCP/IP; matched direct rates differ from LR by +0.03%, +1.21%, +2.18% for flowers/road/waterfall. RGB metrics are whole-stream offline results, not metrics of surviving playback frames. GUI call returns are application events, not physical display measurements.
