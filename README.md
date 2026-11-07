# Video Resolution Enhancement for Streaming Applications

The objective is a complete streaming pipeline with practical trade-offs between
bandwidth, visual quality, latency, and client compute. Neither 320×180→1280×720
nor x4 scaling is a final requirement. Earlier x4 evaluations are model-specific
baselines. Resolution, compression, playback size, and reconstruction should be
compared jointly. NanoVSR remains a native x4 model; CDA authentic-input work is
pending and is outside the current milestone.

The current milestone implements paced, video-only localhost delivery:
source → live FFmpeg H.264 encoding → TCP/MPEG-TS → live FFmpeg decoding →
optional reconstruction/resizing → timed OpenCV playback. See
[the measured milestone report](logs/streaming-milestone-2026-10-15.md).

From this project directory, in kitty:

```bash
.venv/bin/python scripts/run_streaming_demo.py --mode bicubic --output results/my-playback
```

The output directory must be new. Change `--mode` to `direct` or `nanovsr` for
the other comparisons. NanoVSR requires working CUDA access. The existing
`.venv` suffices; do not reinstall dependencies. Escape cancels playback and
records an explicit cancellation error. Runs end automatically after the clip.

Use `--headless` for transport/processing checks without opening a window.
Use `--config configs/streaming-tail.json` for 91 frames including a partial
NanoVSR chunk. Copy/edit `configs/streaming-smoke.json` for transmission size,
common playback size, CRF, encoder preset, chunk size, queue capacity, playout
delay, and late-frame deadline. The included source is **synthetic**.

For independent terminals (same host and configuration):

```bash
.venv/bin/python server/streaming_server.py --mode bicubic --report results/manual-server.json
.venv/bin/python client/streaming_client.py --mode bicubic --report results/manual-client.json
```

Both bind/connect only to 127.0.0.1:8765 by default; choose matching `--port`
values to avoid conflicts. This is a trusted-localhost research demo, not a
remote streaming service. It has no audio, adaptive bitrate, seeking, reconnect,
TLS, or cross-host clock synchronization. Server and client require matching
configs and mode. JSON reports, FFmpeg logs, exact launch/codec argv and the
received TS bitstream are retained. The bitstream capture is diagnostic;
decoding and presentation occur continuously before EOS, not after download.

Reproduce headless comparisons and overload (choose a fresh base directory):

```bash
.venv/bin/python scripts/run_streaming_demo.py --headless --mode direct --output results/recheck/final-direct
.venv/bin/python scripts/run_streaming_demo.py --headless --mode bicubic --output results/recheck/final-bicubic
.venv/bin/python scripts/run_streaming_demo.py --headless --mode nanovsr --output results/recheck/final-nanovsr
.venv/bin/python scripts/run_streaming_demo.py --headless --mode bicubic --processing-delay 0.07 --output results/recheck/overload
.venv/bin/python scripts/verify_streaming.py results/recheck --failure-checks
```

See the report for buffering, timestamp, timing, and display-count definitions.

The focused natural-video study is documented in
[the natural-video milestone report](logs/natural-streaming-milestone-2026-10-15.md).
Source credits/licenses are in [ATTRIBUTION.md](videos/natural-source/ATTRIBUTION.md).
It uses 640×360 references/playback, 320×180 delivery, and unchanged NanoVSR
native1280×720 reconstruction followed by resizing. Offline quality scores
cover every reconstructed frame; timed playback completeness is separate.

Launch the saved road CRF23 asset with explicit NanoVSR warmup:

```bash
.venv/bin/python scripts/run_streaming_demo.py --mode nanovsr --config configs/natural-road-nanovsr.json --output results/my-natural-playback
```

This replays the saved bitstream over localhost. Set `verification: true` in a
copied configuration to enable bitstream capture, frame hashing and timestamp
checks. Keep it false for timing. `warmup_chunks: 0` records cold model operation;
`3` performs three warmup chunks before scheduling and reports that cost. Set
`warmup_pipeline: true` to warm the exact transfer/model/conversion/resize path;
false retains the narrower model-only warmup experiment.

The completed study found NanoVSR quality gains over bicubic on road/waterfall
and a loss on flowers. At the selected near-matched rates, direct delivery won
on quality and client cost. Its road point passed three 150 ms buffer runs;
the current NanoVSR GUI path showed latency drift despite fitting in VRAM.
See the report and CSVs for the measured scope and limitations.

## Progress presentation handoff

See [the report-writing handoff](logs/progress-report-handoff.md) and [presentation assets / fish launch commands](presentation_assets/REPRODUCE.md). The existing comparison is complete. The additional sustained NanoVSR trial stopped at 80 seconds after playback drops exceeded the guard threshold; it is a recorded failure, not a five-minute pass. Direct delivery is the reliable live-demo fallback; a clearly labelled offline comparison backup is provided.

## Repository distribution

See [repository contents and setup notes](docs/REPOSITORY_CONTENTS.md) for upstream model revisions, excluded local assets, dependency records, and fresh-clone requirements. Source attribution and compact measured results are included; pretrained checkpoints, environments and large raw outputs remain local.

## New model: UpStreamVSR

Project goals and milestones are in [docs/GOALS.md](docs/GOALS.md). The model design, current measurements, open risks and cloud training commands are in [docs/MODEL.md](docs/MODEL.md). The model is not yet trained on the full dataset.
