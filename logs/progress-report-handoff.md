# BTP VSR progress-report handoff — 2026-10-15

> **Superseded (October 2026 progress presentation).** This handoff covers the
> earlier study: off-the-shelf NanoVSR, bicubic and direct delivery over the first
> MPEG-TS pipeline (`server/`, `client/`). Its numbers do not describe the final
> system. For the final model, live pipeline and results, see the top-level
> [README](../README.md) and [docs/REPORT.md](../docs/REPORT.md).


Status: evidence and presentation assets consolidated. Existing focused comparison is complete. The new sustained NanoVSR test **failed and stopped early**, as specified; it is not a five-minute stability pass. This is a report-writing handoff, not the academic report.

## Objective and evidence-based outcome

Build a practical video streaming pipeline and evaluate trade-offs among transmitted bandwidth, reconstructed visual quality, application latency and client compute. A fixed upscale factor is not the objective. NanoVSR's native x4 architecture is an existing pretrained baseline, not a project invention or an efficient-final-solution assumption.

On three selected natural clips, NanoVSR improves aligned RGB PSNR/SSIM over bicubic on road and waterfall but worsens flowers. At near-matched MPEG-TS rates, direct 640×360 delivery beats NanoVSR on all three and costs much less client compute. The synchronous NanoVSR GUI path fits memory but misses the short-run latency criterion and loses useful playback during the longer trial. These are measured local findings, not significance tests or evidence of general superiority.

## Completed milestones and exact scope

- Strict NanoVSR checkpoint loading, deployment conversion and synthetic offline x4 smoke evaluation: [x4 milestone](x4-evaluation-milestone-2026-10-14.md).
- CDA CUDA operator and synthetic recurrent model inference, state reuse/reset: [environment](cda-environment-milestone-2026-10-14.md), [inference](cda-inference-milestone-2026-10-14.md). Authentic codec MV/residual integration remains pending.
- Finite-file live H.264 encoding → localhost TCP → live decode → reconstruction/resize → continuous OpenCV playback; explicit EOS, bounded queue, cancellation/connection-failure checks, overload and 91-frame tail-chunk checks: [streaming milestone](streaming-milestone-2026-10-15.md). These functional tests used synthetic video and smaller dimensions.
- Natural three-clip quality comparison: 31 summary points, 9,000 per-frame scores; exact shared LR assets and matched direct rates. Headless verification matches every streaming output to offline arrays for all three modes on flowers. Thirty-nine valid GUI timing runs, 11,340 submissions, zero short-run drops: [natural milestone](natural-streaming-milestone-2026-10-15.md).
- Current handoff: run inventory, independent summary/raw checks, source/checkpoint preservation, sustained trial, visible demo checks and offline backup, presentation plots/data and reproduction commands.

## Run inventory and preservation

Host process/GPU inspection before work found no evaluation jobs running. The prior transcript's unfinished appearance did not imply missing results. Inventory: **39 valid timed runs, 3 diagnostic verification runs, 10 excluded completed runs, 1 failed cancelled attempt; zero missing repetitions**. The 39 consist of three repetitions in each of 12 clip/mode/startup groups at 350 ms, plus three road/direct tests at 150 ms.

The first flowers/direct trial overlapped CPU verification; its exclusion and raw files remain. A serial flowers/direct replacement already exists in `streaming-continuation`, so no duplicate comparison was launched. Nine earlier model-only warm trials remain excluded from the main comparison and replaced by full-pipeline warm trials. The cancelled waterfall/cold attempt remains failed; its replacement completed. See [run inventory](../presentation_assets/data/run-inventory.csv), [group counts and replacement paths](../presentation_assets/data/inventory-summary.json), and copied exclusion/failure JSON. Current stability and demo attempts are additional engineering checks, never inserted into those 39 runs.

Both `.venv` and `.venv-cda` package freezes still match their saved prior freezes. Both pretrained checkpoint hashes pass. Both upstream working trees are clean. Original results, licenses and user files remain. Root `.git` is unavailable/invalid, so a root revision/diff cannot be truthfully supplied; final handoff source hashes identify our implementation. No package install, driver/CUDA change, training or CDA data search was done.

## Architecture and measurement protocol

[Architecture diagram](../presentation_assets/figures/architecture.png): finite CFR file or validated pre-encoded asset → server pacing → H.264/MPEG-TS in framed TCP records → client FFmpeg RGB decode → bounded six-frame queue → direct / bicubic / NanoVSR processing → configured playout wait → OpenCV submission. Header carries configuration, frame PTS and common-monotonic epoch; data records preserve encoded predictive frames; EOS carries frame count and return status. Receiver backpressure bounds queue occupancy; it does not guarantee bounded end-to-end age. No adaptive bitrate, reconnection, audio, cross-host clock synchronization or live-camera ingestion is implemented.

The original functional tests launch a paced encoder during transmission. Their application latency includes server encoder startup and scheduling deviations. The natural comparison and presentation demos instead replay pre-encoded TS assets paced by packet positions and PTS: **no live encoding cost is included**. Hashing/probing occur before source scheduling. The sustained trial stream-copies and remuxes repetitions before playback, preserving decoded frames while creating continuous PTS; its MPEG-TS bytes are not byte-identical to thirty concatenated original containers. All 9,000 decoded frames were independently verified equal to thirty original loops before timing.

All natural modes play at 640×360, 30 fps. Direct transmits 640×360; B/C share exactly one 320×180 bitstream per clip/CRF. libx264 veryfast, zerolatency, one thread, GOP30, no B frames, one reference, disabled scene cuts, yuv420p limited BT.709. CRF18/23/28 screening plus four targeted direct encodes; equal CRF across resolutions is not equal bitrate. Reported kbps = actual MPEG-TS bytes ×8 / reference seconds /1000, including mux overhead, excluding TCP/IP. Per-run JSON separately counts application protocol bytes. Do not use a failed partial run's planned-duration bitrate as an achieved network rate.

NanoVSR is unchanged float32 pretrained inference, native 320×180 → **1280×720**, quantized/clamped to uint8, then CPU bicubic →640×360. Costs include that intermediate output, transfers and final resizing. Independent non-overlapping chunks of three reset both recurrent directions; maximum future lookahead is 2/30 s =66.7 ms. A loop boundary every 300 frames aligns with a chunk reset; model/decoder/connection/schedule are not restarted. The 350 ms playout buffer is a rendering target from the server schedule, not an extra unbounded FIFO. Frames more than 500 ms beyond that target may be dropped after reconstruction; approximately 850 ms latency can occur before dropping, plus GUI call duration. No encoded predictive frames are silently dropped.

Application latency is common-host monotonic GUI-call-return minus scheduled source time. It includes deliberate playout wait and pipeline delays; it is **not physical display or camera-to-display latency**. Percentiles describe submitted frames and are always accompanied by drops. Unique GUI submissions are not physical refresh counts. Processing-only fps = frames / summed reconstruction-processing time, excluding decoding, GUI, waits and startup; active submission fps = (submitted−1)/(last−first submission). Submission/source-duration ratio measures completeness and is not active pacing. Queue age = decode completion to processing start, including chunk collection. PyTorch allocated/reserved memory excludes driver context/compositor; zero in direct/bicubic means no model allocator use.

Cold means no explicit warmup, not reset hardware/OS caches. Full warm means three exact-layout zero-input chunks through transfer/model/quantization/D2H/final resize before connection. Load/warmup times are separate; GUI remains unwarmed. The original timed trials had no simultaneous evaluation job, no PNG writes and no per-frame diagnostic hashes. This handoff's screenshot/demo checks are separate from performance evidence.

## Data, provenance and limitations

| Clip / role | Source / license | Original resolution | Selected source span | Reference frames |
|---|---|---:|---|---:|
| flowers / low motion | Flowers in park, Editor / CC BY 3.0 | 1920×1080 | first 9 s | 270 |
| road / camera and object motion | Narrowing 2+1 road, LA2 / CC BY-SA 3.0 | 1280×720 | 10–20 s | 300 |
| waterfall / fine textures | Professor Creek Waterfall, BLMUtah / Bill Stevens / US federal public domain | 1920×1080 | first 10 s | 300 |

[Attribution and license links](../presentation_assets/ATTRIBUTION.md), [download URLs, hashes, probe metadata and exact conversion argv](../presentation_assets/data/provenance.json). Reference conversion normalizes PTS, converts to 30 fps by nearest-frame duplication/drop, downsamples bicubically to 640×360, removes audio and stores lossless RGB FFV1/NUT plus uint8 arrays. No smaller source was upscaled into a reference. Sources were already compressed; these are converted references, not pristine camera masters. Waterfall includes people and an original title. Initial MKV intermediates are retained but not used due timestamp quantization.

Whole-stream quality is scored before lossy preview encoding on aligned uint8 RGB: float64 PSNR with max255; SSIM Gaussian11×11 sigma1.5, per-channel mean and valid interior; no extra crop. Clip values are arithmetic means of every frame's score. Dropped playback frames never remove quality rows. No perceptual, temporal-flicker, audio or user-study claims follow from these metrics.

## Final focused-comparison results

Three valid repetitions per row, 350 ms buffer; NanoVSR rows below use full-pipeline warmup. Ranges are observed min–max across repetitions, not confidence intervals. Full precision, actual bytes, protocol counters, queue metrics, cold results and lower-buffer rows are in [comparison CSV](../presentation_assets/data/comparison.csv) and [individual repetitions](../presentation_assets/data/valid-repetitions.csv). All quality points, including the initial road match refinement, remain in [quality CSV](../presentation_assets/data/quality-all-points.csv).

| Clip | Mode / CRF | TS kbps (direct Δ vs LR) | PSNR dB / SSIM | Processing ms/frame | Submission fps range | Latency p50 / p95 / max ranges, ms | Drops / submissions | Peak allocated MiB |
|---|---|---:|---:|---:|---:|---|---:|---:|
| flowers | direct / 30.1 | 484.79 (+0.03%) | 30.687 / 0.90241 | 0.01 | 30.01–30.02 | 353.1–353.2 / 353.6–353.7 / 357.4–358.5 | 0 / 810 | 0.00 |
| flowers | bicubic / 23 | 484.62 (+0.00%) | 29.253 / 0.87952 | 0.49 | 30.01–30.01 | 353.2–353.3 / 353.7–353.9 / 356.7–358.2 | 0 / 810 | 0.00 |
| flowers | nanovsr / 23 | 484.62 (+0.00%) | 29.088 / 0.87352 | 31.15 | 29.33–29.91 | 392.0–482.5 / 438.4–589.0 / 445.2–623.1 | 0 / 810 | 358.83 |
| road | direct / 30.4 | 452.70 (+1.21%) | 32.730 / 0.90796 | 0.01 | 30.02–30.02 | 353.2–353.3 / 353.7–353.9 / 358.9–359.9 | 0 / 900 | 0.00 |
| road | bicubic / 23 | 447.29 (+0.00%) | 30.504 / 0.88714 | 0.50 | 30.01–30.01 | 353.3–353.3 / 353.8–353.8 / 356.7–358.1 | 0 / 900 | 0.00 |
| road | nanovsr / 23 | 447.29 (+0.00%) | 31.743 / 0.89821 | 31.31 | 29.24–29.67 | 431.7–508.3 / 507.9–640.6 / 529.1–677.6 | 0 / 900 | 358.83 |
| waterfall | direct / 29.8 | 444.58 (+2.18%) | 29.568 / 0.86169 | 0.01 | 30.01–30.02 | 353.2–353.4 / 353.7–353.9 / 358.4–359.1 | 0 / 900 | 0.00 |
| waterfall | bicubic / 23 | 435.11 (+0.00%) | 27.759 / 0.80361 | 0.47 | 30.01–30.01 | 353.2–353.8 / 353.7–354.8 / 357.9–364.1 | 0 / 900 | 0.00 |
| waterfall | nanovsr / 23 | 435.11 (+0.00%) | 28.216 / 0.81737 | 31.40 | 29.23–29.44 | 475.4–510.4 / 579.0–641.3 / 609.3–677.9 | 0 / 900 | 358.83 |

At CRF23, NanoVSR−bicubic is flowers **−0.165 dB / −0.00600 SSIM**, road **+1.239 / +0.01107**, waterfall **+0.458 / +0.01377**. Signs persist at CRF18 and28. Matched direct rates are +0.03%, +1.21%, +2.18% above LR, respectively; direct gains over NanoVSR are **1.599, 0.986, 1.352 dB**. These are close-rate comparisons with disclosed residual differences, not exact equal-rate claims. Direct CRF28 also beats NanoVSR LR CRF18 in both metrics at lower measured rates on these three clips.

Warm NanoVSR's forward takes approximately24.3–24.6 ms/frame, final resizing5.0–5.4 ms/frame, full processing31.2–31.4 ms/frame. The small remaining33.3 ms frame budget is consumed by the current synchronous renderer/pipeline. This is an inference about this implementation, not proof the model could never sustain30 fps in another pipeline. VRAM capacity is not the observed constraint.

Cold and warm startup remain separate:

| Clip | Startup | First chunk ms | Model load s | Explicit warmup s | Latency p95 ms |
|---|---|---:|---:|---:|---:|
| flowers | cold | 273.5–278.3 | 1.408–1.446 | 0.000–0.000 | 485.2–526.9 |
| flowers | full pipeline warm3 | 91.7–93.3 | 1.380–1.423 | 0.448–0.455 | 438.4–589.0 |
| road | cold | 272.4–279.4 | 1.408–1.450 | 0.000–0.000 | 472.6–525.1 |
| road | full pipeline warm3 | 91.5–92.8 | 1.381–1.430 | 0.448–0.459 | 507.9–640.6 |
| waterfall | cold | 275.9–587.2 | 1.428–2.676 | 0.000–0.000 | 501.1–762.2 |
| waterfall | full pipeline warm3 | 92.4–94.1 | 1.383–1.426 | 0.453–0.460 | 579.0–641.3 |

Direct road at the unchanged matched-rate asset and **150 ms** buffer passed all three repetitions:300/300 submissions each, no drops, active30.014–30.018 fps, p50 153.15–153.23 ms, p95 153.64–153.69 ms, max157.90–158.84 ms. Lower-buffer testing followed the declared three-run criterion (zero drops and p95<450 ms at350 ms buffer); NanoVSR road failed it. This is not an optimal-buffer search.

## Sustained stability result: failed early, settings unchanged

Planned300 seconds/9,000 frames from thirty ten-second road loops, same320×180 CRF23, native x4+resize, chunk3, queue6,350 ms buffer,500 ms late tolerance, full-pipeline warm3. One timed trial ran serially, with no model scoring, preview encoding, figure rendering or screen capture concurrently. Logging was bounded to at most9,000 scalar frame and3,000 chunk records plus progress windows; no reconstructed frames were saved. Five-second `nvidia-smi` samples record temperature/utilization/memory/power where available.

The first preflight failed before connection because rounded first-frame interval extrapolation incorrectly rejected long CFR timestamps. The server now validates against the full-span mean interval; all individual timestamps still must match the CFR grid. That failed preflight is preserved. The subsequent timed trial was intentionally stopped by a declared guard after three consecutive300-frame windows each dropped>150 frames (the alternative guard condition was>1.5 s end schedule age). It reached **2,400 processed frames /80 seconds of source**, not five minutes. It submitted **1,680 (70% of the processed prefix)** and dropped **720 (30%)**; the unprocessed6,600 planned frames are uncompleted, not counted as late drops. Submitted fraction of the full planned stream is18.67%.

Measured processing-only throughput **31.63 fps** versus active unique submissions **20.87 fps** overall; the last three windows delivered only **13.23–13.33 fps**. Submitted-frame latency p50 **751.45 ms**, p95 **851.92 ms**, maximum **852.83 ms**. Queue peak stayed6, maximum decoded queue age308.28 ms. Peak PyTorch allocated/reserved376,262,144 /545,259,520 bytes (358.83 /520 MiB). The GUI return event may exceed850 ms because the late check precedes GUI submission. A bounded queue and average processing capacity do not guarantee acceptable playback under synchronous scheduling.

Temperature rose during the trial; see telemetry and the summary below. These samples do not establish thermal throttling causality. The deliberate early abort closes the connection; retained BrokenPipe/receiver EOF and missing clean EOS are expected consequences, not a passed stream. The overload limitation is unresolved. Do not hide it with a larger buffer or smaller input. The repeated clip is not additional independent quality evidence.

[Ten-second trend figure](../presentation_assets/figures/stability-trends.png), [window CSV](../presentation_assets/data/stability-windows.csv), [summary JSON](../presentation_assets/data/stability-summary.json), [raw client/server evidence](../presentation_assets/evidence/stability/), [GPU samples](../presentation_assets/evidence/gpu-telemetry.csv), [loop preparation/guard definition](../presentation_assets/evidence/loop-protocol.json).


Telemetry observed 18 samples (including setup/teardown): temperature 52–82 °C; GPU utilization 0–77%; maximum sampled whole-device memory 613 MiB. Sampling is sparse and differs from allocator peak accounting.

## Demonstration and presentation assets

[Exact fish-compatible kitty commands](../presentation_assets/REPRODUCE.md) launch direct, bicubic and NanoVSR on the same road clip and playback size. Small window-title labels identify mode/TX/playback/native processing; current UI is reused. Use direct as the reliable live presentation path, then show NanoVSR as a measured comparison with documented cost and long-run failure. Each launch refuses an existing result directory. Escape intentionally cancels; use EOS for a clean demonstration.

Visible playback was checked separately using two playback-window captures per successful mode. These confirm changing natural-video content at observed moments, not every physical frame. Capture attempts are not timing evidence. An initial capture helper still searched the old window title after labels were added; that failed demo attempt is retained. The helper now uses the same title function as the client, and all three modes were rechecked. Window captures are under `presentation_assets/evidence/demo-*-verified/`.

[Offline backup video](../presentation_assets/demo/OFFLINE-road-comparison.mp4) is ten seconds of precomputed aligned reference/direct/bicubic/NanoVSR frames. Every frame is labelled **OFFLINE — NOT live playback**. It is lossy display-only encoding, never scored, not a screen recording and not performance evidence. No screen recording ran during timing. Road derivatives, including the backup, retain CC BY-SA3.0 attribution.

Figures: [flowers rate-quality](../presentation_assets/figures/quality-rate-flowers.png), [road](../presentation_assets/figures/quality-rate-road.png), [waterfall](../presentation_assets/figures/quality-rate-waterfall.png), [latency and completeness](../presentation_assets/figures/latency-playback.png). SVG versions accompany PNGs. Stars mark targeted direct encodes; plot lines connect measured points without asserting interpolation accuracy. [Positive road crops](../presentation_assets/crops/road.png) and [negative flower crops](../presentation_assets/crops/flowers.png) use identical reference-selected coordinates at frame150 and2× nearest display enlargement; labels show all four methods. Crops do not establish temporal quality.

## Final verification and small implementation fixes

Fresh artifact audit passes all31 quality rows/9,000 frame scores and39 valid timing runs/11,340 submissions. Metric self-checks and186 independent saved-array rechecks pass. A new headless road/NanoVSR diagnostic replay after the fixes passed300/300 timestamps, payload/protocol bytes and every decoded/reconstructed hash against offline arrays. All three labelled GUI checks reached300/300 submissions and clean EOS; two changing-content snapshots per mode were inspected. The offline backup probes as300 frames/10 seconds and its labelled frame was inspected. See `presentation_assets/evidence/*audit.log` and final validation JSON.

Edits are limited to full-span CFR validation for long clips, an optional sustained-run guard/progress log, a configurable launcher timeout, shared demo-title labels/capture matching, and handoff preparation/consolidation scripts. Previous live and pre-encoded paths remain. New scripts: `prepare_presentation_demo.py`, `run_presentation_stability.py`, `build_presentation_assets.py`, `plot_presentation.py`, `make_presentation_backup.py`, `write_progress_handoff.py`, `verify_presentation_assets.py` in `scripts/`. Exact launch commands and configs are saved with evidence. No model source changed.

## Software, hardware and model identity

Current recorded environment: Linux7.2.9 Arch x86_64; AMD Ryzen7 4800H; NVIDIA GeForce RTX3050 Laptop,4 GiB nominal VRAM; NVIDIA driver615.78.08. `.venv`: Python3.12.14, torch2.14.0+cu130 (runtime CUDA13.0), OpenCV5.0.0, NumPy2.5.3; FFmpeg n9.0.2. Driver-advertised CUDA capability is not the framework runtime. System Matplotlib3.11.2 generates figures without changing either environment. Full version/build strings and CPU information: [software-hardware JSON](../presentation_assets/evidence/software-hardware.json); dependency freezes in the same evidence directory.

NanoVSR upstream revision `b48e2bad89d01e22226eb8613bee25cb45623fae`, checkpoint `external/nanovsr/checkpoints/nanovsr_226k.pth`, SHA256 `be3df12f2a0d99d9699656aa4a85a51c119663aa84c7040ac6580a0215d0b00e`. Strict loading; architecture and pretrained parameters unchanged.

CDA upstream revision `5630821e5df4e110878d0caae39bcc53f658e9a9`, checkpoint `external/cda-vsr/pretrained_models/best.pth`, SHA256 `afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e`. Earlier verified environment: torch2.1.0+cu121, torchvision0.16.0+cu121, NumPy1.26.4, MMCV2.2.0. Both checkpoint keys strictly load; inference uses `params` consistent with official tests. CDA structural inference is verified only with synthetic RGB/zero MV/zero residual. Authentic matched dense motion-vector/residual extraction, units/reference semantics and integration are pending; no CDA quality or streaming result exists.

## Engineering contribution, limitations and future work

Our work is model integration, strict checkpoint validation, reproducible aligned evaluation, paced transport/decode/playback, bounded queues/EOS/errors, measurements, exact shared bitstreams/rate matching, provenance and presentation evidence. NanoVSR/CDA architectures and pretrained weights belong to their upstream authors. No new model, training result, algorithmic novelty or percentage completion is claimed.

Unresolved limits: sustained NanoVSR playback collapses under current synchronous pacing; only three short compressed natural clips, one model/checkpoint/device/encoder preset; no subjective/temporal metrics, LAN/WAN shaping, ABR, audio synchronization, camera ingest or physical display timing. Warm/cold power state is not standardized. No statistical significance or generalization follows from three repetitions. Prior short runs may pass clean EOS while still having growing latency. GUI/Qt emits font/Wayland warnings but verified playback works through available desktop access.

Candidate future work, **not implemented**: separate bounded reconstruction and presentation workers; investigate timestamp-aware decoded-frame skip/recovery without discarding predictive encoded frames; measure causal chunk1 or a model designed for the actual output scale; evaluate validated mixed precision/GPU resizing and transfers; test more licensed natural content, matched-rate encoders and controlled network conditions; implement/authenticate CDA codec-side preprocessing before any authentic CDA claim. Each requires new correctness and timing checks.

Defensible presentation claims:
- We implemented and functionally verified video-only localhost streaming with optional pretrained reconstruction and explicit measurement scope.
- Whole-stream aligned quality and actual transmitted bytes show content-dependent NanoVSR benefit over bicubic; direct wins at the selected near-matched rates.
- NanoVSR fits this GPU but the present pipeline fails the sustained-playback test; processing-only FPS is insufficient evidence of real-time delivery.
- Direct delivery demonstrated the lower150 ms playout setting on three short road runs; no global optimum is established.
- CDA structural recurrent inference works; authentic integration remains pending.

Claims to avoid: native-x2 NanoVSR; novel VSR architecture; VSR universally beats direct or bicubic; five-minute stability passed; zero drops imply low latency; preview video demonstrates live speed; physical display/camera latency measured; looped clips provide independent quality samples; statistical significance, optimal bandwidth savings, full CDA integration or training completed.

Stop here: no new research or experimental sweep is part of this handoff.
