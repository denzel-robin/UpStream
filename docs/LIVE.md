# Live pipeline

`live/server.py` and `live/client.py` run UpStreamVSR on a real stream: encode, send,
decode with motion vectors, reconstruct causally and play back on time. They replace
`server/` and `client/`, which belong to the earlier NanoVSR study.

```
source ─► ffmpeg (training filter graph, -re) ─┬─ 720p  direct ─┐
                                               ├─ 360p  x2     ─┼─► server sends the one requested
                                               └─ 180p  x4     ─┘     (switch on a shared keyframe)
                                                                              │ TCP
client:  receive ─► decode (PyAV, motion vectors) ─► reconstruct (GPU, exit per frame) ─► display
              ▲                                                                       │
              └──────────────────── switch request (S record) ◄────────────────────────┘
```

## Server

- One ffmpeg reads the source at its frame rate and encodes all three renditions with
  `upstream_vsr/encoding.py`, the same filter graph and x264 settings that built the
  training data. The only difference is `repeat-headers=1` (SPS/PPS before every
  keyframe), so a client can start decoding a rendition at any keyframe. It does not
  change any decoded pixel.
- Keyframes every 30 frames in every rendition, no B-frames, so switching is clean and
  every packet decodes to its frame immediately.
- Each rendition comes back through its own pipe as NUT, whose packets carry their own
  length, so nothing waits for the next frame to find a packet's end.
- Packets are released on the source schedule: frame i leaves at t0 + i / fps, as a
  camera would deliver it. ffmpeg's start-up burst is off. `late_releases` in the report
  counts frames the encoder could not deliver on time.
- Only the selected rendition is sent; the report gives frames and bytes per rendition.
- A switch request takes effect at the next frame that is a keyframe in the requested
  rendition (at most 1 s later at 30 fps).

## Client

Four threads joined by small bounded queues:

| Thread | Work |
|---|---|
| receive | socket records to packets |
| decode | a fresh `MotionDecoder` at each switch (it always starts on a keyframe); RGB, motion field and frame type converted exactly as in training |
| reconstruct | `UpStreamVSR.step` with the recurrent state; direct frames pass through and reset the state; x2 and x4 keep it across a switch |
| main | waits until each frame is due, shows it or skips it if later than `--drop-late`, records timings |

The exit is chosen per frame by `live/policy.py`: the deepest exit whose expected time
fits before the frame is due, capped at one frame interval. Expected times start from
the median of a warm-up run before connecting and then follow the measured times. A
frame that is skipped on screen is still reconstructed, so the state has no gaps.

`--display` shows the video in a window with a status overlay (rung, model exit and
time, sent bitrate, link estimate, latency); `--compare` puts bicubic from the received
frame on the left and UpStreamVSR on the right; Escape closes it. OpenCV is imported
before PyAV when `--display` is given: PyAV bundles its own libxcb, and Qt hangs
opening a window on that copy.

```bash
python live/server.py data/test-clips/NRqG3AEOVGg@1440.mp4 --crfs 18 23 28 33 \
    --direct-crfs 23 28 33 38 43 --start x4@28 --link 0:2500,10:500,20:1500
python live/client.py --checkpoint runs/kaggle-6-v3/run/best-calibrated.pt --controller \
    --half --playout-delay 0.15 --display --compare
```

`--schedule 0:x4,90:x2,180:direct` sends switch requests at fixed frames;
`--controller` lets the trade-off controller choose instead (docs/CONTROLLER.md). `--half` runs the model in float16.

## First measurements (2026-11-10)

RTX 3050 Laptop, one machine, 10 s of a test clip at CRF 23, v2 checkpoint (trained on
the colour-shifted data, so only the timings mean anything), schedule x4 then x2 then
direct then x4, 30 ms playout delay.

| | FP32 | FP16 |
|---|---:|---:|
| Frames, dropped | 300, 0 | 300, 0 |
| Ready after release, mean / p95 / max | 12.9 / 16.8 / 18.3 ms | 11.8 / 13.4 / 14.5 ms |
| Model at x4, deepest exit, mean | 9.6 ms | 7.3 ms |
| Model at x2, deepest exit, mean | 9.9 ms | 7.6 ms |
| On-screen latency | 30 ms (the playout delay) | 30 ms |

"Ready after release" is transport, decoding, reconstruction and the copy back from the
GPU. It does not include encoding, because the server encodes ahead of the release time.
Transport on localhost is 0.15 ms; decoding is 2 ms for x2/x4 frames and about 16 ms for
720p direct frames (RGB conversion included).

Float16 against float32 on 16 test windows: identical PSNR (32.843 at x2, 30.059 at x4),
largest pixel difference 1 level. Sampling positions in `warp` stay float32.

Payload at CRF 23 on this clip: about 250 kbit/s at x4, 280 at x2 and 610 for direct.

With this much headroom the policy always picks the deepest exit. Shallower exits will
be used when the machine is loaded or the playout delay is short; that is measured in
milestone 5.

## Limits

- Latency is only measured with both ends on one machine, where the clocks agree.
- Encoding latency is not in the figures above.
- The threads share Python's interpreter lock; decoding and the model mostly run in C
  and CUDA, but heavy load could add jitter.
- No real network shaping yet (milestone 5); `--link` emulates a bottleneck.

## Running it

```
python live/server.py data/test-clips/NRqG3AEOVGg@1440.mp4 --seconds 10 --start x4 --report results/server.json
python live/client.py --checkpoint runs/kaggle-2/run/best-calibrated.pt --schedule 0:x4,90:x2,180:direct \
    --playout-delay 0.03 --half --display --report results/client.json
```
