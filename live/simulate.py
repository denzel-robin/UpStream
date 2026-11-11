#!/usr/bin/env python3
"""Run the trade-off controller against a simulated link, without video or a GPU.

Frame i of the chosen rung is sent at i / fps with that rung's size (keyframes
key_ratio times the size of other frames, the mean matching the rung's rate), crosses
a first-in first-out link whose rate follows a schedule, and is observed by the
controller when it arrives. A switch request takes effect at the first keyframe sent
after it, as on the real server; a request with skip drops the rest of the group of
pictures. Model times come from a fixed table, scaled by a
GPU load schedule, so resource pressure can be tested too.

Prints one line per second: link rate, rung, queuing delay and predicted PSNR.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from live.controller import TradeoffController, load_table  # noqa: E402
from live.link import parse_link  # noqa: E402
from live.policy import DepthPolicy  # noqa: E402
from upstream_vsr.encoding import KEYINT  # noqa: E402

# Measured on the RTX 3050 Laptop, third design, float32, ms per frame per exit (docs/MODEL.md).
MODEL_MS = {2: (5.0, 6.3, 7.6), 4: (8.1, 9.5, 10.9)}


def at(schedule: list[tuple[float, float]], t: float) -> float:
    value = schedule[0][1]
    for begin, v in schedule:
        if t >= begin:
            value = v
    return value


def simulate(table: dict, ladder: list[str], link: list[tuple[float, float]], seconds: float, start: str,
             fps: float = 30.0, playout_delay: float = 0.15, gpu_load: list[tuple[float, float]] | None = None,
             gpu_share: float = 0.6, key_ratio: float = 4.0, use_model: bool = True) -> dict:
    policy = DepthPolicy(3, 1 / fps) if use_model else None
    controller = TradeoffController(table, ladder, 1 / fps, latency_budget_s=playout_delay - 0.025,
                                    policy=policy, gpu_share=gpu_share)
    controller.start(start, 0.0)

    def sizes(name: str) -> tuple[int, int]:
        """(keyframe bytes, other frame bytes) for a rung."""
        mean = table[name]['kbps'] * 1000 / 8 / fps
        other = mean * KEYINT / (KEYINT - 1 + key_ratio)
        return int(other * key_ratio), int(other)

    def set_gpu(t: float) -> None:
        if policy is None:
            return
        load = at(gpu_load, t) if gpu_load else 1.0
        for scale, times in MODEL_MS.items():
            for e, ms in enumerate(times):
                policy.expected[(scale, e)] = ms * load / 1000

    controller.observe_rates({'kbps': {n: table[n]['kbps'] for n in ladder},
                              'key_bytes': {n: sizes(n)[0] for n in ladder}})
    current, requested, free_at, pending, frames, skipped, skipping = start, start, 0.0, [], [], 0, False
    total = int(seconds * fps)
    for i in range(total + 1):
        now = i / fps if i < total else float('inf')
        # Arrivals up to this send time reach the controller first.
        while pending and pending[0]['arrived'] <= now:
            frame = pending.pop(0)
            set_gpu(frame['arrived'])
            controller.observe_packet(frame['sent'], frame['arrived'], frame['bytes'], frame['keyframe'])
            if (decision := controller.decide(frame['arrived'])) is not None:
                requested, skipping = decision[0], skipping or decision[2]
            frames.append(frame)
        if i == total:
            break
        keyframe = i % KEYINT == 0
        if keyframe:
            current, skipping = requested, False
        elif skipping:
            skipped += 1
            continue
        size = sizes(current)[0 if keyframe else 1]
        rate = at(link, now)
        begin = max(now, free_at)
        free_at = begin + size * 8 / (rate * 1000)
        pending.append({'index': i, 'rung': current, 'sent': now, 'arrived': free_at, 'bytes': size,
                        'keyframe': keyframe, 'link_kbps': rate})

    late = [f for f in frames if f['arrived'] - f['sent'] > playout_delay - 0.025]
    quality = []
    for f in frames:
        set_gpu(f['sent'])
        q = controller.quality(f['rung'])
        quality.append(q if q is not None else float('nan'))
    return {'frames': frames, 'decisions': controller.trace, 'late': len(late), 'skipped': skipped,
            'mean_kbps': sum(f['bytes'] for f in frames) * 8 * fps / len(frames) / 1000,
            'mean_predicted_psnr': sum(q for q in quality if q == q) / max(sum(q == q for q in quality), 1),
            'max_delay_ms': max(f['arrived'] - f['sent'] for f in frames) * 1000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rate-quality', type=Path, default=ROOT / 'configs/rate_quality.json')
    parser.add_argument('--crfs', type=int, nargs='+', default=[18, 23, 28, 33])
    parser.add_argument('--direct-crfs', type=int, nargs='+', default=[23, 28, 33, 38, 43])
    parser.add_argument('--link', default='0:3000,20:700,40:1800', help='kbit/s from each second on')
    parser.add_argument('--gpu-load', help='model time multiplier from each second on, for example 0:1,30:3')
    parser.add_argument('--gpu-share', type=float, default=0.6)
    parser.add_argument('--no-model', action='store_true', help='client without the model: direct only')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--start', default='x4@28')
    parser.add_argument('--playout-delay', type=float, default=0.15)
    parser.add_argument('--json', type=Path, help='write the full result here')
    args = parser.parse_args()

    table = load_table(args.rate_quality)
    ladder = ([f'direct@{c}' for c in args.direct_crfs] + [f'x2@{c}' for c in args.crfs] + [f'x4@{c}' for c in args.crfs])
    result = simulate(table, ladder, parse_link(args.link), args.seconds, args.start,
                      playout_delay=args.playout_delay, gpu_share=args.gpu_share, use_model=not args.no_model,
                      gpu_load=parse_link(args.gpu_load) if args.gpu_load else None)
    fps = 30
    for second in range(int(args.seconds)):
        chunk = [f for f in result['frames'] if second <= f['sent'] < second + 1]
        if not chunk:
            print(f'{second:3d}s skipped')
            continue
        delay = max(f['arrived'] - f['sent'] for f in chunk) * 1000
        rung = chunk[-1]['rung']
        print(f'{second:3d}s link {chunk[0]["link_kbps"]:6.0f}  rung {rung:10s} {table[rung]["kbps"]:7.1f} kbps  '
              f'max delay {delay:6.1f} ms')
    for d in result['decisions']:
        print(d)
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in result.items() if k not in ('frames', 'decisions')})
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
