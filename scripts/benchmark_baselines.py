#!/usr/bin/env python3
"""Benchmark UpStreamVSR against published super-resolution models on the live rungs.

Every test clip is encoded with the live ladder (as scripts/measure_rate_quality.py
does), decoded with motion vectors, and upscaled one frame at a time, causally, by:

  upstream   UpStreamVSR, deepest exit (the live client's usual choice)
  bicubic
  efrlfn, span, nanovsr, cdavsr   (upstream_vsr/baselines.py; nanovsr and cdavsr are x4 only)

Per model and rung: PSNR against the 720p reference, temporal PSNR (frame-to-frame
changes; higher is steadier), and the per-frame time on this GPU with a CUDA sync after
every frame, as the live client measures it. Also parameters and peak GPU memory.

  python scripts/benchmark_baselines.py --half --output results/baselines/offline.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from live.client import load_model  # noqa: E402
from measure_rate_quality import SCALE, encode_ladder, frame_psnr, payload_kbps, temporal, to_tensor  # noqa: E402
from prepare_streamsr import probe  # noqa: E402
from upstream_vsr.baselines import NAMES, Baseline  # noqa: E402
from upstream_vsr.codec import decode_with_motion  # noqa: E402
from upstream_vsr.encoding import frame_step  # noqa: E402
from upstream_vsr.model import crf_to_quality  # noqa: E402


class Upstream:
    """UpStreamVSR behind the baselines' step() interface."""

    def __init__(self, checkpoint: Path, device, half: bool):
        self.model = load_model(checkpoint, device, half)
        self.dtype = torch.float16 if half else torch.float32
        self.device, self.scales, self.state, self.crf = device, (2, 4), None, 23
        self.exit = len(self.model.exits) - 1
        self.parameters = {s: sum(p.numel() for p in self.model.parameters()) for s in self.scales}

    def reset(self):
        self.state = None

    @torch.inference_mode()
    def step(self, x, mv, is_intra, scale):
        intra = torch.tensor([float(is_intra)], device=self.device, dtype=self.dtype)
        quality = crf_to_quality(self.crf).view(1).to(self.device, self.dtype)
        out, self.state = self.model.step(x.to(self.dtype), mv.to(self.dtype), intra, self.state, scale, self.exit,
                                          quality=quality)
        return out.clamp(0, 1)


class Bicubic:
    scales, parameters = (2, 4), {2: 0, 4: 0}

    def reset(self):
        pass

    def step(self, x, mv, is_intra, scale):
        return F.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=False).clamp(0, 1)


@torch.inference_mode()
def run(model, frames, gt, scale: int, device) -> dict:
    lr = to_tensor([f.rgb for f in frames], device)
    mv = torch.from_numpy(np.stack([f.mv for f in frames])).to(device).permute(0, 3, 1, 2)
    model.reset()
    outputs, values, times = [], [], []
    for t, frame in enumerate(frames):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        out = model.step(lr[t:t + 1].float() / 255, mv[t:t + 1], frame.is_intra, scale)
        out = out.float().mul(255).round().div(255)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - begin) * 1000)
        values.append(frame_psnr(out, gt[t]))
        outputs.append(out)
    return {'psnr': float(np.mean(values)), 'temporal': temporal(outputs, gt),
            'ms_median': float(np.median(times[5:])), 'ms_p95': float(np.percentile(times[5:], 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/kaggle-6-v3/run/best-calibrated.pt')
    parser.add_argument('--clips', type=Path, default=ROOT / 'data/test-clips')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seconds', type=float, default=4.0)
    parser.add_argument('--models', nargs='+', default=['upstream', 'bicubic', *NAMES])
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/baselines/offline.json')
    args = parser.parse_args()
    device = torch.device('cuda')

    models, memory = {}, {}
    for name in args.models:
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        models[name] = (Upstream(args.checkpoint, device, args.half) if name == 'upstream' else
                        Bicubic() if name == 'bicubic' else Baseline(name, device, args.half))
        memory[name] = {'weights_mib': (torch.cuda.memory_allocated() - before) / 2 ** 20}

    sources = sorted(args.clips.glob('*.mp4'))[:args.limit]
    clips = []
    for number, source in enumerate(sources, 1):
        begin = time.time()
        info = probe(source)
        fps = info[2] / frame_step(info[2])
        with tempfile.TemporaryDirectory() as temp:
            paths = encode_ladder(source, Path(temp), info[2], args.seconds)
            gt = to_tensor([f.rgb for f in decode_with_motion(paths['reference'])], device)
            rungs = {}
            for rung, path in paths.items():
                rendition = rung.split('@')[0]
                if rendition not in SCALE:
                    continue
                scale, crf = SCALE[rendition], int(rung.split('@')[1])
                frames = list(decode_with_motion(path))
                entry = {'kbps': payload_kbps(path, fps)}
                for name, model in models.items():
                    if scale not in model.scales:
                        continue
                    if name == 'upstream':
                        model.crf = crf
                    torch.cuda.reset_peak_memory_stats()
                    entry[name] = run(model, frames, gt, scale, device)
                    entry[name]['peak_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
                rungs[rung] = entry
        clips.append({'clip': source.name, 'rungs': rungs})
        print(f'[{number}/{len(sources)}] {source.name} {time.time() - begin:.0f} s', flush=True)

    summary = {}
    for rung in clips[0]['rungs']:
        entries = [c['rungs'][rung] for c in clips]
        summary[rung] = {'kbps': float(np.mean([e['kbps'] for e in entries]))}
        for name in models:
            if name in entries[0]:
                summary[rung][name] = {k: float(np.mean([e[name][k] for e in entries]))
                                       for k in ('psnr', 'temporal', 'ms_median', 'ms_p95', 'peak_mib')}
    report = {'checkpoint': str(args.checkpoint), 'half': args.half, 'seconds': args.seconds, 'clips': len(clips),
              'gpu': torch.cuda.get_device_name(), 'parameters': {n: m.parameters for n, m in models.items()},
              'memory': memory, 'summary': summary, 'per_clip': clips}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    names = list(models)
    print(f"{'rung':8s} {'kbps':>6s} " + ' '.join(f'{n:>16s}' for n in names))
    for rung, entry in summary.items():
        cells = [f"{entry[n]['psnr']:6.2f} {entry[n]['ms_median']:5.1f}ms" if n in entry else ' ' * 16 for n in names]
        print(f"{rung:8s} {entry['kbps']:6.0f} " + ' '.join(f'{c:>16s}' for c in cells))


if __name__ == '__main__':
    main()
