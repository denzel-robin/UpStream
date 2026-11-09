#!/usr/bin/env python3
"""Measure UpStreamVSR forward time per exit and scale, optionally beside NanoVSR.

Run this alone on an otherwise idle machine; other GPU work will distort the numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upstream_vsr.model import SCALES, UpStreamVSR  # noqa: E402

INPUT_SIZE = {4: (180, 320), 2: (360, 640)}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


@torch.inference_mode()
def time_upstream(model, device, scale, exit_index, frames) -> dict:
    h, w = INPUT_SIZE[scale]
    x = torch.rand(1, 3, h, w, device=device)
    mv = torch.zeros(1, 2, h // 4, w // 4, device=device)
    intra = torch.zeros(1, device=device)
    quality = torch.full((1,), 0.5, device=device)
    state, times = None, []
    for index in range(frames + 20):
        begin = time.perf_counter()
        output, state = model.step(x, mv, intra, state, scale, exit_index, quality=quality)
        output.clamp(0, 1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if index >= 20:  # discard warmup
            times.append((time.perf_counter() - begin) * 1000)
    return {'mean_ms': sum(times) / len(times), 'p95_ms': percentile(times, .95)}


@torch.inference_mode()
def time_nanovsr(root: Path, device, frames) -> dict:
    """Upstream NanoVSR in the earlier pipeline's mode: independent 3-frame chunks."""
    sys.path.insert(0, str(root))
    from models.nanovsr import NanoVSR
    model = NanoVSR(num_feat=32, num_blocks=8).to(device).eval()  # matches nanovsr_226k.pth
    model.switch_to_deploy()
    x = torch.rand(1, 3, 3, 180, 320, device=device)
    times = []
    for index in range(frames // 3 + 7):
        begin = time.perf_counter()
        model(x).clamp(0, 1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if index >= 7:
            times.append((time.perf_counter() - begin) * 1000 / 3)
    return {'mean_ms': sum(times) / len(times), 'p95_ms': percentile(times, .95),
            'params': sum(p.numel() for p in model.parameters())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--num-feat', type=int, default=32)
    parser.add_argument('--exits', type=int, nargs='+', default=[3, 6, 9])
    parser.add_argument('--memory-scales', type=int, nargs='*', default=list(SCALES))
    parser.add_argument('--refine-motion', action='store_true')
    parser.add_argument('--quality', action='store_true', help='include CRF conditioning')
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--nanovsr-root', type=Path, help='path to an external/nanovsr checkout')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = UpStreamVSR(args.num_feat, args.exits, args.memory_scales, args.refine_motion, args.quality)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model = UpStreamVSR(**checkpoint['config'])
        model.load_checkpoint_state(checkpoint['model'])
    model = model.to(device).eval().fuse()

    report = {
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'weights': 'checkpoint' if args.checkpoint else 'untrained (timing only)',
        'deployed_params': sum(p.numel() for p in model.parameters()),
        'upstream': {},
    }
    for scale in (4, 2):
        for exit_index, depth in enumerate(model.exits):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            result = time_upstream(model, device, scale, exit_index, args.frames)
            if device.type == 'cuda':
                result['peak_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
            report['upstream'][f'x{scale}_blocks{depth}'] = result
    if args.nanovsr_root:
        report['nanovsr_226k_chunk3'] = time_nanovsr(args.nanovsr_root, device, args.frames)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n')


if __name__ == '__main__':
    main()
