#!/usr/bin/env python3
"""Measure why recurrent state helps or not, on prepared windows.

Reports, per scale:
  alignment   mean abs error between frame t and frame t-1 with no motion, with the
              codec motion vectors, and with the vectors negated (a sign check)
  oracle      PSNR of ground-truth frame t-1 moved onto frame t by the vectors, so how
              good a perfect previous output could be as a prediction of frame t
  copied      share of input pixels identical to the previous frame (skipped blocks)
  model       PSNR with and without state by frame index, the warped previous output
              as a predictor of frame t, and the mean gate value
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upstream_vsr.data import WindowDataset  # noqa: E402
from upstream_vsr.model import UpStreamVSR, warp  # noqa: E402


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (-10 * torch.log10((a.clamp(0, 1) - b).pow(2).mean().clamp_min(1e-10))).item()


def input_flow(mv: torch.Tensor, scale: int) -> torch.Tensor:
    """Motion field at input resolution, as the model uses it."""
    return F.interpolate(mv, scale_factor=4, mode='nearest')


def output_flow(mv: torch.Tensor, size, scale: int) -> torch.Tensor:
    return F.interpolate(mv, size=size, mode='nearest') * scale


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('windows', type=Path, help='directory of prepared .npz windows')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = UpStreamVSR(**checkpoint['config']).to(device)
    model.load_checkpoint_state(checkpoint['model'])
    model.eval()

    gates, recording = [], [False]
    model.gate.register_forward_hook(lambda _m, _i, out: recording[0] and gates.append(out.mean().item()))
    dataset = WindowDataset(args.windows, args.frames, train=False)
    report = {'windows': len(dataset)}
    for scale in (2, 4):
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        def add(name: str, value: float) -> None:
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1

        gates.clear()
        for sample in dataset:
            lr = sample[f'lr{scale}'].to(device)
            mv = sample[f'mv{scale}'].to(device)
            intra = sample[f'intra{scale}'].to(device)
            gt = sample['gt'].to(device)
            state, previous_output = None, None
            for t in range(lr.size(0)):
                x, m = lr[t:t + 1], mv[t:t + 1]
                recording[0] = state is not None  # gate is only meaningful with a real past
                with_state, new_state = model.step(x, m, intra[t:t + 1], state, scale)
                recording[0] = False
                without, _ = model.step(x, m, intra[t:t + 1], None, scale)
                add(f'psnr_state_t{t:02d}', psnr(with_state, gt[t:t + 1]))
                add(f'psnr_nostate_t{t:02d}', psnr(without, gt[t:t + 1]))
                if t and not intra[t]:
                    previous = lr[t - 1:t]
                    flow = input_flow(m, scale)
                    add('align_no_motion', (x - previous).abs().mean().item())
                    add('align_mv', (x - warp(previous, flow)).abs().mean().item())
                    add('align_mv_negated', (x - warp(previous, -flow)).abs().mean().item())
                    add('copied_pixels', (x == previous).all(dim=1).float().mean().item())
                    add('mv_nonzero_cells', (m.abs().sum(dim=1) > 0).float().mean().item())
                    hr_flow = output_flow(m, gt.shape[-2:], scale)
                    add('oracle_gt_no_motion', psnr(gt[t - 1:t], gt[t:t + 1]))
                    add('oracle_gt_mv', psnr(warp(gt[t - 1:t], hr_flow), gt[t:t + 1]))
                    add('warped_previous_output', psnr(warp(previous_output, hr_flow), gt[t:t + 1]))
                    add('current_output_nostate', psnr(without, gt[t:t + 1]))
                state, previous_output = new_state, with_state
        result = {name: round(sums[name] / counts[name], 4) for name in sorted(sums)}
        result['gate_mean'] = round(sum(gates) / len(gates), 4)
        report[f'x{scale}'] = result
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + '\n')


if __name__ == '__main__':
    main()
