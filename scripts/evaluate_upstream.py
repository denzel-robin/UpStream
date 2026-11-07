#!/usr/bin/env python3
"""Score a checkpoint against bilinear and bicubic on prepared windows.

The no_state column runs the same weights with state reset on every frame. It is a
quick signal of whether temporal state helps, not a substitute for ablation: a fair
comparison needs a model trained without state.
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
from upstream_vsr.model import UpStreamVSR  # noqa: E402


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean of per-frame RGB PSNR."""
    mse = (prediction.clamp(0, 1) - target).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return (-10 * torch.log10(mse)).mean().item()


@torch.inference_mode()
def run_model(model, lr, mv, intra, scale, exit_index, use_state) -> torch.Tensor:
    state, outputs = None, []
    for t in range(lr.size(0)):
        output, state = model.step(lr[t:t + 1], mv[t:t + 1], intra[t:t + 1],
                                   state if use_state else None, scale, exit_index)
        outputs.append(output)
    return torch.cat(outputs)


def evaluate_split(model, root: Path, frames: int, device) -> dict:
    dataset = WindowDataset(root, frames, train=False)
    report = {'windows': len(dataset)}
    for scale in (2, 4):
        totals: dict[str, float] = {}

        def add(name: str, value: float) -> None:
            totals[name] = totals.get(name, 0.0) + value

        for sample in dataset:
            lr = sample[f'lr{scale}'].to(device)
            mv = sample[f'mv{scale}'].to(device)
            intra = sample[f'intra{scale}'].to(device)
            gt = sample['gt'].to(device)
            for mode in ('bilinear', 'bicubic'):
                add(mode, psnr(F.interpolate(lr, scale_factor=scale, mode=mode, align_corners=False), gt))
            for index, depth in enumerate(model.exits):
                add(f'blocks{depth}', psnr(run_model(model, lr, mv, intra, scale, index, True), gt))
            add('no_state', psnr(run_model(model, lr, mv, intra, scale, -1, False), gt))
        report[f'x{scale}'] = {name: round(value / len(dataset), 3) for name, value in totals.items()}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--splits', nargs='+', default=['val', 'test'])
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = UpStreamVSR(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    report = {'iteration': checkpoint['iteration']}
    for split in args.splits:
        if any((args.data / split).glob('*.npz')):
            report[split] = evaluate_split(model, args.data / split, args.frames, device)
        else:
            print(f'skipping {split}: no windows', flush=True)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + '\n')


if __name__ == '__main__':
    main()
