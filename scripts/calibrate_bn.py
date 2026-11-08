#!/usr/bin/env python3
"""Recompute BatchNorm running statistics separately for x2 and x4.

Training updates BatchNorm once per frame, so running statistics follow whichever
scale the last few batches used. This replays training windows at one scale at a
time with the weights frozen, taking a plain average over all of them, and writes
a checkpoint whose per-scale statistics match what the weights saw in training.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upstream_vsr.data import WindowDataset  # noqa: E402
from upstream_vsr.model import RepBlock, UpStreamVSR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_upstream import choose_exit, scale_inputs  # noqa: E402


def scale_norms(model: UpStreamVSR, scale: int) -> list[torch.nn.BatchNorm2d]:
    key = f'x{scale}'
    norms = []
    for block in model.modules():
        if isinstance(block, RepBlock) and key in block.bn3:
            norms += [block.bn3[key], block.bn1[key]]
            if block.bn_id is not None:
                norms.append(block.bn_id[key])
    return norms


@torch.no_grad()
def calibrate(model: UpStreamVSR, loader: DataLoader, scale: int, batches: int, device) -> None:
    norms = scale_norms(model, scale)
    model.eval()
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None  # cumulative average instead of an exponential one
        norm.train()
    seen = 0
    while seen < batches:
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            state = None
            for t in range(batch['gt'].size(1)):
                x, mv, intra = scale_inputs(batch, scale, t)
                _, state = model.step(x, mv, intra, state, scale, choose_exit(len(model.exits)))
            seen += 1
            if seen >= batches:
                break
    for norm in norms:
        norm.momentum = 0.1
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, help='prepared dataset root with train/')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--batches', type=int, default=100, help='batches per scale')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = UpStreamVSR(**checkpoint['config']).to(device)
    model.load_checkpoint_state(checkpoint['model'])

    dataset = WindowDataset(args.data / 'train', args.frames, train=True)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    for scale in (2, 4):
        calibrate(model, loader, scale, args.batches, device)
        print(f'calibrated x{scale} on {args.batches * args.batch_size} windows', flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'iteration': checkpoint['iteration'],
                'config': checkpoint['config'], 'calibrated_from': str(args.checkpoint),
                'calibration_windows': args.batches * args.batch_size}, args.output)
    print('wrote', args.output)


if __name__ == '__main__':
    main()
