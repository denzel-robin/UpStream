#!/usr/bin/env python3
"""Estimate how much work tile skipping saves and what it costs in quality.

A tile is reused from the previous output when, on a P-frame, every input pixel in the
tile and in a halo the size of the model's receptive field is identical to the previous
frame and every motion vector there is zero. Then the codec copied the region exactly,
so the previous output is still valid for it. Everything else is recomputed.

This is a simulation: the full frame is still computed and the outputs are composited,
so it measures quality and the share of work that would remain, not real speed.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upstream_vsr.codec import decode_with_motion  # noqa: E402
from upstream_vsr.data import WindowDataset  # noqa: E402
from upstream_vsr.model import UpStreamVSR, crf_to_quality  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_streamsr import encode, probe  # noqa: E402

# Grid pixels each output pixel depends on in each direction: entry 3x3, nine 3x3
# blocks and the 3x3 head.
HALO = 11


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (-10 * torch.log10((a.clamp(0, 1) - b).pow(2).mean().clamp_min(1e-10))).item()


def changed_on_grid(x: torch.Tensor, previous: torch.Tensor, mv: torch.Tensor, scale: int) -> torch.Tensor:
    """(1, 1, h, w) bool on the feature grid: True where anything changed."""
    pixels = (x != previous).any(dim=1, keepdim=True).float()
    if scale == 2:
        pixels = F.max_pool2d(pixels, 2)
    motion = F.interpolate((mv.abs().sum(dim=1, keepdim=True) > 0).float(), size=pixels.shape[-2:], mode='nearest')
    return (pixels + motion) > 0


def window_sequences(root: Path, frames: int, scale: int, device):
    for sample in WindowDataset(root, frames, train=False):
        yield sample[f'lr{scale}'], sample[f'mv{scale}'], sample[f'intra{scale}'], sample['gt'], sample['crf']


def clip_sequences(clips: list[Path], seconds: float, crfs: list[int], scale: int, device):
    """Whole frames, encoded with the live settings, decoded one frame at a time."""
    for source in clips:
        info = probe(source)
        if info is None:
            continue
        for crf in crfs:
            with tempfile.TemporaryDirectory() as temp:
                paths = encode(source, Path(temp), crf, info[2], seconds)
                gts = [frame.rgb for frame in decode_with_motion(paths['gt'])]  # same colour conversion as training
                decoded = list(decode_with_motion(paths[f'lr{scale}']))
            n = min(len(gts), len(decoded))
            # Kept on the CPU as uint8; frames move to the GPU one at a time.
            to = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
            lr = torch.stack([to(d.rgb) for d in decoded[:n]]).permute(0, 3, 1, 2)
            mv = torch.stack([to(d.mv) for d in decoded[:n]]).permute(0, 3, 1, 2).float()
            intra = torch.tensor([float(d.is_intra) for d in decoded[:n]])
            gt = torch.stack([to(g) for g in gts[:n]]).permute(0, 3, 1, 2)
            yield lr, mv, intra, gt, torch.tensor(float(crf))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='directory of .npz windows, or of <id>@1440.mp4 clips with --clips')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--clips', action='store_true', help='simulate on whole frames of the source clips')
    parser.add_argument('--seconds', type=float, default=6)
    parser.add_argument('--crfs', type=int, nargs='+', default=[23, 33])
    parser.add_argument('--tiles', type=int, nargs='+', default=[8, 16, 32], help='tile sizes in grid pixels')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = UpStreamVSR(**checkpoint['config']).to(device)
    model.load_checkpoint_state(checkpoint['model'])
    model.eval()
    report = {'source': str(args.source), 'clips': args.clips, 'halo_grid_pixels': HALO}
    for scale in (2, 4):
        totals: dict[str, float] = {}
        frames = 0

        def add(name: str, value: float) -> None:
            totals[name] = totals.get(name, 0.0) + value

        if args.clips:
            sequences = clip_sequences(sorted(args.source.glob('*@1440.mp4')), args.seconds, args.crfs, scale, device)
        else:
            sequences = window_sequences(args.source, args.frames, scale, device)
        for lr_all, mv_all, intra_all, gt_all, crf in sequences:
            quality = crf_to_quality(crf).view(1).to(device)
            def frame(tensor, t):
                value = tensor[t:t + 1].to(device)
                return value.float() / 255 if tensor.dtype == torch.uint8 else value
            state, composite = None, {tile: None for tile in args.tiles}
            for t in range(lr_all.size(0)):
                x, m, gt_t = frame(lr_all, t), frame(mv_all, t), frame(gt_all, t)
                is_intra = bool(intra_all[t])
                full, state = model.step(x, m, intra_all[t:t + 1].to(device), state, scale, quality=quality)
                frames += 1
                add('psnr_full', psnr(full, gt_t))
                if t == 0 or is_intra:
                    changed = torch.ones_like(full[:, :1, ::4, ::4], dtype=torch.bool)
                else:
                    changed = changed_on_grid(x, previous_x, m, scale)
                add('changed_grid_pixels', changed.float().mean().item())
                # Dilating by the halo gives every output pixel whose inputs changed.
                affected = F.max_pool2d(changed.float(), 2 * HALO + 1, 1, HALO) > 0
                add('affected_grid_pixels', affected.float().mean().item())
                previous_x = x
                for tile in args.tiles:
                    h, w = affected.shape[-2:]
                    tiles = F.adaptive_max_pool2d(affected.float(), (-(-h // tile), -(-w // tile))) > 0
                    add(f'tile{tile}_recomputed', tiles.float().mean().item())
                    # Work includes each recomputed tile's halo; neighbouring halos overlap,
                    # so this is an upper bound for a gather-batch-scatter implementation.
                    add(f'tile{tile}_work_upper', min(1.0, tiles.float().mean().item() * ((tile + 2 * HALO) / tile) ** 2))
                    mask = F.interpolate(tiles.float(), size=full.shape[-2:], mode='nearest')
                    previous = composite[tile]
                    output = full if previous is None else mask * full + (1 - mask) * previous
                    composite[tile] = output
                    add(f'tile{tile}_psnr', psnr(output, gt_t))
                    add(f'tile{tile}_psnr_vs_full', psnr(output, full.clamp(0, 1)))
        report[f'x{scale}'] = {name: round(value / frames, 4) for name, value in sorted(totals.items())}
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + '\n')


if __name__ == '__main__':
    main()
