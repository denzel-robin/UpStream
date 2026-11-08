#!/usr/bin/env python3
"""Train UpStreamVSR. Resumes automatically, so interrupted cloud sessions can continue."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upstream_vsr.data import WindowDataset  # noqa: E402
from upstream_vsr.model import UpStreamVSR  # noqa: E402


def endless(loader):
    """Loop over a DataLoader, reshuffling and re-augmenting each epoch.

    itertools.cycle must not be used here: it replays stored first-epoch batches
    and keeps them all in memory.
    """
    while True:
        yield from loader


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).pow(2) + eps * eps).mean()


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kernel = x.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8
    kernels = torch.stack((kernel, kernel.t())).unsqueeze(1).repeat_interleave(x.size(1), 0)
    return F.conv2d(x, kernels, padding=1, groups=x.size(1))


def choose_exit(num_exits: int) -> int:
    """Always train the cheapest and deepest exits heavily, as they bound the range."""
    roll = random.random()
    if roll < 1 / 3:
        return 0
    if roll < 2 / 3:
        return num_exits - 1
    return random.randrange(num_exits)


def scale_inputs(batch: dict, scale: int, t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = 2 if scale == 2 else 4
    return batch[f'lr{key}'][:, t], batch[f'mv{key}'][:, t], batch[f'intra{key}'][:, t]


def train_sequence(model, batch, args) -> torch.Tensor:
    """Run a window with random exits and, sometimes, one resolution switch."""
    frames = batch['gt'].size(1)
    scale = random.choice((2, 4))
    switch_at = random.randrange(1, frames) if random.random() < args.switch_probability else None
    state, loss = None, 0.0
    for t in range(frames):
        if t == switch_at:
            scale = 6 - scale
        x, mv, intra = scale_inputs(batch, scale, t)
        output, state = model.step(x, mv, intra, state, scale, choose_exit(len(model.exits)))
        target = batch['gt'][:, t]
        loss = loss + charbonnier(output, target)
        if args.edge_weight:
            loss = loss + args.edge_weight * charbonnier(sobel_edges(output), sobel_edges(target))
    return loss / frames


@torch.no_grad()
def validate(model, loader, device, exits_to_check) -> dict[str, float]:
    """Mean RGB PSNR per exit and scale on the same windows, so results are comparable."""
    model.eval()
    results = {}
    for scale in (2, 4):
        for exit_index in exits_to_check:
            errors = []
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                state = None
                for t in range(batch['gt'].size(1)):
                    x, mv, intra = scale_inputs(batch, scale, t)
                    output, state = model.step(x, mv, intra, state, scale, exit_index)
                    mse = (output.clamp(0, 1) - batch['gt'][:, t]).pow(2).mean(dim=(1, 2, 3))
                    errors.append(mse.clamp_min(1e-10))
            psnr = (-10 * torch.log10(torch.cat(errors))).mean().item()
            results[f'x{scale}_exit{exit_index}'] = round(psnr, 4)
    model.train()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, help='prepared dataset root with train/ and val/')
    parser.add_argument('output', type=Path)
    parser.add_argument('--iterations', type=int, default=100_000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-feat', type=int, default=32)
    parser.add_argument('--exits', type=int, nargs='+', default=[3, 6, 9])
    parser.add_argument('--edge-weight', type=float, default=0.1)
    parser.add_argument('--switch-probability', type=float, default=0.3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--val-every', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-hours', type=float, help='save and exit cleanly after this long')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)

    train_set = WindowDataset(args.data / 'train', args.frames, train=True)
    val_set = WindowDataset(args.data / 'val', args.frames, train=False)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=len(train_set) >= args.batch_size,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.workers, pin_memory=True)

    model = UpStreamVSR(args.num_feat, args.exits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    iteration, best = 0, -math.inf
    latest = args.output / 'latest.pt'
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=True)
        if set(checkpoint['model']) == set(model.state_dict()):
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            iteration, best = checkpoint['iteration'], checkpoint['best']
            print(f'resumed from iteration {iteration}', flush=True)
        else:  # older layout: keep the weights, restart the optimizer and schedule
            model.load_checkpoint_state(checkpoint['model'])
            print('initialised from an older checkpoint; optimizer and schedule restart', flush=True)

    def save(path: Path) -> None:
        temporary = path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'iteration': iteration, 'best': best,
                    'config': {'num_feat': args.num_feat, 'exits': args.exits}}, temporary)
        temporary.replace(path)

    log = (args.output / 'log.jsonl').open('a')
    exits_to_check = sorted({0, len(args.exits) - 1})
    session_start = started = time.monotonic()
    model.train()
    for batch in endless(train_loader):
        if iteration >= args.iterations:
            break
        if args.max_hours and time.monotonic() - session_start > args.max_hours * 3600:
            save(latest)
            print(f'time budget reached at iteration {iteration}; saved {latest}', flush=True)
            break
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            loss = train_sequence(model, batch, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        iteration += 1

        if iteration % 50 == 0:
            rate = 50 / (time.monotonic() - started)
            started = time.monotonic()
            print(f'iter {iteration} loss {loss.item():.5f} {rate:.2f} it/s', flush=True)
        if iteration % args.val_every == 0 or iteration == args.iterations:
            metrics = validate(model, val_loader, device, exits_to_check)
            score = sum(metrics.values()) / len(metrics)
            log.write(json.dumps({'iteration': iteration, 'loss': loss.item(), **metrics}) + '\n')
            log.flush()
            print(metrics, flush=True)
            if score > best:
                best = score
                save(args.output / 'best.pt')
            save(latest)


if __name__ == '__main__':
    main()
