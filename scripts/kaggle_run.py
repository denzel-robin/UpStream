#!/usr/bin/env python3
"""Kaggle entry points; the notebooks in kaggle/ only clone the repo and call this.

prepare: download StreamSR, build windows, write one tar to the notebook output.
         A tar is used because Kaggle limits how many output files a notebook keeps.
train:   unpack the attached windows, resume from any attached checkpoint, train
         until the time budget, then score the result.

Session limits are deliberately conservative: an output is only kept if the
notebook finishes on time, so the budget stays well below Kaggle's limit.

KAGGLE_ROOT overrides /kaggle so this can be tested on another machine.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get('KAGGLE_ROOT', '/kaggle'))
INPUT, WORKING, TEMP = ROOT / 'input', ROOT / 'working', ROOT / 'temp'

# File IDs from the README of github.com/EvgeneyBogatyrev/EfRLFN
TRAIN_ARCHIVE = ('1yPSpvyIn-g1xywUmZXeurbUCHD5ptNE5', 'youtube-train.tar.gz')
TEST_ARCHIVE = ('1CWqsZxyn31_m67n8TFx8K-Nt9YzreGfj', 'subjective-set.tar.gz')


def run(*command: object) -> None:
    subprocess.run([str(part) for part in command], check=True)


def fetch_clips(file_id: str, name: str, destination: Path) -> Path:
    """Download an archive unless already present and extract only the 1440p clips."""
    archive = TEMP / name
    if not archive.exists():
        run(sys.executable, '-m', 'gdown', file_id, '-O', archive)
    destination.mkdir(parents=True, exist_ok=True)
    run('tar', 'xzf', archive, '-C', destination, '--wildcards', '*@1440.mp4', '--strip-components=1')
    archive.unlink()  # free disk before the much slower preparation step
    return destination


def has_x264_ffmpeg() -> bool:
    if not (shutil.which('ffmpeg') and shutil.which('ffprobe')):
        return False
    encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True)
    return 'libx264' in encoders.stdout


def ensure_ffmpeg() -> None:
    """Fail in the first minute, not after a 12 GB download."""
    if not has_x264_ffmpeg():
        run('apt-get', 'install', '-y', '-qq', 'ffmpeg')
    if not has_x264_ffmpeg():
        raise SystemExit('ffmpeg with libx264 is required and could not be installed.')


def prepare(args: argparse.Namespace) -> None:
    ensure_ffmpeg()
    TEMP.mkdir(parents=True, exist_ok=True)
    WORKING.mkdir(parents=True, exist_ok=True)
    output = TEMP / 'streamsr'
    script = REPO / 'scripts/prepare_streamsr.py'

    # Test clips go first: the manifest then stops them entering train if IDs overlap.
    test_clips = fetch_clips(*TEST_ARCHIVE, TEMP / 'test-clips')
    run(sys.executable, script, test_clips, output, '--split', 'test', '--windows-per-clip', 4)
    shutil.rmtree(test_clips)

    train_clips = fetch_clips(*TRAIN_ARCHIVE, TEMP / 'train-clips')
    command = [sys.executable, script, train_clips, output,
               '--windows-per-clip', 2, '--max-hours', args.max_hours]
    if args.limit:
        command += ['--limit', args.limit]
    run(*command)

    # Plain tar: the windows are already compressed.
    with tarfile.open(WORKING / 'streamsr-windows.tar', 'w') as tar:
        tar.add(output, arcname='streamsr')
    for split in ('train', 'val', 'test'):
        print(split, 'windows:', len(list((output / split).glob('*.npz'))))
    print('Created', WORKING / 'streamsr-windows.tar')


def find_data() -> Path:
    """Locate attached windows. Kaggle may keep the tar or unpack it when attached."""
    for marker in sorted(INPUT.rglob('manifest.jsonl')):
        if (marker.parent / 'train').is_dir():
            return marker.parent
    tars = sorted(INPUT.rglob('streamsr-windows.tar'))
    if not tars:
        raise SystemExit(f'No prepared data under {INPUT}. Attach the output of the prepare notebook.')
    TEMP.mkdir(parents=True, exist_ok=True)
    data = TEMP / 'streamsr'
    if not data.exists():
        run('tar', 'xf', tars[0], '-C', TEMP)
    return data


def latest_checkpoint() -> Path | None:
    """Pick the most advanced checkpoint attached from an earlier session."""
    import torch

    best, best_iteration = None, -1
    for path in INPUT.rglob('latest.pt'):
        iteration = torch.load(path, map_location='cpu', weights_only=True)['iteration']
        if iteration > best_iteration:
            best, best_iteration = path, iteration
    return best


def check_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit('No GPU. In the notebook sidebar choose Accelerator: GPU T4 x2.')
    try:
        torch.ones(1).cuda() + 1
    except RuntimeError as error:
        raise SystemExit(f'This PyTorch build cannot use {torch.cuda.get_device_name(0)}. '
                         f'Choose Accelerator: GPU T4 x2. ({error})')
    print('GPU:', torch.cuda.get_device_name(0))


def train(args: argparse.Namespace) -> None:
    check_gpu()
    data = find_data()
    run_dir = WORKING / 'run'
    run_dir.mkdir(parents=True, exist_ok=True)
    previous = latest_checkpoint()
    if previous:
        print('Resuming from', previous)
        for name in ('latest.pt', 'best.pt', 'log.jsonl'):
            if (previous.parent / name).exists():
                shutil.copy(previous.parent / name, run_dir / name)
    command = [sys.executable, REPO / 'scripts/train_upstream.py', data, run_dir,
               '--max-hours', args.max_hours, '--iterations', args.iterations,
               '--batch-size', args.batch_size, '--frames', args.frames]
    if args.val_every:
        command += ['--val-every', args.val_every]
    run(*command)
    best = run_dir / 'best.pt'
    if best.exists():
        run(sys.executable, REPO / 'scripts/evaluate_upstream.py', data, best, '--report', run_dir / 'evaluation.json')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--limit', type=int, default=2000, help='clips to process; 0 means all')
    prep.add_argument('--max-hours', type=float, default=8)
    fit = commands.add_parser('train')
    fit.add_argument('--max-hours', type=float, default=8)
    fit.add_argument('--iterations', type=int, default=100_000)
    fit.add_argument('--batch-size', type=int, default=8)
    fit.add_argument('--frames', type=int, default=12)
    fit.add_argument('--val-every', type=int)
    args = parser.parse_args()
    {'prepare': prepare, 'train': train}[args.command](args)


if __name__ == '__main__':
    main()
