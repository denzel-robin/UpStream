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
import html
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get('KAGGLE_ROOT', '/kaggle'))
INPUT, WORKING, TEMP = ROOT / 'input', ROOT / 'working', ROOT / 'temp'

# File IDs from the README of github.com/EvgeneyBogatyrev/EfRLFN
TRAIN_ARCHIVE = ('1yPSpvyIn-g1xywUmZXeurbUCHD5ptNE5', 'youtube-train.tar.gz')
TEST_ARCHIVE = ('1CWqsZxyn31_m67n8TFx8K-Nt9YzreGfj', 'subjective-set.tar.gz')


def run(*command: object) -> None:
    subprocess.run([str(part) for part in command], check=True)


def open_range(url: str, start: int):
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-', 'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(request, timeout=60)
    if 'text/html' not in response.headers.get('Content-Type', ''):
        return response
    # Google Drive answers large files with a confirmation form instead of the data.
    page = response.read(1_000_000).decode('utf-8', 'replace')
    fields = dict(re.findall(r'name="(\w+)" value="([^"]*)"', page))
    action = re.search(r'action="([^"]+)"', page)
    if not action or 'confirm' not in fields:
        title = re.search(r'<title>(.*?)</title>', page)
        raise PermissionError(f'server sent a page, not the file: {title.group(1) if title else page[:200]}')
    target = html.unescape(action.group(1)) + '?' + urllib.parse.urlencode(fields)
    request = urllib.request.Request(target, headers={'Range': f'bytes={start}-', 'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(request, timeout=60)
    if 'text/html' in response.headers.get('Content-Type', ''):
        raise PermissionError('server sent a page after confirmation; the download may be over quota')
    return response


def download(url: str, destination: Path, max_stalls: int = 8, retry_wait: float = 15) -> None:
    """Download with retries, continuing from the bytes already received.

    Only attempts that gain no data count toward max_stalls, so a long download
    can survive any number of dropped connections as long as it keeps progressing.
    """
    partial = destination.with_name(destination.name + '.part')
    stalls, next_report = 0, 0
    while True:
        position = partial.stat().st_size if partial.exists() else 0
        total = None
        try:
            with open_range(url, position) as response:
                if response.status == 206:
                    total = int(response.headers['Content-Range'].rsplit('/', 1)[1])
                else:  # the server ignored the range, so start over
                    position = 0
                    total = int(response.headers.get('Content-Length') or 0) or None
                with partial.open('r+b' if position else 'wb') as file:
                    file.seek(position)
                    while chunk := response.read(1 << 20):
                        file.write(chunk)
                        position += len(chunk)
                        stalls = 0
                        if position >= next_report:
                            print(f'{destination.name}: {position / 1e9:.2f} of '
                                  f'{(total or 0) / 1e9:.2f} GB', flush=True)
                            next_report = position + 1_000_000_000
            if total is None or position == total:
                partial.replace(destination)
                return
            error: Exception = ConnectionError(f'connection closed at {position} of {total} bytes')
        except urllib.error.HTTPError as http_error:
            if http_error.code == 416 and position:  # asked for bytes past the end: already complete
                partial.replace(destination)
                return
            error = http_error
        except (OSError, TimeoutError) as os_error:
            if isinstance(os_error, PermissionError):
                raise
            error = os_error
        stalls += 1
        if stalls > max_stalls:
            raise SystemExit(f'Download of {destination.name} made no progress after {max_stalls} '
                             f'retries: {error}. Google Drive may be limiting this file; try again later.')
        print(f'Retrying {destination.name} from {position} bytes ({error})', flush=True)
        time.sleep(retry_wait)


def fetch_clips(file_id: str, name: str, destination: Path) -> Path:
    """Download an archive unless already present and extract only the 1440p clips."""
    archive = TEMP / name
    if not archive.exists():
        download(f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t', archive)
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
