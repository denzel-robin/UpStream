#!/usr/bin/env python3
"""Turn StreamSR clips into training windows that match the live pipeline.

Each usable <id>@1440.mp4 is downscaled to the 720p target, then encoded to 180p
(x4 input) and 360p (x2 input) with the same x264 settings the live server uses.
The re-encode matters: the model must learn the artifacts and I/P-only motion
structure of our stream, whereas YouTube's native renditions contain B-frames.

Clips are assigned to train or val by a hash of the video ID, so no video appears
in both. Use --split test for the subjective set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upstream_vsr.codec import decode_with_motion  # noqa: E402

CRFS = (18, 23, 28, 33)
X264 = ['-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency', '-threads', '1',
        '-x264-params', 'keyint=30:min-keyint=30:scenecut=0:bframes=0:ref=1',
        '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
        '-color_trc', 'bt709', '-color_range', 'tv']


def probe(path: Path) -> tuple[int, int, float] | None:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height,r_frame_rate', '-of', 'json', str(path)],
        capture_output=True, text=True)
    streams = json.loads(result.stdout or '{}').get('streams') if result.returncode == 0 else None
    if not streams:
        return None
    stream = streams[0]
    return stream['width'], stream['height'], float(Fraction(stream['r_frame_rate']))


def encode(source: Path, workdir: Path, crf: int, fps: float, seconds: float) -> dict[str, Path]:
    # 45+ fps sources are halved so that frame spacing resembles the 30 fps live stream.
    step = 'framestep=2,' if fps >= 45 else ''
    scale = ':flags=area+accurate_rnd:out_color_matrix=bt709:out_range=tv'
    # The second scale must be told its input is BT.709. Older ffmpeg (as on Kaggle)
    # does not carry the first scale's output matrix forward and assumes BT.601,
    # which shifts the colours of the low-resolution inputs but not the ground truth.
    rescale = scale + ':in_color_matrix=bt709:in_range=tv'
    paths = {name: workdir / f'{name}.{ext}' for name, ext in
             (('gt', 'nut'), ('lr2', 'mp4'), ('lr4', 'mp4'))}
    graph = (f'[0:v]{step}scale=1280:720{scale},format=yuv444p,split=3[gt][a][b];'
             f'[a]scale=640:360{rescale}[lr2];[b]scale=320:180{rescale}[lr4]')
    command = ['ffmpeg', '-v', 'error', '-y', '-t', str(seconds), '-i', str(source),
               '-filter_complex', graph,
               '-map', '[gt]', '-c:v', 'ffv1', '-colorspace', 'bt709', '-color_range', 'tv', str(paths['gt']),
               '-map', '[lr2]', *X264, '-crf', str(crf), str(paths['lr2']),
               '-map', '[lr4]', *X264, '-crf', str(crf), str(paths['lr4'])]
    subprocess.run(command, check=True)
    return paths


def check_colour(tolerance: float = 2.0) -> None:
    """Fail fast if this ffmpeg makes the inputs' colours differ from the ground truth.

    A saturated solid colour goes through the real filter graph. Any colour-matrix
    or range mismatch between the renditions moves it by several levels.
    """
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / 'colour.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=0xC03020:s=1920x1080:r=30',
                        '-t', '1', '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p',
                        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                        '-color_range', 'tv', str(source)], check=True)
        paths = encode(source, Path(temp), 18, 30, 1)
        means = {name: np.mean([frame.rgb.reshape(-1, 3).mean(0) for frame in decode_with_motion(path)], axis=0)
                 for name, path in paths.items()}
    for name in ('lr2', 'lr4'):
        error = np.abs(means[name] - means['gt']).max()
        if error > tolerance:
            raise SystemExit(f'{name} colours differ from the ground truth by {error:.1f} levels '
                             f'(gt {means["gt"].round(1)}, {name} {means[name].round(1)}); '
                             'this ffmpeg build would produce mismatched training data')
    print('colour check passed', {name: value.round(1).tolist() for name, value in means.items()}, flush=True)


def choose_windows(total: int, length: int, count: int, rng: random.Random) -> list[int]:
    if total < length:
        return []
    starts = list(range(0, total - length + 1, length))
    rng.shuffle(starts)
    return sorted(starts[:count])


def crop_position(frame: np.ndarray, patch: int, rng: random.Random, tries: int = 8) -> tuple[int, int]:
    """Pick the most textured of a few random crops, aligned to the 16 px GT cell grid."""
    h, w = frame.shape[:2]
    best, best_score = (0, 0), -1.0
    for _ in range(tries):
        x = rng.randrange(0, (w - patch) // 16 + 1) * 16
        y = rng.randrange(0, (h - patch) // 16 + 1) * 16
        score = float(frame[y:y + patch, x:x + patch].std())
        if score > best_score:
            best, best_score = (x, y), score
    return best


def extract(paths: dict[str, Path], length: int, count: int, patch: int, rng: random.Random) -> list[dict]:
    from itertools import zip_longest

    frames = zip_longest(*(decode_with_motion(paths[k]) for k in ('gt', 'lr2', 'lr4')))
    # Decode the whole clip once to find the frame count before choosing windows.
    decoded_count = sum(1 for _ in decode_with_motion(paths['lr4']))
    starts = set(choose_windows(decoded_count, length, count, rng))
    windows, current, position = [], None, None
    for index, triple in enumerate(frames):
        if any(item is None for item in triple):
            raise ValueError('renditions have different frame counts')
        gt, lr2, lr4 = triple
        if index in starts:
            position = crop_position(gt.rgb, patch, rng)
            current = {key: [] for key in ('gt', 'lr2', 'lr4', 'mv2', 'mv4', 'intra2', 'intra4')}
        if current is None:
            continue
        x, y = position
        current['gt'].append(gt.rgb[y:y + patch, x:x + patch])
        current['lr2'].append(lr2.rgb[y // 2:(y + patch) // 2, x // 2:(x + patch) // 2])
        current['lr4'].append(lr4.rgb[y // 4:(y + patch) // 4, x // 4:(x + patch) // 4])
        current['mv2'].append(lr2.mv[y // 8:(y + patch) // 8, x // 8:(x + patch) // 8])
        current['mv4'].append(lr4.mv[y // 16:(y + patch) // 16, x // 16:(x + patch) // 16])
        current['intra2'].append(lr2.is_intra)
        current['intra4'].append(lr4.is_intra)
        if len(current['gt']) == length:
            window = {key: np.stack(value) for key, value in current.items()}
            if is_informative(window['gt']):
                windows.append(window)
            current = None
    return windows


def is_informative(gt: np.ndarray, min_std: float = 10.0, min_motion: float = 0.3) -> bool:
    """Reject blank or frozen windows: they teach neither detail nor temporal alignment."""
    frames = gt.astype(np.float32)
    return float(frames.std()) >= min_std and float(np.abs(frames[1:] - frames[:-1]).mean()) >= min_motion


def split_for(clip_id: str, requested: str | None) -> str:
    if requested:
        return requested
    return 'val' if int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % 11 == 0 else 'train'


def process_clip(job: tuple[Path, argparse.Namespace]) -> dict:
    source, args = job
    clip_id = source.name.split('@')[0]
    rng = random.Random(f'{args.seed}:{clip_id}')
    entry = {'clip': clip_id, 'status': 'ok', 'windows': 0}
    info = probe(source)
    # Vertical or non-16:9 clips would be distorted by the fixed 720p target.
    if info is None:
        entry['status'] = 'unreadable'
    elif abs(info[0] / info[1] - 16 / 9) > .02:
        entry['status'] = 'not-16:9'
    elif info[2] < 23:
        entry['status'] = 'low-fps'
    else:
        crf = rng.choice(CRFS)
        entry.update(split=split_for(clip_id, args.split), crf=crf, fps=info[2])
        try:
            with tempfile.TemporaryDirectory() as temp:
                paths = encode(source, Path(temp), crf, info[2], args.seconds)
                windows = extract(paths, args.frames, args.windows_per_clip, args.patch, rng)
        except (subprocess.CalledProcessError, ValueError, OSError, av.error.FFmpegError) as error:
            entry['status'] = f'failed: {type(error).__name__}'
        else:
            target = args.output / entry['split']
            target.mkdir(exist_ok=True)
            for number, window in enumerate(windows):
                np.savez_compressed(target / f'{clip_id}-{number}.npz', crf=crf, **window)
            entry['windows'] = len(windows)
            if not windows:
                entry['status'] = 'no-informative-windows'
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='directory containing <id>@1440.mp4 files')
    parser.add_argument('output', type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], help='force one split (default: by hash)')
    parser.add_argument('--windows-per-clip', type=int, default=2)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--patch', type=int, default=256, help='GT patch size; must be a multiple of 16')
    parser.add_argument('--seconds', type=float, default=6)
    parser.add_argument('--limit', type=int, help='process at most this many clips')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--max-hours', type=float, help='stop cleanly after this long')
    args = parser.parse_args()
    if args.patch % 16:
        parser.error('--patch must be a multiple of 16')
    check_colour()

    clips = sorted(args.source.glob('*@1440.mp4'))[:args.limit]
    manifest = args.output / 'manifest.jsonl'
    args.output.mkdir(parents=True, exist_ok=True)
    done = set()
    if manifest.exists():
        done = {json.loads(line)['clip'] for line in manifest.read_text().splitlines()}

    jobs = [(source, args) for source in clips if source.name.split('@')[0] not in done]
    counts = {'windows': 0, 'skipped': 0, 'remaining': len(jobs)}
    started = time.monotonic()
    with manifest.open('a') as log, multiprocessing.Pool(args.workers) as pool:
        for entry in pool.imap_unordered(process_clip, jobs):
            counts['windows'] += entry['windows']
            counts['skipped'] += entry['status'] != 'ok'
            counts['remaining'] -= 1
            log.write(json.dumps(entry) + '\n')
            log.flush()
            print(entry, flush=True)
            if args.max_hours and time.monotonic() - started > args.max_hours * 3600:
                print('time budget reached; rerun to continue from the manifest', flush=True)
                pool.terminate()
                break
    print(counts)


if __name__ == '__main__':
    main()
