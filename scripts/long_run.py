#!/usr/bin/env python3
"""Long run: does UpStreamVSR's recurrent state stay healthy over a long stream?

The model carries its state from frame to frame and never resets it, not even at
keyframes (the I-frame flag only feeds the gate). A slow drift in that state would
show up only in long streams. This run feeds one continuous stream to the model,
the 18 test clips (30 s each, 9 minutes) played twice in a row (18 minutes,
32,400 frames), without ever resetting the state:

* x4@23 and x2@23, encoded with the live x264 settings (keyframe every 30 frames),
  each with its own state, in float16, deepest exit, as the live client runs them;
* bicubic on the same frames, so the model's gain is measured against a reference
  that has no memory;
* per frame: PSNR against the 720p reference, the model's time, and the size of the
  recurrent state (largest magnitude, mean magnitude, any NaN or inf).

Collapse would show as a falling gain over time, a growing state, non-finite values,
or a lower gain in the second pass than in the first on the very same content.

  bash scripts/make_long_source.sh
  python scripts/long_run.py --out results/long
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from live.client import load_model  # noqa: E402
from upstream_vsr.codec import decode_with_motion  # noqa: E402
from upstream_vsr.encoding import X264, ladder_graph  # noqa: E402
from upstream_vsr.model import crf_to_quality  # noqa: E402

FPS = 30
WINDOW = 30 * FPS  # one clip


def encode(once: Path, twice: Path, work: Path, crf: int) -> dict[str, Path]:
    """The x2 and x4 streams of the 18-minute source, and the 720p reference of the 9-minute one."""
    work.mkdir(parents=True, exist_ok=True)
    paths = {'x2': work / f'x2@{crf}.mp4', 'x4': work / f'x4@{crf}.mp4', 'reference': work / 'reference.mkv'}
    if not all(p.exists() for p in paths.values()):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(twice), '-filter_complex',
                        ladder_graph(FPS, {'lr2': 1, 'lr4': 1}),
                        '-map', '[lr2_0]', *X264, '-crf', str(crf), str(paths['x2']),
                        '-map', '[lr4_0]', *X264, '-crf', str(crf), str(paths['x4'])], check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(once), '-filter_complex',
                        ladder_graph(FPS, {'gt': 1}), '-map', '[gt_0]', '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-qp', '0', '-colorspace', 'bt709', '-color_range', 'tv', str(paths['reference'])], check=True)
    return paths


def reference_frames(path: Path, passes: int):
    """The lossless reference, converted as the training data was, played `passes` times."""
    for _ in range(passes):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            for frame in container.decode(stream):
                yield frame.reformat(format='rgb24', src_colorspace='ITU709').to_ndarray()


def psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    error = (output[0].float().clamp(0, 1).mul(255).round() - target.float()).pow(2).mean().clamp_min(1e-10)
    return float(10 * torch.log10(255 ** 2 / error))


def gpu_reading() -> dict:
    fields = 'temperature.gpu,power.draw,memory.used'
    text = subprocess.run(['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True).stdout.strip().split(', ')
    try:
        return {'temp_c': float(text[0]), 'power_w': float(text[1]), 'memory_mib': float(text[2])}
    except (ValueError, IndexError):
        return {}


def summarise(frames: dict, gpu: list[dict], clips: int) -> dict:
    windows = []
    count = len(frames['x4']['model'])
    for w in range(count // WINDOW):
        part = slice(w * WINDOW, (w + 1) * WINDOW)
        entry = {'window': w, 'pass': w // clips + 1, 'clip': w % clips, **(gpu[w] if w < len(gpu) else {})}
        for scale in ('x2', 'x4'):
            f = frames[scale]
            model, bicubic = np.array(f['model'][part]), np.array(f['bicubic'][part])
            entry[scale] = {'psnr': float(model.mean()), 'bicubic': float(bicubic.mean()),
                            'gain': float((model - bicubic).mean()),
                            'state_max': float(np.max(f['state_max'][part])),
                            'state_mean': float(np.mean(f['state_mean'][part])),
                            'nonfinite': int(np.sum(f['nonfinite'][part])),
                            'ms_median': float(np.median(f['ms'][part])), 'ms_max': float(np.max(f['ms'][part]))}
        windows.append(entry)
    result = {'frames': count, 'minutes': count / FPS / 60, 'windows': windows}
    for scale in ('x2', 'x4'):
        gains = np.array([w[scale]['gain'] for w in windows])
        first = [w[scale]['gain'] for w in windows if w['pass'] == 1]
        second = [w[scale]['gain'] for w in windows if w['pass'] == 2]
        paired = [b - a for a, b in zip(first, second)]
        slope = float(np.polyfit(np.arange(len(gains)) * 0.5, gains, 1)[0]) if len(gains) > 1 else 0.0
        result[scale] = {
            'psnr': float(np.mean(frames[scale]['model'])), 'bicubic': float(np.mean(frames[scale]['bicubic'])),
            'gain': float(gains.mean()) if len(gains) else 0.0, 'gain_min_window': float(gains.min()) if len(gains) else 0.0,
            'gain_slope_db_per_min': slope,
            'second_minus_first_pass_db': {'mean': float(np.mean(paired)), 'min': float(np.min(paired)),
                                           'max': float(np.max(paired)), 'clips': len(paired)} if paired else None,
            'state_max_first_window': windows[0][scale]['state_max'] if windows else None,
            'state_max_last_window': windows[-1][scale]['state_max'] if windows else None,
            'state_max_overall': float(np.max(frames[scale]['state_max'])),
            'nonfinite_frames': int(np.sum(frames[scale]['nonfinite'])),
        }
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/kaggle-6-v3/run/best-calibrated.pt')
    parser.add_argument('--once', type=Path, default=ROOT / 'data/long/once.mp4')
    parser.add_argument('--twice', type=Path, default=ROOT / 'data/long/twice.mp4')
    parser.add_argument('--crf', type=int, default=23)
    parser.add_argument('--clips', type=int, default=18, help='clips in one pass')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/long')
    args = parser.parse_args()
    device = torch.device('cuda')
    args.out.mkdir(parents=True, exist_ok=True)

    begin = time.time()
    paths = encode(args.once, args.twice, ROOT / 'data/long/encoded', args.crf)
    print(f'encoded in {time.time() - begin:.0f} s', flush=True)

    model = load_model(args.checkpoint, device, half=True)
    exit_index = len(model.exits) - 1
    quality = crf_to_quality(args.crf).view(1).to(device, torch.float16)
    frames = {s: {k: [] for k in ('model', 'bicubic', 'ms', 'state_max', 'state_mean', 'nonfinite')}
              for s in ('x2', 'x4')}
    states = {'x2': None, 'x4': None}
    gpu = []
    streams = zip(decode_with_motion(paths['x2']), decode_with_motion(paths['x4']),
                  reference_frames(paths['reference'], 2))
    begin = time.time()
    for t, (x2, x4, reference) in enumerate(streams):
        target = torch.from_numpy(reference).to(device).permute(2, 0, 1)
        for scale_name, frame, scale in (('x2', x2, 2), ('x4', x4, 4)):
            x = torch.from_numpy(frame.rgb).to(device).permute(2, 0, 1)[None].half() / 255
            mv = torch.from_numpy(frame.mv).to(device).permute(2, 0, 1)[None].half()
            intra = torch.tensor([float(frame.is_intra)], device=device, dtype=torch.float16)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out, states[scale_name] = model.step(x, mv, intra, states[scale_name], scale, exit_index, quality=quality)
            torch.cuda.synchronize()
            f = frames[scale_name]
            f['ms'].append((time.perf_counter() - start) * 1000)
            f['model'].append(psnr(out, target))
            f['bicubic'].append(psnr(F.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=False),
                                     target))
            hidden = states[scale_name].hidden.float()
            f['state_max'].append(float(hidden.abs().max()))
            f['state_mean'].append(float(hidden.abs().mean()))
            f['nonfinite'].append(int(not torch.isfinite(hidden).all() or not torch.isfinite(out).all()))
        if (t + 1) % WINDOW == 0:
            gpu.append(gpu_reading())
            w = (t + 1) // WINDOW - 1
            part = slice(w * WINDOW, (w + 1) * WINDOW)
            gain = {s: np.mean(np.array(frames[s]['model'][part]) - np.array(frames[s]['bicubic'][part]))
                    for s in frames}
            print(f"[{(t + 1) / FPS / 60:5.1f} min of stream, {time.time() - begin:4.0f} s] "
                  f"gain over bicubic x2 {gain['x2']:+.2f} x4 {gain['x4']:+.2f} dB | "
                  f"state max x2 {max(frames['x2']['state_max'][part]):.2f} x4 {max(frames['x4']['state_max'][part]):.2f} | "
                  f"non-finite {sum(frames['x2']['nonfinite']) + sum(frames['x4']['nonfinite'])} | "
                  f"x4 {np.median(frames['x4']['ms'][part]):.1f} ms | {gpu[-1].get('temp_c', '?')} °C", flush=True)
            summary = summarise(frames, gpu, args.clips)
            summary.update({'checkpoint': str(args.checkpoint), 'crf': args.crf, 'gpu': torch.cuda.get_device_name(),
                            'wall_seconds': time.time() - begin})
            (args.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    np.savez_compressed(args.out / 'frames.npz', **{f'{s}_{k}': np.array(v) for s in frames for k, v in frames[s].items()})
    print(json.dumps({s: {k: v for k, v in summary[s].items()} for s in ('x2', 'x4')}, indent=1))


if __name__ == '__main__':
    main()
