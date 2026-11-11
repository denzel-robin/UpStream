#!/usr/bin/env python3
"""Measure quality against bitrate for every rung of the delivery ladder.

A rung is a rendition and a CRF: direct (720p sent as is), x2 (360p upscaled) or
x4 (180p upscaled). Each test clip is encoded with the live settings, decoded with
motion vectors, upscaled causally on full frames by UpStreamVSR at every exit and by
bicubic, and scored against the 720p reference.

x2 and x4 use the CRFs the model was trained on. Direct gets more CRFs, down to low
rates, because the fair comparison is against direct delivery at the same bitrate.

Writes a JSON table with per-clip and mean bitrate and PSNR per rung, and BD-rate
against direct delivery. live/controller.py reads its 'rungs' entry.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
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
from upstream_vsr.encoding import X264, frame_step, ladder_graph  # noqa: E402
from upstream_vsr.model import crf_to_quality  # noqa: E402
from prepare_streamsr import CRFS, probe  # noqa: E402

DIRECT_CRFS = (18, 23, 28, 33, 38, 43)
SCALE = {'x2': 2, 'x4': 4}


def rung_name(rendition: str, crf: int) -> str:
    return f'{rendition}@{crf}'


def payload_kbps(path: Path, fps: float) -> float:
    """Encoded payload only, without container overhead: what the live server sends."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        sizes = [packet.size for packet in container.demux(stream) if packet.size]
    return sum(sizes) * 8 / 1000 / (len(sizes) / fps)


def to_tensor(frames: list[np.ndarray], device) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T, 3, H, W) uint8 on the device."""
    return torch.from_numpy(np.stack(frames)).to(device).permute(0, 3, 1, 2)


def frame_psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """RGB PSNR of one (1, 3, H, W) frame in [0, 1] against a (3, H, W) uint8 target."""
    error = (output[0].clamp(0, 1).mul(255).round() - target.float()).pow(2).mean().clamp_min(1e-10)
    return float(10 * torch.log10(255 ** 2 / error))


def temporal(outputs: list[torch.Tensor], target: torch.Tensor) -> float:
    """PSNR of frame-to-frame changes, as in evaluate_upstream.py (higher is steadier)."""
    values = []
    for t in range(1, len(outputs)):
        change = outputs[t] - outputs[t - 1]
        target_change = (target[t].float() - target[t - 1].float()) / 255
        error = (change - target_change).pow(2).mean().clamp_min(1e-10)
        values.append(float(-10 * torch.log10(error)))
    return float(np.mean(values))


@torch.inference_mode()
def upscale_scores(model, frames, gt: torch.Tensor, scale: int, crf: int, device) -> dict:
    """PSNR and flicker for bicubic and each exit over one clip, causally, full frames."""
    lr = to_tensor([f.rgb for f in frames], device)
    mv = torch.from_numpy(np.stack([f.mv for f in frames])).to(device).permute(0, 3, 1, 2)
    intra = torch.tensor([float(f.is_intra) for f in frames], device=device)
    quality = crf_to_quality(crf).view(1).to(device)
    scores = {}
    bicubic = [F.interpolate(lr[t:t + 1].float() / 255, scale_factor=scale, mode='bicubic', align_corners=False).clamp(0, 1)
               for t in range(len(frames))]
    scores['bicubic'] = float(np.mean([frame_psnr(o, gt[t]) for t, o in enumerate(bicubic)]))
    scores['bicubic_temporal'] = temporal(bicubic, gt)
    del bicubic
    psnrs, temporals, times = [], [], []
    for exit_index in range(len(model.exits)):
        state, outputs, values = None, [], []
        torch.cuda.synchronize() if device.type == 'cuda' else None
        begin = time.perf_counter()
        for t in range(len(frames)):
            output, state = model.step(lr[t:t + 1].float() / 255, mv[t:t + 1], intra[t:t + 1], state, scale,
                                       exit_index, quality=quality)
            output = output.clamp(0, 1).mul(255).round().div(255)
            values.append(frame_psnr(output, gt[t]))
            outputs.append(output)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        times.append((time.perf_counter() - begin) / len(frames) * 1000)
        psnrs.append(float(np.mean(values)))
        temporals.append(temporal(outputs, gt))
        del outputs
    scores.update(psnr=psnrs, temporal=temporals, model_ms=times)
    return scores


def direct_scores(path: Path, gt: torch.Tensor, device) -> dict:
    frames = [frame.rgb for frame in decode_with_motion(path)]
    outputs = [to_tensor([f], device).float() / 255 for f in frames]
    return {'psnr': [float(np.mean([frame_psnr(o, gt[t]) for t, o in enumerate(outputs)]))],
            'temporal': [temporal(outputs, gt)]}


def encode_ladder(source: Path, work: Path, fps: float, seconds: float) -> dict[str, Path]:
    """The reference and every rung from one filter graph, as the live server makes them."""
    graph = ladder_graph(fps, {'gt': 1 + len(DIRECT_CRFS), 'lr2': len(CRFS), 'lr4': len(CRFS)})
    paths = {'reference': work / 'reference.nut'}
    command = ['ffmpeg', '-v', 'error', '-y', '-t', str(seconds), '-i', str(source), '-filter_complex', graph,
               '-map', '[gt_0]', '-c:v', 'ffv1', '-colorspace', 'bt709', '-color_range', 'tv', str(paths['reference'])]
    outputs = [('direct', 'gt', DIRECT_CRFS, 1), ('x2', 'lr2', CRFS, 0), ('x4', 'lr4', CRFS, 0)]
    for rendition, label, crfs, first in outputs:
        for i, crf in enumerate(crfs):
            path = paths[rung_name(rendition, crf)] = work / f'{rendition}{crf}.mp4'
            command += ['-map', f'[{label}_{first + i}]', *X264, '-crf', str(crf), str(path)]
    subprocess.run(command, check=True)
    return paths


def measure_clip(model, source: Path, seconds: float, device) -> dict:
    info = probe(source)
    fps = info[2] / frame_step(info[2])
    rungs = {}
    with tempfile.TemporaryDirectory() as temp:
        paths = encode_ladder(source, Path(temp), info[2], seconds)
        gt = to_tensor([frame.rgb for frame in decode_with_motion(paths['reference'])], device)
        for name, path in paths.items():
            if name == 'reference':
                continue
            rendition, crf = name.split('@')
            if rendition == 'direct':
                entry = direct_scores(path, gt, device)
            else:
                frames = list(decode_with_motion(path))
                if len(frames) != len(gt):
                    raise ValueError(f'{source.name}: {len(frames)} {name} frames, {len(gt)} reference frames')
                entry = upscale_scores(model, frames, gt, SCALE[rendition], int(crf), device)
            entry['kbps'] = payload_kbps(path, fps)
            rungs[name] = entry
    return {'clip': source.name, 'fps': fps, 'frames': len(gt), 'rungs': rungs}


def bd_rate(anchor: list[tuple[float, float]], test: list[tuple[float, float]]) -> float | None:
    """Bjontegaard delta rate in percent (negative: test needs fewer bits for equal PSNR).

    Points are (kbps, psnr). log-rate is fitted as a cubic in PSNR for each curve and
    the difference averaged over the PSNR range both cover.
    """
    if len(anchor) < 4 or len(test) < 4:
        return None
    (ra, pa), (rt, pt) = (np.array(sorted(c, key=lambda p: p[1])).T for c in (anchor, test))
    low, high = max(pa.min(), pt.min()), min(pa.max(), pt.max())
    if high <= low:
        return None
    fa, ft = np.polyfit(pa, np.log10(ra), 3), np.polyfit(pt, np.log10(rt), 3)
    ia, it = np.polyint(fa), np.polyint(ft)
    difference = ((np.polyval(it, high) - np.polyval(it, low)) - (np.polyval(ia, high) - np.polyval(ia, low))) / (high - low)
    return float((10 ** difference - 1) * 100)


def curve(rungs: dict, rendition: str, value) -> list[tuple[float, float]]:
    return [(entry['kbps'], value(entry)) for name, entry in rungs.items() if name.startswith(rendition + '@')]


def hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points no cheaper point beats: the efficient part of a set of (kbps, psnr)."""
    best, kept = -np.inf, []
    for rate, quality in sorted(points):
        if quality > best:
            kept.append((rate, quality))
            best = quality
    return kept


def comparisons(rungs: dict) -> dict:
    """BD-rate of each way of delivering against direct delivery."""
    direct = curve(rungs, 'direct', lambda e: e['psnr'][0])
    result = {}
    for rendition in SCALE:
        result[f'{rendition}_model'] = bd_rate(direct, curve(rungs, rendition, lambda e: e['psnr'][-1]))
        result[f'{rendition}_model_exit0'] = bd_rate(direct, curve(rungs, rendition, lambda e: e['psnr'][0]))
        result[f'{rendition}_bicubic'] = bd_rate(direct, curve(rungs, rendition, lambda e: e['bicubic']))
    ladder = hull(direct + curve(rungs, 'x2', lambda e: e['psnr'][-1]) + curve(rungs, 'x4', lambda e: e['psnr'][-1]))
    result['ladder'] = bd_rate(direct, ladder) if len(ladder) >= 4 else None
    return {key: (round(value, 1) if value is not None else None) for key, value in result.items()}


def mean_rungs(clips: list[dict]) -> dict:
    names = clips[0]['rungs'].keys()
    rungs = {}
    for name in names:
        entries = [clip['rungs'][name] for clip in clips]
        rungs[name] = {key: (np.mean([e[key] for e in entries], axis=0).round(3).tolist())
                       for key in entries[0]}
    return rungs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--clips', type=Path, default=ROOT / 'data/test-clips')
    parser.add_argument('--limit', type=int, help='use only the first N clips')
    parser.add_argument('--seconds', type=float, default=4.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/rate_quality.json')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, half=False)
    sources = sorted(args.clips.glob('*.mp4'))[:args.limit]
    clips = []
    for number, source in enumerate(sources, 1):
        begin = time.time()
        clip = measure_clip(model, source, args.seconds, device)
        clip['bd_rate'] = comparisons(clip['rungs'])
        clips.append(clip)
        print(f'[{number}/{len(sources)}] {source.name} {time.time() - begin:.0f} s  {clip["bd_rate"]}', flush=True)

    rungs = mean_rungs(clips)
    per_clip = {key: [c['bd_rate'][key] for c in clips if c['bd_rate'][key] is not None] for key in clips[0]['bd_rate']}
    report = {
        'checkpoint': str(args.checkpoint), 'model_config': model.config(), 'seconds': args.seconds,
        'clips': len(clips), 'crfs': list(CRFS), 'direct_crfs': list(DIRECT_CRFS),
        'rungs': rungs,
        # Mean over clips of each clip's BD-rate, and how many clips had overlapping curves.
        'bd_rate_vs_direct': {key: {'mean': round(float(np.mean(v)), 1) if v else None, 'clips': len(v)}
                              for key, v in per_clip.items()},
        'bd_rate_of_mean_curves': comparisons(rungs),
        'per_clip': clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('bd_rate_vs_direct', 'bd_rate_of_mean_curves')}, indent=1))
    for name, entry in rungs.items():
        print(f'{name:10s} {entry["kbps"]:8.1f} kbps  psnr {entry["psnr"]}')


if __name__ == '__main__':
    main()
