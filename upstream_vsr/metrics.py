"""Perceptual quality metrics alongside PSNR: VMAF (Netflix, through FFmpeg's libvmaf)
and LPIPS (Zhang et al. 2018, AlexNet features; lower is better).

Both take frames as uint8 RGB arrays (H, W, 3), as the live client and the offline
benchmarks produce them. VMAF converts distorted and reference frames to YUV 4:2:0
with the same BT.709 limited-range conversion, so neither side is favoured.

  pip install lpips      # LPIPS; it downloads the AlexNet weights once
  FFmpeg built with --enable-libvmaf   # VMAF; the built-in vmaf_v0.6.1 model is used
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


class Lpips:
    """LPIPS between two RGB frames, on the GPU. Runs in float32: the AlexNet
    normalisation overflows in float16."""

    def __init__(self, device: torch.device | str = 'cuda') -> None:
        import lpips  # optional dependency
        self.device = torch.device(device)
        self.net = lpips.LPIPS(net='alex', verbose=False).to(self.device).eval()

    @staticmethod
    def _tensor(frame: np.ndarray | torch.Tensor, device) -> torch.Tensor:
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(np.ascontiguousarray(frame))
        if frame.dtype == torch.uint8:
            frame = frame.to(device).permute(2, 0, 1).float() / 255
        else:  # already a float image in [0, 1], (3, H, W) or (1, 3, H, W)
            frame = frame.to(device).float()
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        return frame * 2 - 1

    @torch.inference_mode()
    def __call__(self, a, b) -> float:
        return float(self.net(self._tensor(a, self.device), self._tensor(b, self.device)).mean())


def vmaf(distorted: Iterable[np.ndarray], reference: Iterable[np.ndarray], width: int, height: int,
         fps: float = 30.0, threads: int | None = None) -> dict:
    """VMAF of two equally long frame sequences, streamed to FFmpeg through two pipes
    (a 30-second 720p run does not fit in memory). Returns the mean, the harmonic mean
    (which weighs bad frames more) and the per-frame scores."""
    threads = threads or max(os.cpu_count() // 2, 1)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        pipes = [work / 'distorted', work / 'reference']
        for pipe in pipes:
            os.mkfifo(pipe)
        log = work / 'vmaf.json'
        raw = ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps)]
        to_yuv = 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setpts=PTS-STARTPTS'
        command = ['ffmpeg', '-v', 'error', '-nostdin', *raw, '-i', str(pipes[0]), *raw, '-i', str(pipes[1]),
                   '-lavfi', f'[0:v]{to_yuv}[d];[1:v]{to_yuv}[r];'
                             f'[d][r]libvmaf=log_fmt=json:log_path={log}:n_threads={threads}',
                   '-f', 'null', '-']
        process = subprocess.Popen(command, stderr=subprocess.PIPE)
        counts, errors = [0, 0], []

        def feed(index: int, frames: Iterable[np.ndarray]) -> None:
            try:
                with open(pipes[index], 'wb') as handle:
                    for frame in frames:
                        handle.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
                        counts[index] += 1
            except BrokenPipeError as error:
                errors.append(error)

        feeders = [threading.Thread(target=feed, args=(i, f), daemon=True) for i, f in enumerate((distorted, reference))]
        for thread in feeders:
            thread.start()
        for thread in feeders:
            thread.join()
        _, stderr = process.communicate()
        if process.returncode != 0 or errors:
            raise RuntimeError(f'libvmaf failed: {stderr.decode(errors="replace").strip() or errors}')
        if counts[0] != counts[1]:
            raise ValueError(f'VMAF needs equal lengths, got {counts[0]} and {counts[1]} frames')
        result = json.loads(log.read_text())
    per_frame = [frame['metrics']['vmaf'] for frame in result['frames']]
    pooled = result.get('pooled_metrics', {}).get('vmaf', {})
    return {'vmaf': round(float(np.mean(per_frame)), 3),
            'vmaf_harmonic': round(float(pooled.get('harmonic_mean', len(per_frame) / np.sum(1 / (np.array(per_frame) + 1)) - 1)), 3),
            'per_frame': [round(v, 3) for v in per_frame]}
