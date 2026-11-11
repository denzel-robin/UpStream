#!/usr/bin/env python3
"""Keep the GPU busy, as another application on the viewer's machine would.

Runs float16 matrix products in bursts: busy for --busy-ms, then idle for --idle-ms,
until killed. A separate process, so the GPU time-slices between it and the client
just as it would between the client and a game or a second video.

  python scripts/gpu_load.py --busy-ms 40 --idle-ms 10    # about 80% of the GPU
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--busy-ms', type=float, default=40)
    parser.add_argument('--idle-ms', type=float, default=10)
    parser.add_argument('--size', type=int, default=2048, help='matrix side; larger means longer kernels')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    a = torch.randn(args.size, args.size, device='cuda', dtype=torch.float16)
    b = torch.randn(args.size, args.size, device='cuda', dtype=torch.float16)
    print('LOADING', flush=True)
    while True:
        end = time.perf_counter() + args.busy_ms / 1000
        while time.perf_counter() < end:
            a = (a @ b).clamp_(-1, 1)
            torch.cuda.synchronize()
        time.sleep(args.idle_ms / 1000)


if __name__ == '__main__':
    main()
