#!/usr/bin/env python3
"""Mean over clips with 95% confidence intervals, from compare_modes.py summaries.

The unit of replication is the clip: each mode is run once per clip, under the same
link schedule, so differences between modes are paired by clip. For every mode and
metric this prints the mean over clips with a t-based 95% interval, and for the
difference to a reference mode (adaptive by default) the paired mean, its 95%
interval and how many clips the reference wins.

  python scripts/summarise_clips.py results/multiclip/summary.json
  python scripts/summarise_clips.py results/multiclip/summary.json --json results/multiclip/ci.json

Needs only numpy (no SciPy): the t quantiles are tabulated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Two-sided 95% Student t quantiles, t_{0.975, df}.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
        29: 2.045, 30: 2.042}

# name: (path in a compare_modes row, higher is better, format)
METRICS = {
    'viewer_psnr': (('quality', 'viewer_psnr'), True, '.2f'),
    'viewer_vmaf': (('quality', 'viewer_vmaf'), True, '.1f'),
    'viewer_lpips': (('quality', 'viewer_lpips'), False, '.3f'),
    'latency_p95_ms': (('latency', 'p95_ms'), False, '.0f'),
    'late_or_skipped': ((None,), False, '.0f'),
    'longest_freeze_ms': (('latency', 'longest_freeze_ms'), False, '.0f'),
    'kbps': (('bandwidth', 'mean_kbps'), None, '.0f'),
    'model_gpu_pct': (('resources', 'model_gpu_share_pct'), False, '.1f'),
    'power_w': (('resources', 'gpu_power_mean_w'), False, '.1f'),
}


def value(row: dict, metric: str) -> float | None:
    if metric == 'late_or_skipped':
        return row['latency']['dropped_late'] + row['latency']['skipped']
    node = row
    for key in METRICS[metric][0]:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return float(node)


def interval(values: list[float]) -> tuple[float, float]:
    """(mean, half-width of the 95% t interval)."""
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        return float(data.mean()), float('nan')
    t = T975.get(len(data) - 1, 1.96)
    return float(data.mean()), float(t * data.std(ddof=1) / math.sqrt(len(data)))


def summarise(paths: list[Path], reference: str) -> dict:
    by_clip: dict[str, dict[str, dict]] = {}
    for path in paths:
        for clip, rows in json.loads(path.read_text())['clips'].items():
            for row in rows:
                by_clip.setdefault(clip, {})[row['mode']] = row
    modes = list(dict.fromkeys(m for rows in by_clip.values() for m in rows))
    out = {'clips': len(by_clip), 'reference': reference, 'modes': {}}
    for mode in modes:
        entry = {}
        for metric, (_, higher, _) in METRICS.items():
            pairs = [(value(rows[mode], metric), value(rows[reference], metric) if reference in rows else None)
                     for rows in by_clip.values() if mode in rows]
            values = [v for v, _ in pairs if v is not None]
            if not values:
                continue
            mean, half = interval(values)
            entry[metric] = {'mean': mean, 'ci95': half, 'n': len(values)}
            diffs = [r - v for v, r in pairs if v is not None and r is not None]
            if mode != reference and diffs and higher is not None:
                d_mean, d_half = interval(diffs)
                wins = sum((d > 0) if higher else (d < 0) for d in diffs)
                entry[metric]['reference_minus_mode'] = {'mean': d_mean, 'ci95': d_half, 'n': len(diffs),
                                                         'reference_better_clips': wins}
        out['modes'][mode] = entry
    return out


def markdown(result: dict, metrics: list[str]) -> str:
    ref = result['reference']
    lines = [f"{result['clips']} clips; mean ± 95% CI over clips; Δ = {ref} − mode (paired by clip), "
             f"with the clips {ref} is better on", '',
             '| Mode | ' + ' | '.join(metrics) + ' |', '|---|' + '---|' * len(metrics)]
    for mode, entry in result['modes'].items():
        cells = []
        for metric in metrics:
            if metric not in entry:
                cells.append('-')
                continue
            fmt = METRICS[metric][2]
            e = entry[metric]
            cell = f"{e['mean']:{fmt}} ± {e['ci95']:{fmt}}"
            if 'reference_minus_mode' in e:
                d = e['reference_minus_mode']
                cell += f"<br>Δ {d['mean']:+{fmt}} ± {d['ci95']:{fmt}} ({d['reference_better_clips']}/{d['n']})"
            cells.append(cell)
        lines.append(f'| {mode} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('summaries', type=Path, nargs='+', help='summary.json files from compare_modes.py')
    parser.add_argument('--reference', default='adaptive')
    parser.add_argument('--metrics', nargs='+', default=['viewer_psnr', 'viewer_vmaf', 'viewer_lpips',
                                                         'latency_p95_ms', 'late_or_skipped', 'kbps', 'model_gpu_pct'],
                        choices=list(METRICS))
    parser.add_argument('--json', type=Path, help='also write the full result here')
    parser.add_argument('--markdown', type=Path, help='also write the table here')
    args = parser.parse_args()
    result = summarise(args.summaries, args.reference)
    table = markdown(result, args.metrics)
    print(table)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')
    if args.markdown:
        args.markdown.write_text(table)


if __name__ == '__main__':
    main()
