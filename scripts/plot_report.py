#!/usr/bin/env python3
"""Figures for README.md and docs/REPORT.md, drawn from the recorded results.

Needs only matplotlib and numpy (any Python with them, for example the system one):

  python3 scripts/plot_report.py            # writes docs/figures/*.png

Inputs: configs/rate_quality.json, results/baselines/offline.json,
results/no-state/offline.json, results/final/baselines-live/summary.json,
results/final/compare/summary.json, results/netem-*/summary.json,
results/multiclip/ci.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs/figures'

# Validated categorical slots (light surface); ours is always blue, others recede to gray.
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
RENDITION = {'direct': ORANGE, 'x2': AQUA, 'x4': BLUE}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.titlecolor': INK, 'axes.titlelocation': 'left', 'lines.linewidth': 2, 'axes.axisbelow': True,
})


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def runs(path: str) -> dict[str, list[dict]]:
    by_mode: dict[str, list[dict]] = {}
    for clip_runs in load(path)['clips'].values():
        for run in clip_runs:
            by_mode.setdefault(run['mode'], []).append(run)
    return by_mode


def viewer(rs: list[dict]) -> float:
    return float(np.mean([r['quality']['viewer_psnr'] for r in rs]))


def dots(ax, labels, values, colours) -> None:
    """Dot plot: PSNR differences matter, and a bar on a truncated axis would exaggerate them."""
    ax.scatter(values, labels, s=80, color=colours, edgecolor=SURFACE, linewidth=2, zorder=3)
    for i, v in enumerate(values):
        ax.text(v + 0.08, i, f'{v:.2f}', va='center', color=INK2, fontsize=9)
    ax.set_xlim(min(values) - 0.4, max(values) + 0.6)
    ax.set_ylim(-0.6, len(values) - 0.4)
    ax.grid(axis='y', visible=False)


def rate_quality() -> None:
    rungs = load('configs/rate_quality.json')['rungs']
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = [('direct', 'Direct 720p', ORANGE, lambda e: e['psnr'][-1]),
              ('x2', 'x2 + UpStreamVSR', AQUA, lambda e: e['psnr'][-1]),
              ('x4', 'x4 + UpStreamVSR', BLUE, lambda e: e['psnr'][-1]),
              ('x4', 'x4 + bicubic', GRAY, lambda e: e['bicubic'])]
    for rendition, label, colour, value in series:
        points = sorted((e['kbps'], value(e)) for n, e in rungs.items() if n.split('@')[0] == rendition)
        kbps, psnr = zip(*points)
        ax.plot(kbps, psnr, color=colour, marker='o', markersize=5, linestyle='--' if 'bicubic' in label else '-',
                label=label)
        ax.annotate(label, (kbps[-1], psnr[-1]), xytext=(6, 0), textcoords='offset points', va='center',
                    color=INK2, fontsize=9)
    ax.set_xscale('log')
    ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000], ['50', '100', '200', '500', '1000', '2000', '5000'])
    ax.set_xlim(60, 12000)
    ax.set_xlabel('Bitrate, kbit/s (log scale)')
    ax.set_ylabel('PSNR against 720p, dB')
    ax.set_title('Rate-quality: upscaling wins at low bitrates, direct at high')
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'rate_quality.png', dpi=160)
    plt.close(fig)


def speed_quality() -> None:
    offline = load('results/baselines/offline.json')['summary']['x4@18']
    final = load('results/no-state/offline.json')['summary']['x4@18']['no-state']
    points = {'UpStreamVSR (ours)': final, 'CDA-VSR': offline['cdavsr'], 'NanoVSR': offline['nanovsr'],
              'SPAN': offline['span'], 'EfRLFN': offline['efrlfn'], 'Bicubic': offline['bicubic']}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(20, 60, color=GRID, alpha=0.6, linewidth=0)
    ax.text(21, 31.0, 'over the 20 ms\nlive model budget', color=INK2, fontsize=9, va='center')
    for name, entry in points.items():
        ours = 'ours' in name
        ax.scatter(entry['ms_median'], entry['psnr'], s=90 if ours else 60, color=BLUE if ours else GRAY,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{name}  {entry['psnr']:.2f} dB, {entry['ms_median']:.1f} ms", (entry['ms_median'], entry['psnr']),
                    xytext=(-8, 8) if name == 'CDA-VSR' else (8, 4 if name != 'Bicubic' else -12),
                    ha='right' if name == 'CDA-VSR' else 'left', textcoords='offset points',
                    color=INK if ours else INK2, fontsize=9, fontweight='bold' if ours else 'normal')
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 20, 50], ['2', '5', '10', '20', '50'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(1.5, 60)
    ax.set_ylim(30, 33.5)
    ax.set_xlabel('Time per frame, ms (RTX 3050 Laptop, float16, log scale)')
    ax.set_ylabel('PSNR, dB')
    ax.set_title('x4 at CRF 18, 18 clips: best quality and fastest learned model')
    fig.tight_layout()
    fig.savefig(OUT / 'speed_quality.png', dpi=160)
    plt.close(fig)


def live_models() -> None:
    modes = runs('results/final/baselines-live/summary.json')
    names = {'': 'UpStreamVSR (ours)', '-nanovsr': 'NanoVSR', '-cdavsr': 'CDA-VSR', '-span': 'SPAN',
             '-efrlfn': 'EfRLFN', '-bicubic': 'Bicubic'}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=False)
    for ax, prefix, title in ((axes[0], 'adaptive', 'Adaptive controller'),
                              (axes[1], 'x4', 'x4 only (identical delivery)')):
        rows = [(label, viewer(modes[prefix + suffix])) for suffix, label in names.items() if prefix + suffix in modes
                and not (prefix == 'x4' and suffix == '-cdavsr')]
        rows.sort(key=lambda row: row[1])
        labels, values = zip(*rows)
        colours = [BLUE if 'ours' in label else GRAY for label in labels]
        dots(ax, labels, values, colours)
        ax.set_title(title)
        ax.set_xlabel('Viewer PSNR, dB')
        ax.grid(axis='y', visible=False)
    fig.suptitle('Live system, 3 clips x 30 s: UpStreamVSR vs published models in the same pipeline',
                 x=0.01, ha='left', fontsize=11, fontweight='bold', color=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'live_models.png', dpi=160)
    plt.close(fig)


def timeline(clip: str = 'NRqG3AEOVGg@1440.mp4') -> None:
    summary = load('results/final/compare/summary.json')
    run = next(r for r in summary['clips'][clip] if r['mode'] == 'adaptive')
    link = summary['link']
    seconds, fps = summary['seconds'], 30
    first = run['decisions'][0]['from'] if run['decisions'] else run['switches'][0]['to']
    changes = [(0, first)] + [(s['frame'], s['to']) for s in run['switches']]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={'height_ratios': [1, 1.3]})
    times = [t for t, _ in link] + [seconds]
    rates = [r for _, r in link]
    top.step(times, rates + [rates[-1]], where='post', color=INK2)
    top.set_ylabel('Link, kbit/s')
    top.set_ylim(0, 2900)
    top.set_title(f'One adaptive run ({clip.split("@")[0]}): the link drops at 10 s and recovers at 20 s')
    order = {'direct': 2, 'x2': 1, 'x4': 0}
    for (frame, rung), (next_frame, _) in zip(changes, changes[1:] + [(seconds * fps, None)]):
        rendition, crf = rung.split('@')
        start, end = frame / fps, next_frame / fps
        bottom.barh(order[rendition], end - start, left=start, height=0.6, color=RENDITION[rendition],
                    edgecolor=SURFACE, linewidth=2)
        if end - start >= 1.5:
            bottom.text((start + end) / 2, order[rendition], f'CRF {crf}', ha='center', va='center',
                        color='white', fontsize=8, fontweight='bold')
    bottom.set_yticks([0, 1, 2], ['x4 (180p)', 'x2 (360p)', 'direct (720p)'])
    bottom.set_ylim(-0.6, 2.6)
    bottom.set_xlim(0, seconds)
    bottom.set_xlabel('Stream time, s')
    bottom.set_ylabel('Rung sent')
    bottom.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'controller_timeline.png', dpi=160)
    plt.close(fig)


def netem() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    labels = {'direct': 'Direct only', 'x2': 'x2 only', 'x4': 'x4 only', 'adaptive': 'Adaptive'}
    for ax, profile, title in ((axes[0], 'broadband', 'Broadband (40 ms RTT, 0.1% loss)'),
                               (axes[1], 'mobile', 'Mobile (80 ms RTT, 1% loss)')):
        modes = runs(f'results/netem-{profile}/summary.json')
        rows = [(labels[m], viewer(modes[m])) for m in labels]
        names, values = zip(*rows)
        dots(ax, names, values, [BLUE if n == 'Adaptive' else GRAY for n in names])
        ax.set_title(title)
        ax.set_xlabel('Viewer PSNR, dB')
        ax.grid(axis='y', visible=False)
    fig.suptitle('Real TCP shaped with tc netem, 3 clips: adaptive beats every fixed mode', x=0.01, ha='left',
                 fontsize=11, fontweight='bold', color=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'netem.png', dpi=160)
    plt.close(fig)


def multiclip() -> None:
    """Adaptive minus each mode, paired by clip, with 95% intervals (scripts/summarise_clips.py)."""
    ci = load('results/multiclip/ci.json')['modes']
    rows = [('x2', 'x2 only'), ('direct', 'Direct only'), ('x4', 'x4 only'), ('mpc', 'RobustMPC, full ladder'),
            ('mpc-direct', 'RobustMPC, direct only'), ('adaptive-nanovsr', 'NanoVSR in controller'),
            ('adaptive-span', 'SPAN in controller'), ('adaptive-cdavsr', 'CDA-VSR in controller'),
            ('adaptive-efrlfn', 'EfRLFN in controller')][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), sharey=True)
    for ax, metric, title in ((axes[0], 'viewer_psnr', 'Viewer PSNR, dB'), (axes[1], 'viewer_vmaf', 'VMAF')):
        ax.axvline(0, color=INK2, linewidth=1)
        for i, (mode, _) in enumerate(rows):
            d = ci[mode][metric]['reference_minus_mode']
            lo, hi = d['mean'] - d['ci95'], d['mean'] + d['ci95']
            colour = BLUE if lo > 0 else ORANGE if hi < 0 else GRAY
            ax.plot([lo, hi], [i, i], color=colour, linewidth=2, solid_capstyle='round')
            ax.scatter(d['mean'], i, s=60, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(hi, i + 0.28, f"{d['mean']:+.{2 if metric == 'viewer_psnr' else 1}f}", ha='right',
                    color=INK2, fontsize=8)
        ax.set_title(f'Adaptive minus mode: {title}')
        ax.grid(axis='y', visible=False)
        ax.set_ylim(-0.6, len(rows) - 0.3)
    axes[0].set_yticks(range(len(rows)), [label for _, label in rows])
    fig.suptitle('Emulated link, 18 clips, paired 95% intervals. Blue: adaptive better; orange: worse; gray: not significant',
                 x=0.01, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'multiclip.png', dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for figure in (rate_quality, speed_quality, live_models, timeline, netem, multiclip):
        figure()
        print('wrote', figure.__name__)


if __name__ == '__main__':
    main()
