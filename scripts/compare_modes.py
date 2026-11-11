#!/usr/bin/env python3
"""Compare delivery modes end to end on the live pipeline, over the same emulated link.

Modes (the client's controller is limited to one rendition, or uses all of them):

  direct   720p only, CRF chosen by the controller
  x2       360p + UpStreamVSR x2 only
  x4       180p + UpStreamVSR x4 only
  adaptive the full trade-off controller: any rendition and CRF

Each mode adapts its CRF to the link the same way, so the comparison isolates what
switching rendition adds. The server encodes the full ladder in every mode, so the
encoder load is the same.

For every run the four quantities are measured, not predicted:

  latency    display latency of shown frames, late drops, skipped frames, freezes
  resources  model GPU time, exits, GPU utilisation, power and temperature
             (nvidia-smi), peak GPU memory, client CPU time
  quality    PSNR of what the viewer saw against the 720p reference: every frame
             slot, a skipped or late frame counted as the last shown frame held
  bandwidth  payload bitrate sent, per link phase, and its share of the link

plus how many frames each rung and exit got. Writes summary.json and a Markdown
table next to the per-run reports.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from live.link import parse_link  # noqa: E402
from live.netem import CLIENT_IP, HOST_IP, NAMESPACE, NetemPath, PathConditions  # noqa: E402
from upstream_vsr.codec import decode_with_motion  # noqa: E402
from upstream_vsr.encoding import frame_step, ladder_graph  # noqa: E402
from prepare_streamsr import probe  # noqa: E402

MODES = {
    'direct': (['direct'], 'direct@33'),
    'x2': (['x2'], 'x2@28'),
    'x4': (['x4'], 'x4@28'),
    'adaptive': (['direct', 'x2', 'x4'], 'x4@28'),
}
PYTHON = sys.executable


def make_reference(source: Path, seconds: float, folder: Path) -> Path:
    """The 720p reference the server's graph makes, losslessly, for scoring."""
    info = probe(source)
    path = folder / 'reference.nut'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-t', str(seconds), '-i', str(source),
                    '-filter_complex', ladder_graph(info[2], {'gt': 1}), '-map', '[gt_0]',
                    '-c:v', 'ffv1', '-colorspace', 'bt709', '-color_range', 'tv', str(path)], check=True)
    return path


def reference_frames(path: Path):
    """Reference frames decoded as the client decodes them, one at a time (a 5-minute
    run would not fit in memory as a list)."""
    return (frame.rgb for frame in decode_with_motion(path))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    error = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(error, 1e-10)))


def score(client: dict, record: Path, reference, fps: float, phases: list[tuple[float, float]]) -> dict:
    """What the viewer saw, slot by slot, against the reference."""
    shape = tuple(client['record']['shape'])
    frames = np.memmap(record, dtype=np.uint8, mode='r', shape=(client['record']['frames'], *shape))
    shown = {f['index']: f['recorded'] for f in client['frames'] if not f['dropped'] and 'recorded' in f}
    total = client['end']['frames']
    per_slot, held, current, fresh = [], [], None, []
    for i, target in zip(range(total), reference):
        if i in shown:
            current = shown[i]
        fresh.append(i in shown)
        if current is None:  # nothing on screen yet
            per_slot.append(None)
            continue
        per_slot.append(psnr(frames[current], target))
    valid = [v for v in per_slot if v is not None]
    freezes, run = [], 0
    for is_fresh in fresh:
        if is_fresh:
            if run:
                freezes.append(run)
            run = 0
        else:
            run += 1
    if run:
        freezes.append(run)
    result = {'viewer_psnr': round(float(np.mean(valid)), 3),
              'shown_psnr': round(float(np.mean([v for v, f in zip(per_slot, fresh) if f and v is not None])), 3),
              'frozen_slots': int(sum(not f for f in fresh)), 'freeze_events': len(freezes),
              'longest_freeze_ms': round(max(freezes, default=0) / fps * 1000, 1), 'per_phase': []}
    bounds = [p[0] for p in phases] + [len(per_slot) / fps]
    for k, (begin, kbps) in enumerate(phases):
        slots = [v for i, v in enumerate(per_slot) if bounds[k] <= i / fps < bounds[k + 1] and v is not None]
        result['per_phase'].append({'from_s': begin, 'link_kbps': kbps,
                                    'viewer_psnr': round(float(np.mean(slots)), 3) if slots else None})
    result['per_slot'] = [None if v is None else round(v, 3) for v in per_slot]
    return result


def gpu_stats(path: Path, begin: float, end: float) -> dict:
    rows = []
    with open(path) as handle:
        for row in csv.reader(handle):
            try:
                stamp = time.mktime(time.strptime(row[0].strip().split('.')[0], '%Y/%m/%d %H:%M:%S'))
                rows.append((stamp, *(float(v) for v in row[1:5])))
            except (ValueError, IndexError):
                continue
    rows = [r for r in rows if begin - 1 <= r[0] <= end + 1]
    if not rows:
        return {}
    data = np.array([r[1:] for r in rows])
    return {'samples': len(rows), 'util_mean_pct': round(float(data[:, 0].mean()), 1),
            'util_max_pct': round(float(data[:, 0].max()), 1), 'power_mean_w': round(float(data[:, 1].mean()), 2),
            'power_max_w': round(float(data[:, 1].max()), 2), 'temp_max_c': round(float(data[:, 2].max()), 1),
            'memory_max_mib': round(float(data[:, 3].max()), 1)}


def bandwidth(client: dict, fps: float, phases: list[tuple[float, float]]) -> list[dict]:
    """Payload sent per link phase, from the frames the client received."""
    bounds = [p[0] for p in phases] + [float('inf')]
    out = []
    for k, (begin, kbps) in enumerate(phases):
        chosen = [f for f in client['frames'] if bounds[k] <= f['index'] / fps < bounds[k + 1]]
        span = (min(bounds[k + 1], client['end']['frames'] / fps) - begin)
        sent = sum(f['bytes'] for f in chosen) * 8 / 1000 / span if span > 0 else 0.0
        out.append({'from_s': begin, 'link_kbps': kbps, 'sent_kbps': round(sent, 1),
                    'link_share_pct': round(100 * sent / kbps, 1)})
    return out


def rung_usage(client: dict, fps: float) -> dict:
    usage, renditions, exits = {}, {}, {}
    for f in client['frames']:
        name = f"{f['rendition']}@{f['crf']}"
        usage[name] = usage.get(name, 0) + 1
        renditions[f['rendition']] = renditions.get(f['rendition'], 0) + 1
        if f.get('exit') is not None:
            exits[str(f['exit'])] = exits.get(str(f['exit']), 0) + 1
    total = client['end']['frames']
    return {'frames_by_rung': usage, 'frames_by_rendition': renditions, 'exits': exits,
            'share_by_rendition_pct': {k: round(100 * v / total, 1) for k, v in renditions.items()},
            'seconds_by_rendition': {k: round(v / fps, 2) for k, v in renditions.items()},
            'switches': len(client['summary'].get('switches', []))}


def as_user() -> list[str]:
    """Under sudo, run children as the invoking user (their files, their CUDA context)."""
    user = os.environ.get('SUDO_USER')
    if os.geteuid() != 0 or not user:
        return []
    keep = [f'{k}={os.environ[k]}' for k in ('PYTHONPATH', 'PATH') if k in os.environ]
    return ['sudo', '-u', user, 'env', f'HOME={os.path.expanduser("~" + user)}', *keep]


def give_back(path: Path) -> None:
    """Files written as root under sudo go back to the invoking user."""
    if os.geteuid() == 0 and 'SUDO_UID' in os.environ:
        subprocess.run(['chown', '-R', f"{os.environ['SUDO_UID']}:{os.environ['SUDO_GID']}", str(path)], check=False)


def run_mode(args, source: Path, mode: str, out: Path, port: int) -> tuple[dict, dict]:
    renditions, start = MODES[mode]
    out.mkdir(parents=True, exist_ok=True)
    give_back(args.out)
    phases = parse_link(args.link)
    path = None
    if args.netem:
        path = NetemPath(args.path)
        path.setup(phases[0][1])  # a fresh, empty router queue for every run
    host = HOST_IP if args.netem else '127.0.0.1'
    command = [*as_user(), PYTHON, str(ROOT / 'live/server.py'), str(source), '--seconds', str(args.seconds),
               '--crfs', '18', '23', '28', '33', '--direct-crfs', '23', '28', '33', '38', '43', '--start', start,
               '--host', host, '--port', str(port), '--report', str(out / 'server.json')]
    if not args.netem:
        command += ['--link', args.link]
    server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in server.stdout:
        if line.startswith('READY'):
            break
    else:
        raise RuntimeError(f'server for {mode} did not start')
    started, stop = threading.Event(), threading.Event()

    def drain_output() -> None:
        for line in server.stdout:
            if line.startswith('STREAMING'):
                started.set()

    reader = threading.Thread(target=drain_output, daemon=True)
    reader.start()
    schedule = path.follow(phases, started, stop) if path else None
    gpu_log = open(out / 'gpu.csv', 'w')
    sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,power.draw,temperature.gpu,memory.used',
                                '--format=csv,noheader,nounits', '-lms', '200'], stdout=gpu_log)
    try:
        command = [*as_user(), PYTHON, str(ROOT / 'live/client.py'), '--checkpoint', str(args.checkpoint), '--controller',
                   '--renditions', *renditions, '--playout-delay', str(args.playout_delay), '--host', host,
                   '--port', str(port), '--record', str(out / 'frames.raw'), '--report', str(out / 'client.json')]
        if args.netem:
            command = ['ip', 'netns', 'exec', NAMESPACE, *command]
        if args.half:
            command.append('--half')
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    finally:
        sampler.terminate()
        sampler.wait()
        gpu_log.close()
        server.wait(timeout=60)
        reader.join(timeout=10)
        stop.set()
        if path is not None:
            schedule.join(timeout=5)
            (out / 'network.json').write_text(json.dumps({
                'conditions': vars(args.path), 'server_ip': HOST_IP, 'client_ip': CLIENT_IP,
                'rate_applied': path.applied, 'tc_statistics': path.statistics()}, indent=1) + '\n')
        give_back(args.out)
    return json.loads((out / 'client.json').read_text()), json.loads((out / 'server.json').read_text())


def summarise(mode: str, client: dict, server: dict, quality: dict, gpu: dict, fps: float, phases) -> dict:
    s = client['summary']
    shown = s['frames'] - s['dropped']
    model = [f['model_s'] * 1000 for f in client['frames'] if f.get('exit') is not None]
    res = client['resources']
    return {
        'mode': mode,
        'latency': {'p50_ms': s['latency']['p50_ms'], 'p95_ms': s['latency']['p95_ms'], 'max_ms': s['latency']['max_ms'],
                    'ready_after_release_p95_ms': s['ready_after_release']['p95_ms'],
                    'shown': shown, 'dropped_late': s['dropped'], 'skipped': client['skipped'],
                    'frames': client['end']['frames'], 'frozen_slots': quality['frozen_slots'],
                    'freeze_events': quality['freeze_events'], 'longest_freeze_ms': quality['longest_freeze_ms'],
                    'server_backlog_skips': server['backlog_skips']},
        'resources': {'model_frames': len(model),
                      'model_ms_mean': round(float(np.mean(model)), 2) if model else 0.0,
                      'model_ms_p95': round(float(np.percentile(model, 95)), 2) if model else 0.0,
                      'model_gpu_s': res['model_gpu_s'],
                      'model_gpu_share_pct': round(100 * res['model_gpu_s'] / (client['end']['frames'] / fps), 1),
                      'decode_ms_mean': s['decode']['mean_ms'], 'peak_torch_mib': res['peak_gpu_mib'],
                      'client_cpu_s': res['client_cpu_s'],
                      'client_cpu_pct_of_core': round(100 * res['client_cpu_s'] / res['wall_s'], 1), **{f'gpu_{k}': v for k, v in gpu.items()}},
        'quality': {k: v for k, v in quality.items() if k != 'per_slot'},
        'bandwidth': {'mean_kbps': round(server['payload_bitrate_kbps'], 1), 'per_phase': bandwidth(client, fps, phases)},
        'usage': rung_usage(client, fps),
    }


def markdown(rows: list[dict]) -> str:
    head = ('| Mode | Viewer PSNR | Shown PSNR | Latency p50/p95/max ms | Shown/late/skipped | Freeze longest ms | '
            'kbps | Model GPU % | GPU util % | Power W | Rendition share |')
    lines = [head, '|' + '---|' * (head.count('|') - 1)]
    for r in rows:
        l, q, b, u, res = r['latency'], r['quality'], r['bandwidth'], r['usage'], r['resources']
        share = ', '.join(f'{k} {v}%' for k, v in u['share_by_rendition_pct'].items())
        lines.append(f"| {r['mode']} | {q['viewer_psnr']:.2f} | {q['shown_psnr']:.2f} | {l['p50_ms']:.0f}/{l['p95_ms']:.0f}/"
                     f"{l['max_ms']:.0f} | {l['shown']}/{l['dropped_late']}/{l['skipped']} | {q['longest_freeze_ms']:.0f} | "
                     f"{b['mean_kbps']:.0f} | {res['model_gpu_share_pct']} | {res.get('gpu_util_mean_pct', '-')} | "
                     f"{res.get('gpu_power_mean_w', '-')} | {share} |")
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sources', type=Path, nargs='+')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/kaggle-6-v3/run/best-calibrated.pt')
    parser.add_argument('--modes', nargs='+', default=list(MODES), choices=list(MODES))
    parser.add_argument('--link', default='0:2500,10:500,20:1500')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--playout-delay', type=float, default=0.15)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--keep-frames', action='store_true', help='keep the raw recordings (2.5 GB per 30 s run)')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/compare')
    parser.add_argument('--port', type=int, default=8810)
    parser.add_argument('--netem', action='store_true',
                        help='real TCP over a veth pair shaped by tc netem (needs root: run under sudo)')
    parser.add_argument('--path', type=PathConditions.parse, default=PathConditions(),
                        help='with --netem: one-way delay and jitter (ms), loss (%%), router queue (packets), '
                             'for example delay=20,jitter=5,loss=0.1,limit=300')
    args = parser.parse_args()
    if args.netem and os.geteuid() != 0:
        raise SystemExit('--netem needs root to create the namespace and shape it; run under sudo')

    phases = parse_link(args.link)
    everything, port = {'link': phases, 'seconds': args.seconds, 'playout_delay_s': args.playout_delay,
                        'half': args.half, 'checkpoint': str(args.checkpoint), 'clips': {},
                        'network': ({'kind': 'netem', **vars(args.path)} if args.netem else {'kind': 'emulated link'})}, args.port
    try:
        for source in args.sources:
            info = probe(source)
            fps = info[2] / frame_step(info[2])
            print(f'{source.name}: reference', flush=True)
            work = tempfile.TemporaryDirectory()
            reference_path = make_reference(source, args.seconds, Path(work.name))
            rows = []
            for mode in args.modes:
                out = args.out / source.stem / mode
                print(f'{source.name}: {mode}', flush=True)
                for attempt in range(3):  # ffmpeg occasionally crashes with 13 outputs; run again
                    client, server = run_mode(args, source, mode, out, port)
                    port += 1
                    if server.get('returncode') == 0:
                        break
                    print(f'{source.name}: {mode}: encoder exited with {server.get("returncode")}, running again', flush=True)
                else:
                    raise RuntimeError(f'{source.name}: {mode}: encoder kept failing')
                gpu = gpu_stats(out / 'gpu.csv', client['resources']['unix_begin'], client['resources']['unix_end'])
                quality = score(client, out / 'frames.raw', reference_frames(reference_path), fps, phases)
                (out / 'quality.json').write_text(json.dumps(quality) + '\n')
                if not args.keep_frames:
                    (out / 'frames.raw').unlink()
                row = summarise(mode, client, server, quality, gpu, fps, phases)
                row['decisions'] = client.get('controller', {}).get('decisions', [])
                row['switches'] = client['summary'].get('switches', [])
                rows.append(row)
                print(markdown([row]).splitlines()[-1], flush=True)
                time.sleep(3)  # let the GPU settle between runs
            everything['clips'][source.name] = rows
            work.cleanup()
            (args.out / source.stem / 'table.md').write_text(markdown(rows))
        (args.out / 'summary.json').write_text(json.dumps(everything, indent=1) + '\n')
        for name, rows in everything['clips'].items():
            print(f'\n{name}\n' + markdown(rows))
    finally:
        if args.netem:
            NetemPath(args.path).teardown()
        give_back(args.out)


if __name__ == '__main__':
    main()
