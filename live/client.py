#!/usr/bin/env python3
"""Live client: receive, decode with motion vectors, reconstruct causally, play on time.

Threads, joined by small bounded queues:
  receive      socket records -> encoded packets
  decode       one MotionDecoder per rendition run -> RGB, motion field, frame type
  reconstruct  UpStreamVSR.step on the GPU, exit chosen per frame by DepthPolicy;
               direct frames pass through
  main         waits until each frame is due, shows or skips it, records timings

The rung is chosen by --schedule (fixed frames) or by --controller, which runs the
trade-off controller (live/controller.py) on every packet in the receive thread.

The server releases frame i at t0 + i / fps, as a camera would produce it, where
t0 is when frame 0 was sent. Frame i is due on screen at t0 + i / fps + playout
delay. A frame later than --drop-late is skipped on screen, but it was still
reconstructed, so the recurrent state never has a gap.

Latency per frame is display time minus t0 + i / fps: transport, decoding,
reconstruction and waiting in the playout buffer. Encoding is not included,
because the server encodes ahead of the release time. Both ends must run on one
machine for the clocks to agree.
"""
from __future__ import annotations

import argparse
import json
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

if '--display' in sys.argv:
    # OpenCV's Qt window must load before PyAV (imported by upstream_vsr.codec): PyAV
    # bundles its own libxcb, and Qt then hangs opening the window on that copy.
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')  # the wheel has no Wayland plugin; XWayland works
    os.environ.setdefault('QT_LOGGING_RULES', 'qt.qpa.*=false;qt.text.font.db=false')
    import cv2  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from live.controller import TradeoffController, load_table  # noqa: E402
from live.policy import DepthPolicy  # noqa: E402
from live.protocol import LABEL, SCALE, Channel, parse_packet, parse_rung, rung_name  # noqa: E402
from upstream_vsr.encoding import SIZES  # noqa: E402
from upstream_vsr.codec import MotionDecoder  # noqa: E402
from upstream_vsr.baselines import NAMES as BASELINES, Baseline  # noqa: E402
from upstream_vsr.model import UpStreamVSR, crf_to_quality  # noqa: E402

END = object()


def load_model(path: Path, device: torch.device, half: bool) -> UpStreamVSR:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = UpStreamVSR(**checkpoint['config'])
    model.load_checkpoint_state(checkpoint['model'])
    model = model.eval().fuse().to(device)
    return model.half() if half else model


def parse_schedule(text: str | None) -> list[tuple[int, str]]:
    """'0:x4,90:x2@28,180:direct' -> [(0, 'x4'), (90, 'x2@28'), (180, 'direct')]."""
    if not text:
        return []
    items = [(int(frame), name) for frame, name in (part.split(':') for part in text.split(','))]
    for _, name in items:
        parse_rung(name, 0)  # raises on an unknown rendition
    return sorted(items)


class Reconstructor:
    """Holds the model and its recurrent state; one call per decoded frame."""

    def __init__(self, model: UpStreamVSR | None, device: torch.device, policy: DepthPolicy | None, half: bool,
                 bicubic: bool = False, baseline: Baseline | None = None):
        self.model, self.device, self.policy, self.half = model, device, policy, half
        self.bicubic = bicubic  # ablation: bicubic in place of the model, same rung choices
        self.baseline = baseline  # comparison: a published model in place of UpStreamVSR
        self.dtype = torch.float16 if half else torch.float32
        self.state = None
        self.last_scale = None

    def to_device(self, array: np.ndarray, scale: float = 1.0) -> torch.Tensor:
        tensor = torch.from_numpy(array).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.dtype) * scale

    def warm_up(self, crf: int = 23, steps: int = 15) -> None:
        """Load kernels and seed the policy with the median time of each scale and exit.

        Runs before connecting, so the stream does not queue up behind it.
        """
        for name in ('x4', 'x2'):
            width, height = SIZES[LABEL[name]]
            frame = np.zeros((height, width, 3), np.uint8)
            motion = np.zeros((height // 4, width // 4, 2), np.float32)
            if self.baseline is not None:
                # One depth: every exit gets the baseline's time, so the controller prices
                # its rungs by what it really costs. A scale it lacks never fits.
                times = [float('inf')]
                if SCALE[name] in self.baseline.scales:
                    self.baseline.reset()
                    times = [self.run_baseline({'rgb': frame, 'mv': motion, 'is_intra': False}, SCALE[name])[1]['model_s']
                             for _ in range(steps)][steps // 3:]
                for exit_index in range(len(self.model.exits)):
                    self.policy.expected[(SCALE[name], exit_index)] = float(np.median(times))
                continue
            for exit_index in range(len(self.model.exits)):
                self.state, times = None, []
                for _ in range(steps):
                    times.append(self.run(frame, motion, False, SCALE[name], exit_index, crf)[1])
                self.policy.expected[(SCALE[name], exit_index)] = float(np.median(times[steps // 3:]))
        self.state = None

    @torch.inference_mode()
    def run(self, rgb, mv, is_intra, scale, exit_index, crf) -> tuple[np.ndarray, float]:
        begin = time.perf_counter()
        x = self.to_device(rgb, 1 / 255)
        motion = self.to_device(mv)
        intra = torch.tensor([float(is_intra)], device=self.device, dtype=self.dtype)
        quality = crf_to_quality(crf).view(1).to(self.device, self.dtype)
        output, self.state = self.model.step(x, motion, intra, self.state, scale, exit_index, quality=quality)
        frame = output[0].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        seconds = time.perf_counter() - begin  # .cpu() waited for the GPU
        self.policy.update(scale, exit_index, seconds)
        return frame, seconds

    @torch.inference_mode()
    def upscale_bicubic(self, rgb: np.ndarray, scale: int) -> tuple[np.ndarray, dict]:
        begin = time.perf_counter()
        x = self.to_device(rgb, 1 / 255)
        y = torch.nn.functional.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=False)
        frame = y[0].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return frame, {'exit': None, 'model_s': time.perf_counter() - begin, 'bicubic': True}

    def run_baseline(self, item: dict, scale: int) -> tuple[np.ndarray, dict]:
        begin = time.perf_counter()
        output = self.baseline.step(self.to_device(item['rgb'], 1 / 255), self.to_device(item['mv']),
                                    item['is_intra'], scale)
        frame = output[0].mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        seconds = time.perf_counter() - begin
        for exit_index in range(self.policy.num_exits):
            self.policy.update(scale, exit_index, seconds)
        return frame, {'exit': 0, 'model_s': seconds, 'upscaler': self.baseline.name}

    def __call__(self, item: dict, time_left: float) -> tuple[np.ndarray, dict]:
        scale = SCALE[item['rendition']]
        if scale == 1 or self.model is None:
            self.state = None  # direct frames carry no model state
            self.last_scale = scale
            return item['rgb'], {'exit': None, 'model_s': 0.0}
        if self.last_scale == 1:
            self.state = None
        self.last_scale = scale
        if self.bicubic:
            return self.upscale_bicubic(item['rgb'], scale)
        if self.baseline is not None:
            if scale not in self.baseline.scales:  # only if the controller was told to use it anyway
                return self.upscale_bicubic(item['rgb'], scale)
            if self.last_scale != scale:
                self.baseline.reset()
            return self.run_baseline(item, scale)
        exit_index = self.policy.choose(scale, time_left)
        frame, seconds = self.run(item['rgb'], item['mv'], item['is_intra'], scale, exit_index, item['crf'])
        return frame, {'exit': exit_index, 'model_s': seconds}


class Recorder:
    """Writes every reconstructed frame, raw RGB, in a thread so display is not held up.

    Dropped frames are recorded too; the report says which frames were shown, so a
    scorer can rebuild what the viewer saw.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file, self.count, self.shape = open(path, 'wb'), 0, None
        self.queue: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self.write, name='record', daemon=True)
        self.thread.start()

    def add(self, frame: np.ndarray) -> int:
        if self.shape is None:
            self.shape = list(frame.shape)
        elif list(frame.shape) != self.shape:
            raise ValueError(f'frame shape {frame.shape} differs from {self.shape}')
        self.queue.put(np.ascontiguousarray(frame))
        self.count += 1
        return self.count - 1

    def write(self) -> None:
        while (frame := self.queue.get()) is not None:
            self.file.write(frame.tobytes())

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        self.file.close()


def open_window(name: str):
    """Open the display window before the worker threads start (cv2 is imported at the
    top of this module, ahead of PyAV, when --display is given)."""
    import cv2
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.waitKey(1)
    return cv2


def overlay(cv2, frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Status text on a dark band at the top-left of a BGR frame."""
    scale = frame.shape[0] / 720
    height = int((14 + 26 * len(lines)) * scale)
    width = int(min(frame.shape[1], 700 * scale))
    band = frame[:height, :width]
    band[:] = (band * 0.35).astype(np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (int(10 * scale), int((30 + 26 * i) * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62 * scale, (255, 255, 255), max(1, int(1.4 * scale)), cv2.LINE_AA)
    return frame


def run(args: argparse.Namespace) -> dict:
    cv2 = open_window('UpStream live') if args.display else None
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, args.half) if args.checkpoint else None
    schedule = parse_schedule(args.schedule)
    if schedule and args.controller:
        raise SystemExit('use --schedule or --controller, not both')
    report = {'checkpoint': str(args.checkpoint) if args.checkpoint else None, 'half': args.half,
              'playout_delay_s': args.playout_delay, 'drop_late_s': args.drop_late,
              'schedule': schedule, 'frames': [], 'errors': [], 'skipped': 0, 'guard': [],
              'ablation': {'upscaler': args.upscaler, 'fixed_exit': args.fixed_exit, 'gpu_aware': args.gpu_aware}}
    policy = None
    baseline = Baseline(args.upscaler, device, args.half) if args.upscaler in BASELINES else None
    reconstruct = Reconstructor(model, device, None, args.half, bicubic=args.upscaler == 'bicubic', baseline=baseline)
    if model is not None:
        report['model_config'] = model.config()
        # Frame interval is not known until the header; 30 fps until then.
        policy = reconstruct.policy = DepthPolicy(len(model.exits), 1 / 30)
        if args.fixed_exit is not None:
            policy.fixed = args.fixed_exit % len(model.exits)
        warm = time.perf_counter()
        reconstruct.warm_up()
        report['warm_up_s'] = time.perf_counter() - warm
        report['expected_ms_after_warm_up'] = {f'x{s}_exit{e}': round(v * 1000, 2)
                                               for (s, e), v in sorted(policy.expected.items())}

    sock = socket.create_connection((args.host, args.port), timeout=args.connect_timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(None)
    channel = Channel(sock)
    kind, payload = channel.receive()
    if kind != b'H':
        raise RuntimeError('expected a header')
    header = json.loads(payload)
    report['header'] = header
    fps = header['fps']
    if policy is not None:
        policy.frame_interval = 1 / fps
    controller = None
    if args.controller:
        # --renditions limits the choice, for comparing one rendition against the full ladder.
        ladder = [name for name in header['ladder'] if parse_rung(name, 0)[0] in args.renditions]
        if header['start'] not in ladder:
            raise SystemExit(f"the server starts on {header['start']}, outside --renditions {args.renditions}")
        controller = TradeoffController(
            load_table(args.rate_quality), ladder, 1 / fps,
            latency_budget_s=max(args.playout_delay - args.processing_margin, 0.01), policy=policy,
            gpu_share=args.gpu_share, deadline_s=args.playout_delay + args.drop_late - args.processing_margin,
            gpu_aware=args.gpu_aware)
        controller.start(header['start'], time.monotonic())
        report['controller'] = {'gpu_share': args.gpu_share, 'latency_budget_s': controller.latency_budget_s,
                                'rate_quality': str(args.rate_quality), 'ladder': ladder}

    packets: queue.Queue = queue.Queue(maxsize=args.queue)
    decoded: queue.Queue = queue.Queue(maxsize=args.queue)
    shown: queue.Queue = queue.Queue(maxsize=args.queue)
    anchor: list[float] = []  # t0: when the server released frame 0

    def due(index: int) -> float:
        return anchor[0] + index / fps + args.playout_delay

    def guarded(name, body):
        def wrapper():
            try:
                body()
            except Exception as error:  # reported, and the pipeline is shut down cleanly
                report['errors'].append(f'{name}: {error!r}')
            finally:
                downstream = {'receive': packets, 'decode': decoded, 'reconstruct': shown}[name]
                downstream.put(END)
        return threading.Thread(target=wrapper, name=name, daemon=True)

    received = [0]
    reported = [1e-9]  # capacity last sent to the server

    def receive():
        while True:
            record = channel.receive()
            if record is None:
                raise EOFError('server closed without an end record')
            kind, payload = record
            if kind == b'E':
                report['end'] = json.loads(payload)
                # Includes frames dropped at the very end, which no later keyframe reveals.
                report['skipped'] = report['end']['frames'] - received[0]
                return
            if kind == b'R':
                if controller is not None:
                    controller.observe_rates(json.loads(payload))
                continue
            if kind == b'G':
                guard = json.loads(payload)
                report['guard'].append(guard)  # the server dropped frames to keep its backlog short
                continue
            if kind != b'P':
                raise ValueError(f'unexpected record {kind!r}')
            packet = parse_packet(payload)
            arrived = time.monotonic()
            received[0] += 1
            if not anchor:
                anchor.append(packet.sent)
            if controller is not None:
                controller.observe_packet(packet.sent, arrived, len(packet.data), packet.keyframe)
                if (decision := controller.decide(arrived)) is not None:
                    channel.send_json(b'S', {'rung': decision[0], 'skip': decision[2]})
                if controller.capacity and abs(controller.capacity - reported[0]) > 0.1 * reported[0]:
                    reported[0] = controller.capacity
                    # The server's guard drops only frames that would arrive too late to be shown.
                    channel.send_json(b'C', {'kbps': round(controller.capacity, 1), 'budget_ms': round(
                        (args.playout_delay + args.drop_late - args.processing_margin) * 1000, 1)})
            packets.put((packet, arrived))

    def decode():
        decoder, rung, previous = None, None, None
        while (entry := packets.get()) is not END:
            packet, arrived = entry
            if (packet.rendition, packet.crf) != rung:
                if not packet.keyframe:
                    raise RuntimeError(f'switch to {packet.rendition} at frame {packet.index} is not on a keyframe')
                decoder, rung = MotionDecoder(), (packet.rendition, packet.crf)
            # A gap is allowed only up to a keyframe: the server dropped the rest of a
            # group to drain the queue, and the keyframe starts a clean one.
            if previous is not None and packet.index != previous + 1 and not packet.keyframe:
                raise RuntimeError(f'frame {previous + 1} missing')
            previous = packet.index
            begin = time.perf_counter()
            for frame in decoder.decode(packet.data):
                decoded.put({'index': packet.index, 'rendition': packet.rendition, 'crf': packet.crf,
                             'rgb': frame.rgb, 'mv': frame.mv, 'is_intra': frame.is_intra,
                             'bytes': len(packet.data), 'sent': packet.sent, 'arrived': arrived,
                             'decode_s': time.perf_counter() - begin})

    def reconstruct_loop():
        steps = list(schedule)
        while (item := decoded.get()) is not END:
            while steps and steps[0][0] <= item['index']:
                channel.send_json(b'S', {'rung': steps.pop(0)[1]})
            started = time.monotonic()
            frame, info = reconstruct(item, due(item['index']) - started)
            item.update(info, started=started, reconstructed=time.monotonic(), output=frame)
            if args.compare and item['rendition'] != 'direct':
                item['input'] = item['rgb']  # for the bicubic half of the side-by-side view
            del item['rgb'], item['mv']
            shown.put(item)

    threads = [guarded('receive', receive), guarded('decode', decode), guarded('reconstruct', reconstruct_loop)]
    recorder = Recorder(args.record) if args.record else None
    cpu_begin, wall_begin, unix_begin = time.process_time(), time.monotonic(), time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for thread in threads:
        thread.start()

    window = 'UpStream live' if cv2 is not None else None
    recent_bytes: list[tuple[float, int]] = []
    late_count = 0
    try:
        while (item := shown.get()) is not END:
            target = due(item['index'])
            wait = target - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            late = time.monotonic() - target
            dropped = late > args.drop_late
            late_count += dropped
            now = time.monotonic()
            recent_bytes = [(t, b) for t, b in recent_bytes if t > now - 1.0] + [(now, item['bytes'])]
            if not dropped and window:
                picture = cv2.cvtColor(item.pop('output'), cv2.COLOR_RGB2BGR)
                name = rung_name(item['rendition'], item['crf'])
                size = {'direct': '720p', 'x2': '360p', 'x4': '180p'}[item['rendition']]
                if item['exit'] is None:
                    model = 'no model (sent as 720p)'
                else:
                    model = (f"{baseline.name} {size} -> 720p, {item['model_s'] * 1000:.1f} ms" if baseline else
                             f"UpStreamVSR {size} -> 720p, exit {item['exit'] + 1}, {item['model_s'] * 1000:.1f} ms")
                capacity = f'{controller.capacity:.0f} kbit/s' if controller and controller.capacity else 'measuring'
                lines = [f'{name}   {model}',
                         f'sent {sum(b for _, b in recent_bytes) * 8 / 1000:.0f} kbit/s   link estimate {capacity}',
                         f"latency {(now - anchor[0] - item['index'] / fps) * 1000:.0f} ms   frame {item['index']}   "
                         f"late {late_count}"]
                if args.compare:
                    if 'input' in item:
                        left = cv2.resize(cv2.cvtColor(item.pop('input'), cv2.COLOR_RGB2BGR),
                                          (picture.shape[1], picture.shape[0]), interpolation=cv2.INTER_CUBIC)
                        left_label = f'bicubic from {size}'
                    else:
                        left, left_label = picture.copy(), 'direct 720p'
                    picture = np.hstack([overlay(cv2, left, [left_label]), overlay(cv2, picture, lines)])
                elif args.hud:
                    picture = overlay(cv2, picture, lines)
                cv2.imshow(window, picture)
                if cv2.waitKey(1) & 255 == 27:
                    raise KeyboardInterrupt('closed with Escape')
            item.pop('input', None)
            output = item.pop('output', None)
            if recorder is not None and output is not None:
                item['recorded'] = recorder.add(output)
            displayed = time.monotonic()
            item.update(dropped=dropped, displayed=displayed, late_s=late,
                        latency_s=displayed - (anchor[0] + item['index'] / fps),
                        ready_s=item['reconstructed'] - (anchor[0] + item['index'] / fps))
            report['frames'].append(item)
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
        for thread in threads:
            thread.join(timeout=5)
        if window:
            import cv2
            cv2.destroyAllWindows()
        if recorder is not None:
            recorder.close()
    report['resources'] = {
        'client_cpu_s': round(time.process_time() - cpu_begin, 3), 'wall_s': round(time.monotonic() - wall_begin, 3),
        'unix_begin': unix_begin, 'unix_end': time.time(),
        'peak_gpu_mib': round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1) if device.type == 'cuda' else None,
        'model_gpu_s': round(sum(f.get('model_s', 0.0) for f in report['frames']), 3)}
    if recorder is not None:
        report['record'] = {'path': str(args.record), 'frames': recorder.count, 'shape': recorder.shape}
    report['bytes_received'] = channel.bytes_received
    if controller is not None:
        report['controller']['decisions'] = controller.trace
    report['summary'] = summarise(report['frames'], fps)
    if report['summary']:
        report['summary']['skipped'] = report['skipped']
    return report


def summarise(frames: list[dict], fps: float) -> dict:
    if not frames:
        return {}

    def stats(values):
        values = np.asarray(values, dtype=float) * 1000
        return {'mean_ms': round(float(values.mean()), 2), 'p50_ms': round(float(np.median(values)), 2),
                'p95_ms': round(float(np.percentile(values, 95)), 2), 'max_ms': round(float(values.max()), 2)}

    summary = {'frames': len(frames), 'dropped': sum(f['dropped'] for f in frames),
               'latency': stats([f['latency_s'] for f in frames]),
               # When the frame was ready to show, against its release: the pipeline's
               # own latency, without the playout buffer.
               'ready_after_release': stats([f['ready_s'] for f in frames]),
               'arrival_after_send': stats([f['arrived'] - f['sent'] for f in frames]),
               'decode': stats([f['decode_s'] for f in frames]),
               'queue_before_reconstruct': stats([f['started'] - f['arrived'] for f in frames])}
    by_rendition = {}
    for name in ('direct', 'x2', 'x4'):
        chosen = [f for f in frames if f['rendition'] == name]
        if not chosen:
            continue
        entry = {'frames': len(chosen), 'kbps': round(sum(f['bytes'] for f in chosen) * 8 * fps / len(chosen) / 1000, 1)}
        if name != 'direct':
            entry['model'] = stats([f['model_s'] for f in chosen])
            entry['exits'] = {str(e): sum(f['exit'] == e for f in chosen) for e in sorted({f['exit'] for f in chosen})}
        by_rendition[name] = entry
    summary['by_rendition'] = by_rendition
    rungs = [rung_name(f['rendition'], f['crf']) for f in frames]
    summary['by_rung'] = {name: {'frames': rungs.count(name),
                                 'kbps': round(sum(f['bytes'] for f, r in zip(frames, rungs) if r == name)
                                               * 8 * fps / rungs.count(name) / 1000, 1)}
                          for name in dict.fromkeys(rungs)}
    summary['switches'] = [{'frame': frames[i]['index'], 'to': rungs[i]}
                           for i in range(1, len(frames)) if rungs[i] != rungs[i - 1]]
    summary['kbps'] = round(sum(f['bytes'] for f in frames) * 8 * fps / len(frames) / 1000, 1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, help='UpStreamVSR checkpoint; without it x2/x4 frames are shown as received')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--schedule', help='switch requests by frame, for example 0:x4,90:x2@28,180:direct')
    parser.add_argument('--controller', action='store_true', help='choose rungs with the trade-off controller')
    parser.add_argument('--rate-quality', type=Path, default=Path(__file__).resolve().parent.parent / 'configs/rate_quality.json',
                        help='table from scripts/measure_rate_quality.py, for --controller')
    parser.add_argument('--renditions', nargs='+', default=['direct', 'x2', 'x4'], choices=['direct', 'x2', 'x4'],
                        help='renditions the controller may choose, for example x4 alone')
    parser.add_argument('--record', type=Path, help='write every reconstructed frame here, raw RGB, for scoring')
    parser.add_argument('--gpu-share', type=float, default=0.6, help='share of the frame interval the model may use')
    parser.add_argument('--processing-margin', type=float, default=0.025,
                        help='seconds of the playout delay kept for decoding and reconstruction')
    parser.add_argument('--playout-delay', type=float, default=0.1, help='seconds of buffer after frame 0 arrives')
    parser.add_argument('--drop-late', type=float, default=0.05, help='skip showing a frame this much past due')
    parser.add_argument('--queue', type=int, default=8, help='frames each stage may buffer')
    parser.add_argument('--half', action='store_true', help='float16 inference')
    parser.add_argument('--upscaler', choices=['model', 'bicubic', *BASELINES], default='model',
                        help='bicubic (ablation) or a published model (upstream_vsr/baselines.py) upscales x2/x4 '
                             'frames instead of UpStreamVSR; the controller is unchanged and times it')
    parser.add_argument('--fixed-exit', type=int, help='ablation: always this exit (-1 = deepest), not the anytime choice')
    parser.add_argument('--no-gpu-awareness', dest='gpu_aware', action='store_false',
                        help='ablation: the controller assumes the deepest exit always fits')
    parser.add_argument('--display', action='store_true', help='show the video in a window (Escape closes it)')
    parser.add_argument('--no-hud', dest='hud', action='store_false', help='with --display: no status overlay')
    parser.add_argument('--compare', action='store_true',
                        help='with --display: bicubic from the received frame on the left, UpStreamVSR on the right')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--connect-timeout', type=float, default=30)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = run(args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        slim = dict(report)
        slim['frames'] = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in f.items()} for f in report['frames']]
        args.report.write_text(json.dumps(slim, indent=1, default=str) + '\n')
    print(json.dumps({key: report[key] for key in ('summary', 'errors') if key in report}, indent=2))
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
