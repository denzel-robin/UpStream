#!/usr/bin/env python3
"""Live client: receive, decode with motion vectors, reconstruct causally, play on time.

Threads, joined by small bounded queues:
  receive      socket records -> encoded packets
  decode       one MotionDecoder per rendition run -> RGB, motion field, frame type
  reconstruct  UpStreamVSR.step on the GPU, exit chosen per frame by DepthPolicy;
               direct frames pass through
  main         waits until each frame is due, shows or skips it, records timings

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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from live.policy import DepthPolicy  # noqa: E402
from live.protocol import LABEL, SCALE, Channel, parse_packet  # noqa: E402
from upstream_vsr.encoding import SIZES  # noqa: E402
from upstream_vsr.codec import MotionDecoder  # noqa: E402
from upstream_vsr.model import UpStreamVSR, crf_to_quality  # noqa: E402

END = object()


def load_model(path: Path, device: torch.device, half: bool) -> UpStreamVSR:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = UpStreamVSR(**checkpoint['config'])
    model.load_checkpoint_state(checkpoint['model'])
    model = model.eval().fuse().to(device)
    return model.half() if half else model


def parse_schedule(text: str | None) -> list[tuple[int, str]]:
    """'0:x4,90:x2,180:direct' -> [(0, 'x4'), (90, 'x2'), (180, 'direct')]."""
    if not text:
        return []
    items = [(int(frame), name) for frame, name in (part.split(':') for part in text.split(','))]
    for _, name in items:
        if name not in SCALE:
            raise ValueError(f'unknown rendition {name}')
    return sorted(items)


class Reconstructor:
    """Holds the model and its recurrent state; one call per decoded frame."""

    def __init__(self, model: UpStreamVSR | None, device: torch.device, policy: DepthPolicy | None, half: bool):
        self.model, self.device, self.policy, self.half = model, device, policy, half
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

    def __call__(self, item: dict, time_left: float) -> tuple[np.ndarray, dict]:
        scale = SCALE[item['rendition']]
        if scale == 1 or self.model is None:
            self.state = None  # direct frames carry no model state
            self.last_scale = scale
            return item['rgb'], {'exit': None, 'model_s': 0.0}
        if self.last_scale == 1:
            self.state = None
        self.last_scale = scale
        exit_index = self.policy.choose(scale, time_left)
        frame, seconds = self.run(item['rgb'], item['mv'], item['is_intra'], scale, exit_index, item['crf'])
        return frame, {'exit': exit_index, 'model_s': seconds}


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, args.half) if args.checkpoint else None
    schedule = parse_schedule(args.schedule)
    report = {'checkpoint': str(args.checkpoint) if args.checkpoint else None, 'half': args.half,
              'playout_delay_s': args.playout_delay, 'drop_late_s': args.drop_late,
              'schedule': schedule, 'frames': [], 'errors': []}
    policy = None
    reconstruct = Reconstructor(model, device, None, args.half)
    if model is not None:
        report['model_config'] = model.config()
        # Frame interval is not known until the header; 30 fps until then.
        policy = reconstruct.policy = DepthPolicy(len(model.exits), 1 / 30)
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

    def receive():
        while True:
            record = channel.receive()
            if record is None:
                raise EOFError('server closed without an end record')
            kind, payload = record
            if kind == b'E':
                report['end'] = json.loads(payload)
                return
            if kind != b'P':
                raise ValueError(f'unexpected record {kind!r}')
            packet = parse_packet(payload)
            arrived = time.monotonic()
            if not anchor:
                anchor.append(packet.sent)
            packets.put((packet, arrived))

    def decode():
        decoder, rendition, previous = None, None, None
        while (entry := packets.get()) is not END:
            packet, arrived = entry
            if packet.rendition != rendition:
                if not packet.keyframe:
                    raise RuntimeError(f'switch to {packet.rendition} at frame {packet.index} is not on a keyframe')
                decoder, rendition = MotionDecoder(), packet.rendition
            if previous is not None and packet.index != previous + 1:
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
                channel.send_json(b'S', {'rendition': steps.pop(0)[1]})
            started = time.monotonic()
            frame, info = reconstruct(item, due(item['index']) - started)
            item.update(info, started=started, reconstructed=time.monotonic(), output=frame)
            del item['rgb'], item['mv']
            shown.put(item)

    threads = [guarded('receive', receive), guarded('decode', decode), guarded('reconstruct', reconstruct_loop)]
    for thread in threads:
        thread.start()

    window = None
    if args.display:
        import cv2
        window = 'UpStream live'
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        while (item := shown.get()) is not END:
            target = due(item['index'])
            wait = target - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            late = time.monotonic() - target
            dropped = late > args.drop_late
            if not dropped and window:
                import cv2
                cv2.imshow(window, cv2.cvtColor(item.pop('output'), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 255 == 27:
                    raise KeyboardInterrupt('closed with Escape')
            item.pop('output', None)
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
    report['bytes_received'] = channel.bytes_received
    report['summary'] = summarise(report['frames'], fps)
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
    renditions = [f['rendition'] for f in frames]
    summary['switches'] = [{'frame': frames[i]['index'], 'to': renditions[i]}
                           for i in range(1, len(frames)) if renditions[i] != renditions[i - 1]]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, help='UpStreamVSR checkpoint; without it x2/x4 frames are shown as received')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--schedule', help='switch requests by frame, for example 0:x4,90:x2,180:direct')
    parser.add_argument('--playout-delay', type=float, default=0.1, help='seconds of buffer after frame 0 arrives')
    parser.add_argument('--drop-late', type=float, default=0.05, help='skip showing a frame this much past due')
    parser.add_argument('--queue', type=int, default=8, help='frames each stage may buffer')
    parser.add_argument('--half', action='store_true', help='float16 inference')
    parser.add_argument('--display', action='store_true', help='show the video in a window')
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
