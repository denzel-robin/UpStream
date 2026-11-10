#!/usr/bin/env python3
"""Live server: encode direct, x2 and x4 renditions at once and send the one asked for.

One ffmpeg reads the source at its own frame rate (-re) and produces all three
renditions with the training filter graph and x264 settings, keyframes aligned
every 30 frames. Each rendition comes back through its own pipe as NUT, whose
packets carry their own length, so a packet is available as soon as it is written.

Packets are released on the source's schedule: frame i leaves at t0 + i / fps,
where t0 is when frame 0 was ready, as a camera would deliver it. ffmpeg's start-up
burst is switched off, so the encoder does not run far ahead.

The client asks for a rendition with an S record. The server switches at the next
frame that is a keyframe in the requested rendition; until then it keeps sending
the current one. Only the selected rendition goes on the wire, so the bytes sent
are what the chosen bandwidth costs.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from live.protocol import LABEL, RENDITIONS, Channel, Packet  # noqa: E402
from upstream_vsr.encoding import SIZES, filter_graph, frame_step, x264_live  # noqa: E402
from prepare_streamsr import probe  # noqa: E402


def start_encoder(source: Path, fps: float, crf: int, seconds: float | None) -> tuple[subprocess.Popen, dict[str, int]]:
    """Start ffmpeg; returns the process and a readable pipe per rendition."""
    pipes = {name: os.pipe() for name in RENDITIONS}
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-re', '-readrate_initial_burst', '0']
    if seconds:
        command += ['-t', str(seconds)]
    command += ['-i', str(source), '-filter_complex', filter_graph(fps)]
    for name, (_, write) in pipes.items():
        command += ['-map', f'[{LABEL[name]}]', *x264_live(), '-crf', str(crf),
                    '-flush_packets', '1', '-f', 'nut', f'pipe:{write}']
    process = subprocess.Popen(command, pass_fds=[write for _, write in pipes.values()])
    for _, write in pipes.values():
        os.close(write)
    return process, {name: read for name, (read, _) in pipes.items()}


def read_rendition(name: str, fd: int, out: queue.Queue, crf: int) -> None:
    """Demux one rendition's NUT pipe into Packets, in order. None marks the end."""
    try:
        # Unbuffered, so a read returns as soon as any bytes arrive.
        with os.fdopen(fd, 'rb', buffering=0) as pipe, av.open(pipe, format='nut') as container:
            index = 0
            for packet in container.demux(container.streams.video[0]):
                if packet.size == 0:
                    continue
                out.put(Packet(name, index, bool(packet.is_keyframe), crf, 0.0, bytes(packet)))
                index += 1
    finally:
        out.put(None)


def serve(args: argparse.Namespace) -> dict:
    info = probe(args.source)
    if info is None:
        raise SystemExit(f'cannot read {args.source}')
    fps = info[2] / frame_step(info[2])
    if args.start not in RENDITIONS:
        raise SystemExit(f'--start must be one of {RENDITIONS}')
    report = {'source': str(args.source), 'fps': fps, 'crf': args.crf, 'start': args.start,
              'switches': [], 'frames_by_rendition': dict.fromkeys(RENDITIONS, 0),
              'bytes_by_rendition': dict.fromkeys(RENDITIONS, 0), 'late_releases': 0, 'max_release_lag_s': 0.0}

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        listener.settimeout(args.accept_timeout)
        print(f'READY {args.host}:{args.port}', flush=True)
        conn, _ = listener.accept()
    with conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        channel = Channel(conn)
        process, pipes = start_encoder(args.source, fps, args.crf, args.seconds)
        epoch = time.monotonic()
        channel.send_json(b'H', {'fps': fps, 'crf': args.crf, 'epoch': epoch, 'start': args.start,
                                 'sizes': {name: SIZES[LABEL[name]] for name in RENDITIONS}})
        queues = {name: queue.Queue() for name in RENDITIONS}
        readers = [threading.Thread(target=read_rendition, args=(name, pipes[name], queues[name], args.crf),
                                    daemon=True) for name in RENDITIONS]
        for reader in readers:
            reader.start()

        requested = [args.start]

        def listen() -> None:
            while True:
                record = channel.receive()
                if record is None:
                    return
                kind, payload = record
                if kind == b'S':
                    wanted = json.loads(payload)['rendition']
                    if wanted in RENDITIONS:
                        requested[0] = wanted

        threading.Thread(target=listen, daemon=True).start()

        current, index, t0 = args.start, 0, None
        try:
            while True:
                packets = {name: queues[name].get() for name in RENDITIONS}
                if any(packet is None for packet in packets.values()):
                    break
                wanted = requested[0]
                if wanted != current and packets[wanted].keyframe:
                    report['switches'].append({'frame': index, 'from': current, 'to': wanted})
                    current = wanted
                now = time.monotonic()
                if t0 is None:
                    t0 = now
                elif (wait := t0 + index / fps - now) > 0:
                    time.sleep(wait)
                elif -wait > 1 / fps:  # the encoder fell more than a frame behind the source
                    report['late_releases'] += 1
                    report['max_release_lag_s'] = max(report['max_release_lag_s'], -wait)
                packet = packets[current]._replace(sent=time.monotonic())
                if index == 0 and not packet.keyframe:
                    raise RuntimeError('stream does not start with a keyframe')
                channel.send_packet(packet)
                report['frames_by_rendition'][current] += 1
                report['bytes_by_rendition'][current] += len(packet.data)
                index += 1
            report['returncode'] = process.wait(timeout=30)
            channel.send_json(b'E', {'frames': index, 'returncode': report['returncode']})
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    report['frames'] = index
    report['duration_s'] = index / fps
    report['payload_bitrate_kbps'] = sum(report['bytes_by_rendition'].values()) * 8 / 1000 / max(index / fps, 1e-9)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path)
    parser.add_argument('--crf', type=int, default=23)
    parser.add_argument('--seconds', type=float, help='stream only the start of the source')
    parser.add_argument('--start', default='x4', help='first rendition: direct, x2 or x4')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--accept-timeout', type=float, default=60)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = serve(args)
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n')


if __name__ == '__main__':
    main()
