#!/usr/bin/env python3
"""Live server: encode every rung of the ladder at once and send the one asked for.

A rung is a rendition (direct, x2, x4) at a CRF. One ffmpeg reads the source at its
own frame rate (-re) and produces every rung with the training filter graph and x264
settings, keyframes aligned every 30 frames. Each rung comes back through its own
pipe as NUT, whose packets carry their own length, so a packet is available as soon
as it is written.

Packets are released on the source's schedule: frame i leaves at t0 + i / fps,
where t0 is when frame 0 was ready, as a camera would deliver it. ffmpeg's start-up
burst is switched off, so the encoder does not run far ahead.

The client asks for a rung with an S record. The server switches at the next frame
that is a keyframe in the requested rung; until then it keeps sending the current
one. With "skip" in the request it also stops sending until that keyframe, so a
queue that has built up in the network drains (the client holds its last frame).
Only the selected rung goes on the wire, so the bytes sent are what the chosen
bandwidth costs. At every keyframe an R record tells the client what each rung cost
over the last second, so the controller can price the options on this content.

The server also guards latency itself (live/backlog.py): it knows how much it has
sent that the network has not delivered, long before the client sees the delay. If
that backlog would take longer to drain than the client can wait for a frame
(playout delay plus drop-late less processing; --max-backlog-ms until it says) at the client's capacity estimate (both in C records), it drops frames up to the next keyframe instead of
queueing them, and says so with a G record.

--link puts an emulated bottleneck (live/link.py) between server and client.
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
from live.backlog import backlog_bytes  # noqa: E402
from live.link import EmulatedLink, parse_link  # noqa: E402
from live.protocol import LABEL, RENDITIONS, Channel, Packet, parse_rung, rung_name  # noqa: E402
from upstream_vsr.encoding import KEYINT, SIZES, frame_step, ladder_graph, x264_live  # noqa: E402
from prepare_streamsr import probe  # noqa: E402


def build_ladder(crfs: list[int], direct_crfs: list[int]) -> list[tuple[str, int]]:
    return ([('direct', crf) for crf in direct_crfs] + [('x2', crf) for crf in crfs] + [('x4', crf) for crf in crfs])


def start_encoder(source: Path, source_fps: float, ladder: list[tuple[str, int]],
                  seconds: float | None) -> tuple[subprocess.Popen, dict[str, int]]:
    """Start ffmpeg; returns the process and a readable pipe per rung.

    source_fps is the source's own rate: the graph halves 45+ fps sources itself.
    """
    pipes = {rung_name(*rung): os.pipe() for rung in ladder}
    copies = {label: sum(LABEL[rendition] == label for rendition, _ in ladder) for label in LABEL.values()}
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-re', '-readrate_initial_burst', '0']
    if seconds:
        command += ['-t', str(seconds)]
    command += ['-i', str(source), '-filter_complex', ladder_graph(source_fps, copies)]
    used = dict.fromkeys(LABEL.values(), 0)
    for rendition, crf in ladder:
        label = LABEL[rendition]
        write = pipes[rung_name(rendition, crf)][1]
        command += ['-map', f'[{label}_{used[label]}]', *x264_live(), '-crf', str(crf),
                    '-flush_packets', '1', '-f', 'nut', f'pipe:{write}']
        used[label] += 1
    process = subprocess.Popen(command, pass_fds=[write for _, write in pipes.values()])
    for _, write in pipes.values():
        os.close(write)
    return process, {name: read for name, (read, _) in pipes.items()}


def read_rung(rendition: str, crf: int, fd: int, out: queue.Queue) -> None:
    """Demux one rung's NUT pipe into Packets, in order. None marks the end."""
    try:
        # Unbuffered, so a read returns as soon as any bytes arrive.
        with os.fdopen(fd, 'rb', buffering=0) as pipe, av.open(pipe, format='nut') as container:
            index = 0
            for packet in container.demux(container.streams.video[0]):
                if packet.size == 0:
                    continue
                out.put(Packet(rendition, index, bool(packet.is_keyframe), crf, 0.0, bytes(packet)))
                index += 1
    finally:
        out.put(None)


def serve(args: argparse.Namespace) -> dict:
    info = probe(args.source)
    if info is None:
        raise SystemExit(f'cannot read {args.source}')
    fps = info[2] / frame_step(info[2])
    crfs = args.crfs or [args.crf]
    direct_crfs = args.direct_crfs or [args.crf]
    ladder = build_ladder(crfs, direct_crfs)
    names = [rung_name(*rung) for rung in ladder]

    def resolve(text: str) -> str | None:
        """A requested rung, or the nearest CRF of the requested rendition."""
        try:
            rendition, crf = parse_rung(text, args.crf)
        except ValueError:
            return None
        options = [c for r, c in ladder if r == rendition]
        return rung_name(rendition, min(options, key=lambda c: abs(c - crf))) if options else None

    start = resolve(args.start)
    if start is None:
        raise SystemExit(f'--start {args.start!r} is not in the ladder {names}')
    link_schedule = parse_link(args.link) if args.link else None
    report = {'source': str(args.source), 'fps': fps, 'crf': args.crf, 'ladder': names, 'start': start,
              'link': link_schedule, 'switches': [], 'frames_by_rung': {}, 'bytes_by_rung': {},
              'late_releases': 0, 'max_release_lag_s': 0.0, 'skipped': 0, 'backlog_skips': 0,
              'max_backlog_ms': args.max_backlog_ms, 'requests': [], 'max_encoder_lead_s': 0.0}

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        listener.settimeout(args.accept_timeout)
        print(f'READY {args.host}:{args.port}', flush=True)
        conn, _ = listener.accept()
    with conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        link = EmulatedLink(conn, link_schedule) if link_schedule else None
        channel = Channel(conn, link)
        client_kbps = [None]  # the client's capacity estimate (C records), for the backlog guard
        limit_ms = [args.max_backlog_ms]  # replaced by the client's latency budget when it sends one
        process, pipes = start_encoder(args.source, info[2], ladder, args.seconds)
        channel.send_json(b'H', {'fps': fps, 'crf': args.crf, 'start': start, 'ladder': names, 'keyint': KEYINT,
                                 'sizes': {name: SIZES[LABEL[name]] for name in RENDITIONS}})
        queues = {name: queue.Queue() for name in names}
        for rendition, crf in ladder:
            name = rung_name(rendition, crf)
            threading.Thread(target=read_rung, args=(rendition, crf, pipes[name], queues[name]), daemon=True).start()

        requested = [start]
        skipping = [False]  # drop frames until the next keyframe, so the client's queue drains

        def listen() -> None:
            while True:
                record = channel.receive()
                if record is None:
                    return
                kind, payload = record
                if kind == b'S':
                    message = json.loads(payload)
                    wanted = resolve(message.get('rung') or message.get('rendition', ''))
                    report['requests'].append({'at': time.monotonic(), 'rung': wanted, 'skip': bool(message.get('skip'))})
                    if wanted is not None:
                        requested[0] = wanted
                        if message.get('skip'):
                            skipping[0] = True
                elif kind == b'C':
                    message = json.loads(payload)
                    client_kbps[0] = message.get('kbps')
                    if message.get('budget_ms'):
                        limit_ms[0] = message['budget_ms']

        threading.Thread(target=listen, daemon=True).start()

        recent = {name: [] for name in names}  # packet sizes of the last KEYINT frames
        key_bytes = {}  # size of each rung's latest keyframe
        current, index, t0 = start, 0, None
        previous_backlog = 0
        try:
            while True:
                packets = {name: queues[name].get() for name in names}
                if any(packet is None for packet in packets.values()):
                    break
                for name, packet in packets.items():
                    recent[name] = recent[name][-(KEYINT - 1):] + [len(packet.data)]
                    if packet.keyframe:
                        key_bytes[name] = len(packet.data)
                # Wait for the frame's release time before acting on it. The encoder can
                # run ahead of the release clock (frames read while ffmpeg starts up come
                # out in a burst), and a switch or skip decided on dequeue would then
                # apply to frames that are still seconds from release. The guard below
                # also needs the backlog as it is when the frame would be sent.
                now = time.monotonic()
                if t0 is None:
                    t0 = report['t0'] = now
                    if link is not None:
                        link.start_clock()
                elif (wait := t0 + index / fps - now) > 0:
                    report['max_encoder_lead_s'] = max(report['max_encoder_lead_s'], wait)
                    time.sleep(wait)
                elif -wait > 1 / fps:  # the encoder fell more than a frame behind the source
                    report['late_releases'] += 1
                    report['max_release_lag_s'] = max(report['max_release_lag_s'], -wait)
                wanted = requested[0]
                if packets[wanted].keyframe:
                    skipping[0] = False
                    if wanted != current:
                        report['switches'].append({'frame': index, 'from': current, 'to': wanted,
                                                   'at': time.monotonic()})
                        current = wanted
                elif skipping[0]:
                    report['skipped'] += 1
                    index += 1
                    continue
                # Sender-side guard: do not add to a backlog that would take longer than
                # --max-backlog-ms to drain. Dropping one frame breaks the rest of its
                # group, so drop up to the next keyframe.
                # Only while the backlog is still growing, that is, sending this frame
                # would leave more queued than after the previous one: the stream is
                # faster than the link. A shrinking backlog (the tail of a keyframe
                # burst) drains by itself, and dropping the group would freeze the
                # picture for longer than the late frames would.
                waiting, rate = backlog_bytes(conn, link), client_kbps[0]
                growing = waiting + len(packets[current].data) > previous_backlog
                if rate and index > 0 and growing and waiting * 8 / (rate * 1000) > limit_ms[0] / 1000:
                    skipping[0] = True
                    channel.send_json(b'G', {'frame': index, 'capacity_kbps': rate, 'backlog_bytes': waiting})
                    report['backlog_skips'] += 1
                    report['skipped'] += 1
                    index += 1
                    continue
                packet = packets[current]._replace(sent=time.monotonic())
                if index == 0 and not packet.keyframe:
                    raise RuntimeError('stream does not start with a keyframe')
                if packet.keyframe and index > 0:
                    channel.send_json(b'R', {'frame': index, 'kbps': {
                        name: round(sum(sizes) * 8 * fps / len(sizes) / 1000, 1) for name, sizes in recent.items()},
                        'key_bytes': key_bytes})
                channel.send_packet(packet)
                previous_backlog = backlog_bytes(conn, link)  # including this packet
                report['frames_by_rung'][current] = report['frames_by_rung'].get(current, 0) + 1
                report['bytes_by_rung'][current] = report['bytes_by_rung'].get(current, 0) + len(packet.data)
                index += 1
            report['returncode'] = process.wait(timeout=30)
            channel.send_json(b'E', {'frames': index, 'returncode': report['returncode']})
            if link is not None:
                link.close()
                report['link_max_delay_s'] = round(link.max_delay_s, 4)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    report['frames'] = index
    report['duration_s'] = index / fps
    report['payload_bitrate_kbps'] = sum(report['bytes_by_rung'].values()) * 8 / 1000 / max(index / fps, 1e-9)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path)
    parser.add_argument('--crf', type=int, default=23, help='default CRF, and the only one without --crfs')
    parser.add_argument('--crfs', type=int, nargs='+', help='x2 and x4 CRFs in the ladder, for example 18 23 28 33')
    parser.add_argument('--direct-crfs', type=int, nargs='+', help='direct CRFs in the ladder, for example 23 28 33 38 43')
    parser.add_argument('--seconds', type=float, help='stream only the start of the source')
    parser.add_argument('--start', default='x4', help='first rung, for example x4@28, or a rendition at --crf')
    parser.add_argument('--max-backlog-ms', type=float, default=100,
                        help='skip to the next keyframe rather than queue more than this in the network, '
                             'until the client sends its latency budget')
    parser.add_argument('--link', help='emulated bottleneck, kbit/s from each second on, for example 0:3000,10:600')
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
