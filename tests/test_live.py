import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from live.client import parse_schedule  # noqa: E402
from live.policy import DepthPolicy  # noqa: E402
from live.protocol import Channel, Packet, parse_packet  # noqa: E402
from upstream_vsr.codec import MotionDecoder, decode_with_motion  # noqa: E402
from upstream_vsr.encoding import X264, x264_live  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='needs ffmpeg with libx264')


def test_packet_round_trip():
    a, b = socket.socketpair()
    with a, b:
        sender, receiver = Channel(a), Channel(b)
        packet = Packet('x2', 1234, True, 28, 12.5, b'\x00\x00\x01payload')
        sender.send_packet(packet)
        sender.send_json(b'S', {'rendition': 'x4'})
        kind, payload = receiver.receive()
        assert kind == b'P' and parse_packet(payload) == packet
        assert receiver.receive() == (b'S', b'{"rendition": "x4"}')
        a.shutdown(socket.SHUT_WR)  # close() alone waits for the makefile reader
        assert receiver.receive() is None


def test_policy_picks_deepest_exit_that_fits():
    policy = DepthPolicy(3, frame_interval=1 / 30, margin=0.0)
    for exit_index, seconds in enumerate((0.005, 0.010, 0.020)):
        policy.update(4, exit_index, seconds)
    assert policy.choose(4, 1.0) == 2  # plenty of time, capped at one frame interval
    assert policy.choose(4, 0.015) == 1
    assert policy.choose(4, 0.001) == 0  # nothing fits: cheapest
    assert policy.choose(2, 1.0) == 0  # no measurements for this scale yet


def test_policy_follows_measurements():
    policy = DepthPolicy(2, frame_interval=1 / 30, margin=0.0, smoothing=0.5)
    policy.update(4, 1, 0.010)
    policy.update(4, 1, 0.050)
    assert policy.expected[(4, 1)] == pytest.approx(0.030)


def test_schedule_parsing():
    assert parse_schedule('90:x2,0:x4@28,180:direct') == [(0, 'x4@28'), (90, 'x2'), (180, 'direct')]
    with pytest.raises(ValueError):
        parse_schedule('0:x3')


def test_live_settings_only_add_repeated_headers():
    live = x264_live()
    assert len(live) == len(X264)
    differences = [(a, b) for a, b in zip(X264, live) if a != b]
    assert differences == [(differences[0][0], differences[0][0] + ':repeat-headers=1')]


@needs_ffmpeg
def test_packet_decoder_matches_training_decoder():
    """The live client must see exactly the pixels and vectors the model was trained on."""
    import av
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'clip.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=320x180:r=30', '-t', '2',
                        *x264_live(), '-crf', '28', str(path)], check=True)
        reference = list(decode_with_motion(path))
        decoder, live = MotionDecoder(), []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for packet in container.demux(stream):
                if packet.size:
                    live += decoder.decode(bytes(av.BitStreamFilterContext('h264_mp4toannexb', stream).filter(packet)[0]))
    assert len(live) == len(reference) == 60
    for a, b in zip(live, reference):
        assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.mv, b.mv) and a.is_intra == b.is_intra


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


@needs_ffmpeg
def test_live_stream_switches_on_keyframes():
    """Server and client end to end, without a model: switches land on keyframes."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        source = temp / 'source.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=1280x720:r=30', '-t', '3',
                        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(source)], check=True)
        port = free_port()
        server = subprocess.Popen([sys.executable, str(ROOT / 'live/server.py'), str(source), '--port', str(port),
                                   '--start', 'x4', '--report', str(temp / 'server.json')],
                                  stdout=subprocess.PIPE, text=True)
        try:
            assert server.stdout.readline().startswith('READY')
            client = subprocess.run([sys.executable, str(ROOT / 'live/client.py'), '--port', str(port), '--device', 'cpu',
                                     '--schedule', '0:x4,10:x2,40:direct', '--playout-delay', '0.2',
                                     '--report', str(temp / 'client.json')], capture_output=True, text=True, timeout=60)
            assert client.returncode == 0, client.stdout + client.stderr
            server.wait(timeout=30)
        finally:
            if server.poll() is None:
                server.kill()
        served = json.loads((temp / 'server.json').read_text())
        played = json.loads((temp / 'client.json').read_text())
    assert served['frames'] == 90 and played['summary']['frames'] == 90
    assert served['switches'] == [{'frame': 30, 'from': 'x4@23', 'to': 'x2@23'},
                                  {'frame': 60, 'from': 'x2@23', 'to': 'direct@23'}]
    assert [f['rendition'] for f in played['frames']] == ['x4'] * 30 + ['x2'] * 30 + ['direct'] * 30
    assert played['summary']['dropped'] == 0


@needs_ffmpeg
def test_controller_drives_the_live_stream_through_a_link_drop():
    """Controller and emulated link end to end, without a model: switches and skips land on keyframes."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        source = temp / 'source.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=1280x720:r=30', '-t', '8',
                        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(source)], check=True)
        table = {f'{r}@{c}': {'kbps': 100.0, 'psnr': [40.0 - c / 2] * (1 if r == 'direct' else 3)}
                 for r in ('direct', 'x2', 'x4') for c in (23, 43)}
        (temp / 'table.json').write_text(json.dumps({'rungs': table}))
        port = free_port()
        server = subprocess.Popen([sys.executable, str(ROOT / 'live/server.py'), str(source), '--port', str(port),
                                   '--start', 'direct@43', '--crfs', '23', '43', '--direct-crfs', '23', '43',
                                   '--link', '0:20000,4:300', '--report', str(temp / 'server.json')],
                                  stdout=subprocess.PIPE, text=True)
        try:
            assert server.stdout.readline().startswith('READY')
            client = subprocess.run([sys.executable, str(ROOT / 'live/client.py'), '--port', str(port), '--device', 'cpu',
                                     '--controller', '--rate-quality', str(temp / 'table.json'),
                                     '--playout-delay', '0.2', '--report', str(temp / 'client.json')],
                                    capture_output=True, text=True, timeout=90)
            assert client.returncode == 0, client.stdout + client.stderr
            server.wait(timeout=60)
        finally:
            if server.poll() is None:
                server.kill()
        served = json.loads((temp / 'server.json').read_text())
        played = json.loads((temp / 'client.json').read_text())
    decisions = played['controller']['decisions']
    assert decisions and decisions[0]['to'] == 'direct@23'  # a fast link: the best direct rung
    assert all(s['frame'] % 30 == 0 for s in served['switches'])
    assert played['summary']['frames'] + played['skipped'] == served['frames'] == 240
    assert played['skipped'] == served['skipped']
    assert played['summary']['by_rung'].get('direct@43', {}).get('frames', 0) > 0  # back down after the drop
