import socket
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from live.controller import DeliveryMonitor, TradeoffController  # noqa: E402
from live.link import EmulatedLink, parse_link  # noqa: E402
from live.policy import DepthPolicy  # noqa: E402
from live.simulate import simulate  # noqa: E402

# Shaped like the measured table: x4 is cheapest, the model adds to x2 and x4.
TABLE = {
    'direct@28': {'kbps': 2800, 'psnr': [34.9]},
    'direct@38': {'kbps': 1150, 'psnr': [30.2]},
    'direct@43': {'kbps': 830, 'psnr': [27.5]},
    'x2@23': {'kbps': 1580, 'psnr': [32.98, 33.04, 33.06]},
    'x2@28': {'kbps': 930, 'psnr': [30.79, 30.85, 30.88]},
    'x4@23': {'kbps': 540, 'psnr': [28.46, 28.59, 28.62]},
    'x4@28': {'kbps': 315, 'psnr': [26.57, 26.67, 26.69]},
    'x4@33': {'kbps': 195, 'psnr': [24.68, 24.73, 24.74]},
}
LADDER = list(TABLE)


def policy_with(ms_by_scale: dict[int, tuple[float, ...]]) -> DepthPolicy:
    policy = DepthPolicy(3, 1 / 30)
    for scale, times in ms_by_scale.items():
        for e, ms in enumerate(times):
            policy.expected[(scale, e)] = ms / 1000
    return policy


FAST = {2: (5.0, 6.3, 7.6), 4: (8.1, 9.5, 10.9)}


def feed_idle_link(monitor_or_controller, kbps: float, seconds: float = 2.0, fps: float = 30.0) -> float:
    """Packets at a quarter of the link rate, keyframes 4x larger; returns the end time."""
    observe = getattr(monitor_or_controller, 'observe_packet', None) or monitor_or_controller.add
    t = 0.0
    for i in range(int(seconds * fps)):
        t = i / fps
        other = int(kbps * 1000 / 8 / fps / 4)
        size = 4 * other if i % 30 == 0 else other
        observe(t, t + 0.005 + size * 8 / (kbps * 1000), size, i % 30 == 0)
    return t


def test_monitor_fits_capacity_and_fixed_delay_from_idle_packets():
    monitor = DeliveryMonitor()
    feed_idle_link(monitor, 1500)
    fixed, capacity = monitor.fit
    assert capacity == pytest.approx(1500, rel=1e-6)
    assert fixed == pytest.approx(0.005, abs=1e-9)
    assert monitor.standing_queue == pytest.approx(0, abs=1e-9)


def test_monitor_sees_standing_queue_and_delivery_rate_on_a_slow_link():
    monitor, free_at = DeliveryMonitor(), 0.0
    for i in range(60):  # 1200 kbit/s of frames into a 600 kbit/s link
        sent = i / 30
        free_at = max(sent, free_at) + 5000 * 8 / 600_000
        monitor.add(sent, free_at, 5000, False)
    assert monitor.standing_queue > 0.5
    assert monitor.delivery_kbps == pytest.approx(600, rel=0.01)


def test_controller_picks_best_quality_within_headroom():
    controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.1, policy=policy_with(FAST))
    controller.start('x4@28', 0.0)
    end = feed_idle_link(controller, 2000)
    rung, reason, skip = controller.decide(end)
    # 0.85 x 2000 = 1700: x2@23 (33.06 dB at 1580) beats direct@38 (30.2 dB at 1150).
    assert (rung, reason, skip) == ('x2@23', 'capacity', False)


def test_controller_leaves_the_model_when_the_gpu_is_too_slow():
    slow = {2: (30.0, 40.0, 50.0), 4: (30.0, 40.0, 50.0)}  # nothing fits 0.6 x 33 ms
    controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.1, policy=policy_with(slow))
    controller.start('x4@28', 0.0)
    end = feed_idle_link(controller, 2000)
    assert controller.decide(end)[0] == 'direct@38'
    assert controller.quality('x2@23') is None


def test_controller_credits_the_deepest_exit_that_fits():
    controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.1, policy=policy_with({2: (5.0, 15.0, 25.0)}))
    assert controller.quality('x2@23') == TABLE['x2@23']['psnr'][1]  # 25 ms > 20 ms budget


def test_controller_goes_to_x4_on_a_tight_link():
    controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.1, policy=policy_with(FAST))
    controller.start('x2@28', 0.0)
    end = feed_idle_link(controller, 450)
    assert controller.decide(end)[0] == 'x4@28'  # 0.85 x 450 = 382


def test_simulated_drop_drains_and_recovers():
    result = simulate(TABLE, LADDER, parse_link('0:3000,10:700,20:1800'), 30, 'x4@28')
    rungs = {second: [f['rung'] for f in result['frames'] if second <= f['sent'] < second + 1][-1]
             for second in (5, 15, 25)}
    assert rungs[5] == 'x2@23' and rungs[15] == 'x4@23' and rungs[25] in ('x2@23', 'x2@28')
    assert result['max_delay_ms'] < 800 and result['late'] < 20
    assert any(d.get('skip') for d in result['decisions'])


def test_simulated_link_never_congests_when_it_is_steady():
    result = simulate(TABLE, LADDER, parse_link('0:1200'), 20, 'x4@28')
    assert result['late'] == 0 and result['skipped'] == 0
    assert len(result['decisions']) <= 2


def test_emulated_link_paces_records():
    a, b = socket.socketpair()
    with a, b:
        link = EmulatedLink(a, parse_link('0:800'))
        begin = time.monotonic()
        for _ in range(10):
            link.send(b'x' * 5000)  # 400 kbit at 800 kbit/s: half a second
        link.close()
        elapsed = time.monotonic() - begin
        assert link.queued_bytes == 0
        received = 0
        b.settimeout(1)
        while received < 50000:
            received += len(b.recv(65536))
    assert 0.45 < elapsed < 0.7


def test_link_schedule_must_start_at_zero():
    assert parse_link('10:500,0:2000') == [(0.0, 2000.0), (10.0, 500.0)]
    with pytest.raises(ValueError):
        parse_link('5:1000')
