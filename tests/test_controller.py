import socket
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from live.controller import DeliveryMonitor, MpcController, TradeoffController  # noqa: E402
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


def feed_idle_link(monitor_or_controller, kbps: float, seconds: float = 2.0, fps: float = 30.0,
                   start: float = 0.0) -> float:
    """Packets at a quarter of the link rate, keyframes 4x larger; returns the end time."""
    observe = getattr(monitor_or_controller, 'observe_packet', None) or monitor_or_controller.add
    t = start
    for i in range(int(seconds * fps)):
        t = start + i / fps
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


def test_controller_runs_the_model_late_when_the_ladder_has_no_direct_rung():
    slow = {4: (30.0, 40.0, 50.0)}
    ladder = [n for n in LADDER if n.startswith('x4')]
    controller = TradeoffController(TABLE, ladder, 1 / 30, 0.1, policy=policy_with(slow))
    controller.start('x4@28', 0.0)
    end = feed_idle_link(controller, 2000)
    decision = controller.decide(end)
    assert decision is None or decision[0] in ladder


def test_controller_credits_the_deepest_exit_that_fits():
    controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.1, policy=policy_with({2: (5.0, 15.0, 25.0)}))
    assert controller.quality('x2@23') == TABLE['x2@23']['psnr'][1]  # 25 ms > 20 ms budget


def test_ablation_without_gpu_awareness_credits_the_deepest_exit_even_when_too_slow():
    slow = {2: (30.0, 40.0, 50.0), 4: (30.0, 40.0, 50.0)}
    controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.1, policy=policy_with(slow), gpu_aware=False)
    assert controller.quality('x2@23') == TABLE['x2@23']['psnr'][2]


def test_ablation_fixed_exit_ignores_the_time_left():
    policy = policy_with({4: (8.0, 12.0, 30.0)})
    assert policy.choose(4, 0.02) == 1
    policy.fixed = 2
    assert policy.choose(4, 0.001) == 2


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


def test_netem_path_conditions_build_tc_arguments():
    from live.netem import PathConditions
    path = PathConditions.parse('delay=40,jitter=15,loss=1,limit=200')
    assert path.netem(500) == ['netem', 'delay', '40.0ms', '15.0ms', 'distribution', 'normal', 'loss', '1.0%',
                               'rate', '500kbit', 'limit', '200']
    assert 'rate' not in PathConditions(jitter_ms=0, loss_pct=0).netem(None)
    with pytest.raises(ValueError):
        PathConditions.parse('latency=20')


def test_mpc_prediction_is_a_discounted_harmonic_mean():
    controller = MpcController(TABLE, LADDER, 1 / 30, 0.1, policy=policy_with(FAST))
    controller.samples.extend([1000, 500])
    controller.errors.extend([0.1, 0.25])
    assert controller.predict() == pytest.approx((2 / (1 / 1000 + 1 / 500)) / 1.25)


def test_mpc_plan_takes_the_best_rung_the_throughput_sustains():
    controller = MpcController(TABLE, LADDER, 1 / 30, 0.1, policy=policy_with(FAST))
    controller.start('x4@28', 0.0)
    options = {n: controller.quality(n) for n in LADDER}
    assert controller.plan(options, 2000, 0.0)[0] == 'x2@23'  # 1580 kbit/s fits, direct@28 does not
    assert controller.plan(options, 600, 0.0)[0] == 'x4@23'


def test_mpc_decides_once_per_group_of_pictures():
    controller = MpcController(TABLE, LADDER, 1 / 30, 0.1, policy=policy_with(FAST))
    controller.start('x4@28', 0.0)
    decisions = 0
    for i in range(90):
        t = i / 30
        size = 8000 if i % 30 == 0 else 2000
        controller.observe_packet(t, t + 0.005 + size * 8 / 2_000_000, size, i % 30 == 0)
        decisions += controller.decide(t + 0.01) is not None
    assert decisions == 1 and controller.current == 'x2@23'


def test_simulated_mpc_follows_a_drop():
    result = simulate(TABLE, LADDER, parse_link('0:3000,10:700,20:1800'), 30, 'x4@28', abr='mpc')
    rungs = {second: [f['rung'] for f in result['frames'] if second <= f['sent'] < second + 1][-1]
             for second in (8, 18)}
    assert TABLE[rungs[18]]['kbps'] < 700 < TABLE[rungs[8]]['kbps']


VMAF = {'direct@28': [99.0], 'direct@38': [72.0], 'direct@43': [53.0], 'x2@23': [94.0, 94.5, 94.7],
        'x2@28': [81.0, 81.6, 81.8], 'x4@23': [66.0, 67.5, 67.7], 'x4@28': [50.6, 52.2, 52.4],
        'x4@33': [34.6, 36.1, 36.2]}
TABLE_VMAF = {name: {**entry, 'vmaf': VMAF[name]} for name, entry in TABLE.items()}


def tcp_record(segs: int, retrans: int, rtt_ms: float = 80.0, rto_ms: float = 280.0, mss: int = 1448) -> dict:
    return {'rate': None, 'kbps': {}, 'tcp': {'rtt_us': int(rtt_ms * 1000), 'rttvar_us': 10000, 'rto_us': int(rto_ms * 1000),
                                               'mss': mss, 'data_segs_out': segs, 'total_retrans': retrans}}


def test_vmaf_metric_prefers_x2_where_psnr_prefers_direct():
    # About 1.2 Mbit/s: direct@38 has the higher PSNR, x2@28 the far higher VMAF.
    table = {**TABLE_VMAF, 'direct@38': {**TABLE_VMAF['direct@38'], 'psnr': [31.0]}}
    choices = {}
    for metric in ('psnr', 'vmaf'):
        controller = TradeoffController(table, LADDER, 1 / 30, 0.125, policy=policy_with(FAST), metric=metric)
        controller.start('x4@28', 0.0)
        controller.decide(feed_idle_link(controller, 1400))
        choices[metric] = controller.current
    assert choices == {'psnr': 'direct@38', 'vmaf': 'x2@28'}


def test_vmaf_metric_needs_a_vmaf_column():
    with pytest.raises(ValueError, match='vmaf'):
        TradeoffController(TABLE, LADDER, 1 / 30, 0.125, metric='vmaf')


def test_path_loss_and_tcp_ceiling_from_server_counters():
    controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.125, loss_aware=True)
    controller.observe_rates(tcp_record(1000, 0))
    assert controller.path_loss() is None  # one record: no window yet
    controller.observe_rates(tcp_record(1100, 1))
    assert controller.path_loss() is None  # too few segments for a rate
    controller.observe_rates(tcp_record(2000, 10))
    assert controller.path_loss() == pytest.approx(0.01)
    # Mathis: 1448 B * 8 * sqrt(3/2) / (0.08 s * sqrt(0.01)) = 1.77 Mbit/s
    assert controller.tcp_capacity() == pytest.approx(1773, rel=1e-3)


def test_loss_aware_controller_stays_lower_on_a_lossy_path():
    choices = {}
    for loss_aware in (False, True):
        controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.125, policy=policy_with(FAST), loss_aware=loss_aware,
                                        loss_weight=10.0)
        controller.start('x4@28', 0.0)
        for segs, retrans in ((0, 0), (1000, 10), (2000, 20)):  # 1% of segments retransmitted
            controller.observe_rates(tcp_record(segs, retrans))
        controller.decide(feed_idle_link(controller, 2500))
        choices[loss_aware] = controller.current
    assert choices[False] == 'x2@23'
    assert controller.rate(choices[True]) < controller.rate(choices[False])
    assert controller.stall_cost('x2@23') > controller.stall_cost('x4@28') > 0


def test_loss_aware_controller_is_unchanged_without_loss():
    choices = []
    for loss_aware in (False, True):
        controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.125, policy=policy_with(FAST), loss_aware=loss_aware)
        controller.start('x4@28', 0.0)
        for segs in (0, 1000, 2000):
            controller.observe_rates(tcp_record(segs, 0))
        controller.decide(feed_idle_link(controller, 2500))
        choices.append(controller.current)
        assert controller.stall_cost('x2@23') == 0 and controller.usable_capacity() == controller.capacity
    assert choices[0] == choices[1]


def test_tcp_path_reads_the_kernel_counters():
    from live.backlog import tcp_path
    server = socket.create_server(('127.0.0.1', 0))
    client = socket.create_connection(server.getsockname())
    peer, _ = server.accept()
    client.sendall(b'x' * 10000)
    peer.recv(65536)
    info = tcp_path(client)
    for sock in (client, peer, server):
        sock.close()
    assert info is not None and info['data_segs_out'] >= 1 and info['mss'] > 0 and info['rto_us'] > 0


def test_loss_aware_default_caps_capacity_without_pricing_stalls():
    controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.125, policy=policy_with(FAST), loss_aware=True)
    controller.start('x4@28', 0.0)
    for segs, retrans in ((0, 0), (1000, 10), (2000, 20)):
        controller.observe_rates(tcp_record(segs, retrans))
    controller.decide(feed_idle_link(controller, 2500))
    assert controller.stall_cost('x2@23') == 0
    assert controller.usable_capacity() == pytest.approx(0.5 * 1773, rel=1e-3)  # half the Mathis ceiling
    assert controller.rate(controller.current) <= 0.85 * 0.5 * 1773


def test_stale_capacity_is_withdrawn_so_the_guard_cannot_lock_on_it():
    """After a stall only keyframes get through (the guard skips the rest of each
    group); one packet a second measures nothing, so the old estimate must expire."""
    controller = TradeoffController(TABLE, LADDER, 1 / 30, 0.125, policy=policy_with(FAST))
    controller.start('x4@28', 0.0)
    end = feed_idle_link(controller, 1500)
    controller.decide(end)
    assert controller.fresh_capacity(end) == pytest.approx(1500, rel=1e-6)
    t = end
    for second in range(1, 6):  # keyframes alone, one a second, on an idle link
        t = end + second
        controller.observe_packet(t, t + 0.02, 4000, True)
        controller.decide(t + 0.02)
    assert controller.capacity is not None  # the controller keeps its last estimate
    assert controller.fresh_capacity(t + 0.02) is None  # but does not offer it to the guard
    resumed = feed_idle_link(controller, 1500, start=t + 1)  # packets flow again: measured afresh
    controller.decide(resumed)
    assert controller.fresh_capacity(resumed) == pytest.approx(1500, rel=1e-3)
