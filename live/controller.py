"""Trade-off controller: choose the rung (rendition and CRF) to stream, live.

The four quantities and where each comes from:

  bandwidth   the rate each rung costs on this content (R records from the server)
              against the link's capacity, measured from packet send and arrival
              times: when a standing queue builds, arrivals are paced by the link,
              so their rate is the capacity.
  latency     queuing delay must stay inside the playout delay; a keyframe burst
              of a rung must cross the link within that budget, and a growing
              queue forces a step down to drain it.
  resources   an x2 or x4 rung is usable only if some exit of the model fits the
              GPU share of the frame interval (measured times from DepthPolicy);
              the deepest exit that fits sets the quality it is credited with.
  quality     predicted PSNR of each rung at that exit, from the rate-quality table
              (scripts/measure_rate_quality.py).

The rule: among usable rungs within headroom x capacity (stay_headroom for the
current rung, so an estimate near the edge does not flip it), take the best predicted
quality; among rungs within tie_db of it, the cheapest. Step down at once when the
current rung no longer fits; step up only for at least min_gain_db, with a short
queue, and up_hold_s after the last step down. When the queue
outgrows the latency budget, drop to half the capacity and skip the rest of the
group of pictures, so the queue drains before the switch. Without any capacity
measurement, try the next better rung after a quiet probe interval.

Direct rungs need no GPU, so under GPU load the controller moves to direct when the
link allows it; under a tight link it moves to x4, where the model recovers the most.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import NamedTuple

from live.protocol import SCALE, parse_rung, rung_name


class Rung(NamedTuple):
    rendition: str
    crf: int

    @property
    def name(self) -> str:
        return rung_name(self.rendition, self.crf)

    @property
    def scale(self) -> int:
        return SCALE[self.rendition]


class DeliveryMonitor:
    """Link capacity, queuing delay and delivery rate from packet send and arrival times.

    A packet that enters an idle link (the previous one had already arrived when it
    was sent) takes fixed + size / capacity to arrive, where fixed is propagation
    plus any clock offset between the machines. A least-squares line through recent
    idle packets of different sizes (keyframes are several times larger than other
    frames) gives the capacity from its slope and the fixed delay from its intercept,
    so every keyframe is a free probe. While the link is the bottleneck no packet
    finds it idle, but arrivals are then paced by it, so the delivery rate is the
    capacity.

    Queuing delay of a packet is its delay less the fixed delay and its own
    transmission time. The standing queue is the smallest over the last standing_s:
    a keyframe burst raises the delay briefly, a link slower than the stream raises
    it for good.
    """

    def __init__(self, window_s: float = 1.0, standing_s: float = 0.5, idle_window_s: float = 2.0,
                 min_spread: float = 1.5) -> None:
        self.window_s, self.standing_s, self.idle_window_s, self.min_spread = window_s, standing_s, idle_window_s, min_spread
        self.arrivals: deque[tuple[float, int]] = deque()
        self.queueing: deque[tuple[float, float]] = deque()
        self.idle: deque[tuple[float, int, float]] = deque()  # (arrived, bytes, delay)
        self.busy: deque[tuple[float, int, float]] = deque()  # (arrived, bytes, gap after the previous)
        self.min_delay = float('inf')
        self.previous_arrival = float('-inf')
        self.fit: tuple[float, float] | None = None  # (fixed delay s, capacity kbit/s)
        self.fit_at = float('-inf')
        self.latest = float('-inf')
        self.last_key: tuple[float, int] | None = None  # (sent, bytes) of the latest keyframe
        self.excess = 0.0  # queuing delay of the latest packet beyond its keyframe's burst

    def add(self, sent: float, arrived: float, size: int, keyframe: bool = False) -> None:
        delay = arrived - sent
        self.latest = arrived
        self.min_delay = min(self.min_delay, delay)
        if self.previous_arrival <= sent:
            self.idle.append((arrived, size, delay))
            while self.idle and self.idle[0][0] < arrived - self.idle_window_s:
                self.idle.popleft()
            self._refit()
        elif arrived > self.previous_arrival:
            # Queued behind the previous packet: it arrives one transmission time later.
            self.busy.append((arrived, size, arrived - self.previous_arrival))
        while self.busy and self.busy[0][0] < arrived - self.window_s:
            self.busy.popleft()
        self.previous_arrival = arrived
        fixed, capacity = self.fit if self.fit else (self.min_delay, None)
        own = size * 8 / (capacity * 1000) if capacity else 0.0
        queued = max(delay - fixed - own, 0.0)
        if keyframe:
            self.last_key = (sent, size)
        # A packet sent while its keyframe is still crossing the link waits for the rest
        # of it; only the delay beyond that means the link is slower than estimated.
        burst = 0.0
        if capacity and self.last_key and not keyframe:
            burst = max(self.last_key[1] * 8 / (capacity * 1000) - (sent - self.last_key[0]), 0.0)
        self.excess = max(queued - burst, 0.0)
        self.arrivals.append((arrived, size))
        self.queueing.append((arrived, queued))
        while self.arrivals and self.arrivals[0][0] < arrived - self.window_s:
            self.arrivals.popleft()
        while self.queueing and self.queueing[0][0] < arrived - self.standing_s:
            self.queueing.popleft()

    def _refit(self) -> None:
        sizes = [s for _, s, _ in self.idle]
        if len(sizes) < 3 or max(sizes) < self.min_spread * min(sizes):
            return
        bits = [s * 8 for s in sizes]
        delays = [d for _, _, d in self.idle]
        mean_b, mean_d = sum(bits) / len(bits), sum(delays) / len(delays)
        covariance = sum((b - mean_b) * (d - mean_d) for b, d in zip(bits, delays))
        variance = sum((b - mean_b) ** 2 for b in bits)
        if variance <= 0 or covariance <= 0:
            return  # no slope: the link is much faster than the packets need
        seconds_per_bit = covariance / variance
        self.fit = (mean_d - seconds_per_bit * mean_b, 1 / seconds_per_bit / 1000)
        self.fit_at = self.idle[-1][0]

    def forget_idle(self) -> None:
        """The link changed under load: earlier idle measurements no longer hold."""
        self.idle.clear()
        self.fit = None if self.fit is None else (self.fit[0], None)

    @property
    def idle_capacity(self) -> float | None:
        """None once no fit has been possible for a while: an old fit may be stale."""
        if not self.fit or self.latest - self.fit_at > 1.5 * self.idle_window_s:
            return None
        return self.fit[1]

    @property
    def busy_capacity(self) -> float | None:
        """kbit/s from packets that queued behind the previous one (packet dispersion).

        Unlike the delivery rate, this holds when the stream is intermittent, for
        example while the server is dropping frames to drain the queue."""
        if len(self.busy) < 3:
            return None
        return sum(s for _, s, _ in self.busy) * 8 / sum(g for _, _, g in self.busy) / 1000

    @property
    def queue_delay(self) -> float:
        return self.queueing[-1][1] if self.queueing else 0.0

    @property
    def standing_queue(self) -> float:
        return min(q for _, q in self.queueing) if self.queueing else 0.0

    @property
    def delivery_kbps(self) -> float | None:
        """Payload rate over the window; None until the window is half full."""
        if len(self.arrivals) < 3:
            return None
        span = self.arrivals[-1][0] - self.arrivals[0][0]
        if span < self.window_s / 2:
            return None
        return sum(size for _, size in list(self.arrivals)[1:]) * 8 / span / 1000


def load_table(path: Path) -> dict[str, dict]:
    """Mean kbps and PSNR per rung from scripts/measure_rate_quality.py."""
    return json.loads(Path(path).read_text())['rungs']


class TradeoffController:
    def __init__(self, table: dict[str, dict], ladder: list[str], frame_interval: float, latency_budget_s: float,
                 policy=None, gpu_share: float = 0.6, headroom: float = 0.85, stay_headroom: float = 0.95,
                 congestion_s: float = 0.04,
                 probe_s: float = 4.0, min_gain_db: float = 0.1, keyint: int = 30, rate_memory: int = 3,
                 up_hold_s: float = 2.0,
                 tie_db: float = 0.05, deadline_s: float | None = None, gpu_aware: bool = True) -> None:
        missing = [name for name in ladder if name not in table]
        if missing:
            raise ValueError(f'rate-quality table has no entry for {missing}')
        self.table, self.policy = table, policy
        self.rungs = {name: Rung(*parse_rung(name)) for name in ladder}
        self.frame_interval, self.latency_budget_s = frame_interval, latency_budget_s
        # The display deadline (playout delay plus drop-late less processing): a packet
        # may wait up to it and still be shown, as the P-frames behind a keyframe do.
        self.deadline_s = latency_budget_s if deadline_s is None else max(deadline_s, latency_budget_s)
        self.gpu_share, self.headroom, self.stay_headroom = gpu_share, headroom, stay_headroom
        self.gpu_aware = gpu_aware  # ablation: False credits every rung with the deepest exit
        self.congestion_s = congestion_s
        self.probe_s = probe_s
        self.min_gain_db, self.tie_db = min_gain_db, tie_db
        self.monitor = DeliveryMonitor()
        self.rates: dict[str, float] = {}
        self.rate_history: dict[str, deque] = {}
        self.rate_memory = rate_memory
        self.up_hold_s = up_hold_s
        self.down_at = float('-inf')
        self.key_bytes: dict[str, int] = {}
        self.capacity: float | None = None  # kbit/s, measured while the link was the bottleneck
        self.congested = False
        self.current: str | None = None
        self.changed_at = 0.0
        self.trace: list[dict] = []
        self.keyint = keyint
        self.skipped_at = float('-inf')

    # Inputs -------------------------------------------------------------

    def observe_packet(self, sent: float, arrived: float, size: int, keyframe: bool) -> None:
        self.monitor.add(sent, arrived, size, keyframe)

    def observe_rates(self, message: dict) -> None:
        """Each rung is priced at its peak over the last few seconds: content changes
        the rate faster than a switch can follow, so the average would flip rungs."""
        for name, kbps in message.get('kbps', {}).items():
            history = self.rate_history.setdefault(name, deque(maxlen=self.rate_memory))
            history.append(kbps)
            self.rates[name] = max(history)
        self.key_bytes.update(message.get('key_bytes', {}))

    # Model of each rung -------------------------------------------------

    def rate(self, name: str) -> float:
        return self.rates.get(name, self.table[name]['kbps'])

    def exit_for(self, scale: int) -> int | None:
        """Deepest exit whose measured time fits the GPU share; None if none does."""
        if self.policy is None:
            return None
        if not self.gpu_aware:
            return self.policy.num_exits - 1
        budget = self.gpu_share * self.frame_interval
        fitting = [e for e in range(self.policy.num_exits) if self.policy.expected.get((scale, e), float('inf')) <= budget]
        return max(fitting) if fitting else None

    def quality(self, name: str) -> float | None:
        """Predicted PSNR, or None when the client cannot run this rung in time."""
        rung, entry = self.rungs[name], self.table[name]
        if rung.scale == 1:
            return entry['psnr'][0]
        exit_index = self.exit_for(rung.scale)
        return None if exit_index is None else entry['psnr'][exit_index]

    def burst_fits(self, name: str) -> bool:
        """A keyframe of this rung crosses the link before the display deadline, so it
        and the frames queued behind it are still shown."""
        if self.capacity is None or name not in self.key_bytes:
            return True
        return self.key_bytes[name] * 8 / (self.capacity * 1000) <= self.deadline_s

    # Decision -----------------------------------------------------------

    def _update_capacity(self) -> None:
        measured = self.monitor.busy_capacity or self.monitor.delivery_kbps
        if self.monitor.standing_queue > self.congestion_s and measured is not None:
            if not self.congested:
                self.monitor.forget_idle()
            self.congested = True
            self.capacity = measured
        elif self.monitor.standing_queue < self.congestion_s / 2:
            self.congested = False
            if self.monitor.idle_capacity is not None:
                self.capacity = self.monitor.idle_capacity

    def decide(self, now: float) -> tuple[str, str, bool] | None:
        """(rung, reason, skip) when the target should change, else None.

        skip asks the server to drop the rest of the current group of pictures, so
        the queue drains before the switch; the screen holds the last frame meanwhile.
        """
        self._update_capacity()
        runnable = {name: q for name in self.rungs if (q := self.quality(name)) is not None}
        if not runnable:  # the model fits nowhere: direct rungs need no GPU
            runnable = {n: self.table[n]['psnr'][0] for n, r in self.rungs.items() if r.scale == 1}
        if not runnable:  # and the ladder has no direct rung: run the model anyway, late
            runnable = {n: self.table[n]['psnr'][0] for n in self.rungs}
        options = {n: q for n, q in runnable.items() if self.burst_fits(n)}
        if not options:  # no keyframe fits the budget: the cheapest runnable rung
            cheapest = min(runnable, key=self.rate)
            options = {cheapest: runnable[cheapest]}

        # Drain for a queue that persists past the budget, or for one that the last
        # keyframe does not explain. The short queue behind a keyframe clears by itself,
        # and skipping to the next keyframe would only repeat it; rungs whose keyframes
        # are too large for the link are excluded by burst_fits instead.
        draining = (self.monitor.standing_queue > self.latency_budget_s
                    or self.monitor.excess > self.latency_budget_s)
        if draining:
            measured = self.capacity or self.monitor.delivery_kbps or self.rate(self.current)
            allowed, reason = 0.5 * measured, 'drain'
        elif self.capacity is not None:
            allowed, reason = self.headroom * self.capacity, 'capacity'
        else:
            allowed, reason = self.rate(self.current), 'hold'
            if now - self.changed_at >= self.probe_s and not self.congested:
                current_q = options.get(self.current, float('-inf'))
                better = [n for n, q in options.items() if q >= current_q + self.min_gain_db and self.rate(n) > allowed]
                if better:
                    allowed, reason = self.rate(min(better, key=self.rate)), 'probe'

        fitting = {n: q for n, q in options.items() if self.rate(n) <= allowed}
        if reason == 'capacity' and self.current in options and self.rate(self.current) <= self.stay_headroom * self.capacity:
            fitting[self.current] = options[self.current]  # hysteresis: keep a rung that still fits
        if not fitting:
            cheapest = min(options, key=self.rate)
            fitting, reason = {cheapest: options[cheapest]}, 'floor'
        best = max(fitting.values())
        choice = min((n for n, q in fitting.items() if q >= best - self.tie_db), key=self.rate)

        if choice == self.current:
            if draining and now - self.skipped_at >= self.frame_interval * self.keyint:
                self.skipped_at = now  # already on the right rung: only drop the rest of this group
                self.trace.append({'t': round(now, 3), 'to': choice, 'from': choice, 'reason': 'drain', 'skip': True,
                                   'queue_ms': round(self.monitor.queue_delay * 1000, 1)})
                return choice, 'drain', True
            return None
        if self.current in fitting and abs(fitting[choice] - fitting[self.current]) < self.min_gain_db:
            return None  # the current rung still fits and the change gains too little
        upgrade = self.rate(choice) > self.rate(self.current)
        if upgrade and (self.monitor.queue_delay > self.congestion_s or now - self.down_at < self.up_hold_s):
            return None  # let the queue drain and the last step down settle first
        skip = draining and self.rate(choice) < self.rate(self.current)
        if not upgrade:
            self.down_at = now
        if skip:
            self.skipped_at = now
        self.trace.append({'t': round(now, 3), 'to': choice, 'from': self.current, 'reason': reason, 'skip': skip,
                           'allowed_kbps': round(allowed, 1), 'rate_kbps': round(self.rate(choice), 1),
                           'predicted_psnr': round(fitting[choice], 3),
                           'capacity_kbps': None if self.capacity is None else round(self.capacity, 1),
                           'queue_ms': round(self.monitor.queue_delay * 1000, 1),
                           'standing_ms': round(self.monitor.standing_queue * 1000, 1)})
        self.current, self.changed_at = choice, now
        return choice, reason, skip

    def start(self, name: str, now: float) -> None:
        """The rung the server starts on."""
        self.current, self.changed_at = name, now
