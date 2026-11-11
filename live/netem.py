"""A real network path on one machine: a network namespace, a veth pair and tc netem.

The client runs inside the namespace; the server stays outside. Their traffic crosses
the kernel's real TCP stack (congestion control, retransmission, send buffers) over a
veth pair. netem on each end adds what a real path has:

  server -> client   rate limit (following a schedule), delay, jitter, loss, and a
                     bounded router queue (limit, in packets)
  client -> server   delay, jitter and loss (acknowledgements, switch requests)

Both ends share one clock, so latency is still measured exactly, which two machines
could not do without clock synchronisation.

Offloads (TSO, GSO, GRO) are switched off on both ends, so netem shapes real
1500-byte packets instead of 64 KB super-packets.

Needs root. Everything is removed by teardown(), and deleting the namespace also
removes the veth pair.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

NAMESPACE = 'upstream-client'
HOST_DEV, CLIENT_DEV = 'veth-up-srv', 'veth-up-cli'
HOST_IP, CLIENT_IP = '10.77.0.1', '10.77.0.2'


@dataclass(frozen=True)
class PathConditions:
    """One-way delay and jitter in ms, loss in percent, router queue in packets."""
    delay_ms: float = 20.0
    jitter_ms: float = 5.0
    loss_pct: float = 0.1
    limit_packets: int = 300

    @classmethod
    def parse(cls, text: str) -> 'PathConditions':
        """'delay=20,jitter=5,loss=0.1,limit=300' (units: ms, ms, %, packets)."""
        names = {'delay': 'delay_ms', 'jitter': 'jitter_ms', 'loss': 'loss_pct', 'limit': 'limit_packets'}
        values = {}
        for part in filter(None, text.split(',')):
            key, value = part.split('=')
            if key not in names:
                raise ValueError(f'unknown path setting {key!r}; use {sorted(names)}')
            values[names[key]] = int(value) if key == 'limit' else float(value)
        return cls(**values)

    def netem(self, rate_kbps: float | None = None) -> list[str]:
        args = ['netem', 'delay', f'{self.delay_ms}ms']
        if self.jitter_ms:
            args += [f'{self.jitter_ms}ms', 'distribution', 'normal']
        if self.loss_pct:
            args += ['loss', f'{self.loss_pct}%']
        if rate_kbps is not None:
            args += ['rate', f'{int(rate_kbps)}kbit']
        return args + ['limit', str(self.limit_packets)]


def run(*command: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), check=check, capture_output=True, text=True)


class NetemPath:
    def __init__(self, conditions: PathConditions) -> None:
        self.conditions = conditions
        self.applied: list[tuple[float, float]] = []  # (seconds after start, kbit/s) as applied

    # Set-up and tear-down -----------------------------------------------

    def setup(self, rate_kbps: float) -> None:
        self.teardown()
        run('ip', 'netns', 'add', NAMESPACE)
        run('ip', 'link', 'add', HOST_DEV, 'type', 'veth', 'peer', 'name', CLIENT_DEV)
        run('ip', 'link', 'set', CLIENT_DEV, 'netns', NAMESPACE)
        run('ip', 'addr', 'add', f'{HOST_IP}/24', 'dev', HOST_DEV)
        run('ip', 'link', 'set', HOST_DEV, 'up')
        self.in_namespace('ip', 'addr', 'add', f'{CLIENT_IP}/24', 'dev', CLIENT_DEV)
        self.in_namespace('ip', 'link', 'set', CLIENT_DEV, 'up')
        self.in_namespace('ip', 'link', 'set', 'lo', 'up')
        for prefix, dev in (([], HOST_DEV), (['ip', 'netns', 'exec', NAMESPACE], CLIENT_DEV)):
            run(*prefix, 'ethtool', '-K', dev, 'tso', 'off', 'gso', 'off', 'gro', 'off', check=False)
        run('tc', 'qdisc', 'add', 'dev', HOST_DEV, 'root', *self.conditions.netem(rate_kbps))
        self.in_namespace('tc', 'qdisc', 'add', 'dev', CLIENT_DEV, 'root', *self.conditions.netem(None))

    def teardown(self) -> None:
        run('ip', 'netns', 'del', NAMESPACE, check=False)  # removes the veth pair too
        run('ip', 'link', 'del', HOST_DEV, check=False)

    def in_namespace(self, *command: str) -> subprocess.CompletedProcess:
        return run('ip', 'netns', 'exec', NAMESPACE, *command)

    # Rate schedule -------------------------------------------------------

    def set_rate(self, kbps: float) -> None:
        run('tc', 'qdisc', 'change', 'dev', HOST_DEV, 'root', *self.conditions.netem(kbps))

    def follow(self, schedule: list[tuple[float, float]], started: threading.Event, stop: threading.Event) -> threading.Thread:
        """Apply each (seconds, kbit/s) step, timed from when started is set."""
        def body() -> None:
            if not started.wait(timeout=300):
                return
            begin = time.monotonic()
            for at, kbps in schedule:
                if at > 0 and stop.wait(max(begin + at - time.monotonic(), 0)):
                    return
                self.set_rate(kbps)
                self.applied.append((round(time.monotonic() - begin, 3), kbps))
        thread = threading.Thread(target=body, name='netem-schedule', daemon=True)
        thread.start()
        return thread

    def statistics(self) -> str:
        """tc's counters on the server side (sent, dropped, overlimits, backlog)."""
        return run('tc', '-s', 'qdisc', 'show', 'dev', HOST_DEV, check=False).stdout
