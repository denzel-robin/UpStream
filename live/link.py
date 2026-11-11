"""An emulated bottleneck between server and client, for developing the controller.

Records queue in order and leave at the link rate, as at a router whose buffer is
never full: a stream above the rate builds up queuing delay instead of losing data.
The rate follows a schedule of (seconds from start, kbit/s). Propagation delay and
loss are left to real network shaping (tc netem) in the evaluation.
"""
from __future__ import annotations

import queue
import socket
import threading
import time


def parse_link(text: str) -> list[tuple[float, float]]:
    """'0:3000,10:600' -> [(0.0, 3000.0), (10.0, 600.0)]: kbit/s from each time on."""
    steps = sorted((float(at), float(kbps)) for at, kbps in (part.split(':') for part in text.split(',')))
    if not steps or steps[0][0] != 0 or any(kbps <= 0 for _, kbps in steps):
        raise ValueError('link schedule must start at 0 and have positive rates')
    return steps


class EmulatedLink:
    def __init__(self, sock: socket.socket, schedule: list[tuple[float, float]]) -> None:
        self.sock, self.schedule = sock, schedule
        self.records: queue.Queue = queue.Queue()
        self.start = time.monotonic()
        self.max_delay_s = 0.0
        self.queued_bytes = 0  # handed to the link, not yet delivered
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name='link', daemon=True)
        self.thread.start()

    def rate_kbps(self, at: float) -> float:
        rate = self.schedule[0][1]
        for begin, kbps in self.schedule:
            if at - self.start >= begin:
                rate = kbps
        return rate

    def start_clock(self) -> None:
        """Time the rate schedule from now (the server calls it when frame 0 leaves)."""
        self.start = time.monotonic()

    def send(self, record: bytes) -> None:
        if self.error is not None:
            raise ConnectionError('link failed') from self.error
        self.queued_bytes += len(record)
        self.records.put((time.monotonic(), record))

    def close(self, timeout: float = 60) -> None:
        """Wait until everything queued has been delivered."""
        self.records.put(None)
        self.thread.join(timeout)

    def _run(self) -> None:
        free_at = 0.0
        try:
            while (item := self.records.get()) is not None:
                queued, record = item
                begin = max(time.monotonic(), free_at)
                done = begin + len(record) * 8 / (self.rate_kbps(begin) * 1000)
                if (wait := done - time.monotonic()) > 0:
                    time.sleep(wait)
                self.sock.sendall(record)
                self.queued_bytes -= len(record)
                free_at = done
                self.max_delay_s = max(self.max_delay_s, time.monotonic() - queued)
        except BaseException as error:  # reported by the next send
            self.error = error
