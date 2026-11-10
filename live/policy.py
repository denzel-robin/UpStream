"""Per-frame choice of the anytime exit.

The deepest exit whose expected run time fits the time left before the frame is
due on screen. Expected times are exponential moving averages per scale and exit,
seeded at start-up, so the choice follows the real machine and load.
"""
from __future__ import annotations


class DepthPolicy:
    def __init__(self, num_exits: int, frame_interval: float, margin: float = 0.004, smoothing: float = 0.2):
        self.num_exits = num_exits
        self.frame_interval = frame_interval
        self.margin = margin
        self.smoothing = smoothing
        self.expected: dict[tuple[int, int], float] = {}

    def update(self, scale: int, exit_index: int, seconds: float) -> None:
        key = (scale, exit_index)
        previous = self.expected.get(key)
        self.expected[key] = seconds if previous is None else previous + self.smoothing * (seconds - previous)

    def budget(self, time_left: float) -> float:
        """At most one frame interval, so the reconstruction keeps up with the stream."""
        return min(time_left, self.frame_interval) - self.margin

    def choose(self, scale: int, time_left: float) -> int:
        budget = self.budget(time_left)
        for exit_index in reversed(range(self.num_exits)):
            expected = self.expected.get((scale, exit_index))
            if expected is not None and expected <= budget:
                return exit_index
        return 0  # nothing fits: the cheapest exit keeps the state going
