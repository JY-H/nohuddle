"""Playback health: is a stream stuck, or merely laggy?"""

from collections import deque
from dataclasses import dataclass


@dataclass
class HealthSettings:
    window: float = 60.0  # seconds of history to judge lag over
    min_window: float = 30.0  # need at least this much before calling a stream laggy
    min_play_ratio: float = 0.85  # seconds of video played per wall-clock second
    max_rebuffers: int = 3  # player "waiting" events within the window
    stall_after: float = 15.0  # no progress at all for this long -> stalled


class PlaybackHealth:
    def __init__(self, settings: HealthSettings | None = None):
        self.s = settings or HealthSettings()
        self.samples: deque[tuple[float, float, int]] = deque()  # (wall time, currentTime, rebuffer count)
        self.progressed_at: float | None = None

    def reset(self, now: float) -> None:
        self.samples.clear()
        self.progressed_at = now

    def update(self, now: float, current_time: float | None, rebuffers: int) -> None:
        if self.progressed_at is None:
            self.progressed_at = now
        if current_time is None:
            return
        if not self.samples or current_time != self.samples[-1][1]:
            self.progressed_at = now
        self.samples.append((now, current_time, rebuffers))
        while self.samples and now - self.samples[0][0] > self.s.window:
            self.samples.popleft()

    def stalled(self, now: float) -> bool:
        return self.progressed_at is not None and now - self.progressed_at > self.s.stall_after

    def play_ratio(self) -> float | None:
        if len(self.samples) < 2 or self.samples[-1][0] - self.samples[0][0] < self.s.min_window:
            return None
        played = 0.0
        for (t0, c0, _), (t1, c1, _) in zip(self.samples, list(self.samples)[1:]):
            # Live players occasionally jump to the live edge or restart; don't let a seek count
            # as smooth playback (or as negative playback).
            played += min(max(c1 - c0, 0.0), (t1 - t0) * 1.5)
        return played / (self.samples[-1][0] - self.samples[0][0])

    def rebuffers(self) -> int:
        if len(self.samples) < 2:
            return 0
        return max(0, self.samples[-1][2] - self.samples[0][2])

    def laggy(self) -> bool:
        ratio = self.play_ratio()
        if ratio is None:
            return False
        return ratio < self.s.min_play_ratio or self.rebuffers() >= self.s.max_rebuffers
