"""Football-only break detection.

Kept out of the core detector (see sports.py): tuned on football broadcasts only.
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FieldSettings:
    # Measured on real broadcasts: >=35% field green in 3-7% of ad frames, but in ~65% of
    # overlay-free game footage (replays, highlights shot from the field).
    share: float = 0.35
    smooth: int = 3  # frames averaged, so one grassy frame in an ad doesn't count
    cap: float = 45.0  # only this long after the game was last clearly on screen
    back_after: int = 3  # frames of overlays in a row before the allowance resets


def field_share(frame_bgr: np.ndarray) -> float:
    """Share of the frame that's field green."""
    hsv = cv2.cvtColor(cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((h >= 30) & (h <= 90) & (s >= 50) & (v >= 40)).mean())


class FieldOnScreen:
    """Replays and highlight packages often drop the scorebug and the logo, but they're still
    shots of a football field. Unlike a learned colour signature, "lots of field green" needs no
    learning, so an ad can't teach it anything."""

    def __init__(self, settings: FieldSettings | None = None):
        self.s = settings or FieldSettings()
        self._shares: deque[float] = deque(maxlen=self.s.smooth)
        self._game_seen_at: float | None = None
        self._overlay_streak = 0

    def observe(self, frame_bgr: np.ndarray, *, any_overlay: bool, clearly_game: bool, live: bool, t: float) -> bool:
        self._overlay_streak = self._overlay_streak + 1 if any_overlay else 0
        if clearly_game or self._overlay_streak >= self.s.back_after:
            self._game_seen_at = t
        if any_overlay:
            self._shares.clear()
            return False
        if self._game_seen_at is None or t - self._game_seen_at > self.s.cap:
            return False
        self._shares.append(field_share(frame_bgr))
        return len(self._shares) == self._shares.maxlen and float(np.mean(self._shares)) >= self.s.share
