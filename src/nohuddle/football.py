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
    # ...unless it's unmistakably a football field: yard lines (thin white lines with field green
    # on both sides) in every frame. Rare in ads (a green pool, a golf graphic) and only briefly,
    # but they keep a feed whose graphics dropped out for minutes on screen.
    yard_lines: int = 3
    back_after: int = 3  # frames of overlays in a row before the allowance resets
    # A sportsbook ad on a green "field" backdrop has the share, and even straight white lines (its
    # panel borders), but its green is all around the edges and doesn't move. A field fills the
    # middle of the picture (close-ups ~10%+), and players or the camera move over it (a still
    # pre-snap shot is green in the middle).
    graphic_center: float = 0.06  # at most this share of the middle green...
    graphic_still: float = 1.5  # ...and the green changing less than this (mean grey level) a frame


def _green(small_bgr: np.ndarray, min_saturation: int = 50) -> np.ndarray:
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 30) & (h <= 90) & (s >= min_saturation) & (v >= 40)


def field_share(frame_bgr: np.ndarray, min_saturation: int = 50) -> float:
    """Share of the frame that's field green. (Some turf, e.g. MetLife's, is greyer: ~45.)"""
    return float(_green(cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA), min_saturation).mean())


def green_layout(frame_bgr: np.ndarray, prev_bgr: np.ndarray | None) -> tuple[float, float, float | None]:
    """(green share of the frame, of its middle quarter, how much the green changed since `prev`)."""
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    green = _green(small)
    moved = None
    if prev_bgr is not None and green.any():
        prev = cv2.resize(prev_bgr, (160, 90), interpolation=cv2.INTER_AREA)
        diff = np.abs(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                      - cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32))
        moved = float(diff[green].mean())
    return float(green.mean()), float(green[22:68, 40:120].mean()), moved


def yard_lines(frame_bgr: np.ndarray) -> int:
    """Long, thin white lines painted on field green."""
    hsv = cv2.cvtColor(cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = ((h >= 30) & (h <= 90) & (s >= 50) & (v >= 40)).astype(np.uint8)
    white = ((s < 70) & (v > 160) & (cv2.dilate(green, np.ones((7, 7), np.uint8)) > 0)).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(white, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=6)
    count = 0
    for x1, y1, x2, y2 in ([] if lines is None else lines.reshape(-1, 4)):
        d = np.array([x2 - x1, y2 - y1], float)
        normal = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
        sides = [np.array([x1, y1]) + a * d + side * 5 * normal
                 for a in np.linspace(0.1, 0.9, 9) for side in (1, -1)]
        inside = [(int(x), int(y)) for x, y in sides if 0 <= x < 320 and 0 <= y < 180]
        if inside and np.mean([green[y, x] for x, y in inside]) >= 0.8:
            count += 1
    return count


class FieldOnScreen:
    """Replays and highlight packages often drop the scorebug and the logo, but they're still
    shots of a football field. Unlike a learned colour signature, "lots of field green" needs no
    learning, so an ad can't teach it anything."""

    def __init__(self, settings: FieldSettings | None = None):
        self.s = settings or FieldSettings()
        self._shares: deque[float] = deque(maxlen=self.s.smooth)
        self._lines: deque[int] = deque(maxlen=self.s.smooth)
        self._centers: deque[float] = deque(maxlen=self.s.smooth)
        self._moves: deque[float] = deque(maxlen=self.s.smooth)
        self._prev: np.ndarray | None = None
        self._game_seen_at: float | None = None
        self._overlay_streak = 0
        self.field_since: float | None = None  # the picture's been the field since (overlays or not)

    def _graphic(self) -> bool:
        """A green frame around something else, standing still. (A cut into it is one big change.)"""
        return (float(np.mean(self._centers)) < self.s.graphic_center and len(self._moves) == self._moves.maxlen
                and float(np.median(self._moves)) < self.s.graphic_still)

    def observe(self, frame_bgr: np.ndarray, *, any_overlay: bool, clearly_game: bool, live: bool, t: float) -> bool:
        self._overlay_streak = self._overlay_streak + 1 if any_overlay else 0
        if clearly_game or self._overlay_streak >= self.s.back_after:
            self._game_seen_at = t
        prev, self._prev = self._prev, frame_bgr
        # Keep measuring through overlay frames: a small overlay that comes and goes (a corner
        # ticker cycling other games' scores, with the scorebug off) must not restart the average.
        share, center, moved = green_layout(frame_bgr, prev)
        self._shares.append(share)
        self._centers.append(center)
        if moved is not None:
            self._moves.append(moved)
        on_field = len(self._shares) == self._shares.maxlen and float(np.mean(self._shares)) >= self.s.share \
            and not self._graphic()
        self.field_since = (self.field_since or t) if on_field else None
        if any_overlay:
            self._lines.clear()
            return False  # the overlay already says it's the game
        if self._game_seen_at is None:
            return False
        if len(self._shares) < self._shares.maxlen or float(np.mean(self._shares)) < self.s.share:
            self._lines.clear()
            return False
        if self._graphic():
            return False
        if t - self._game_seen_at <= self.s.cap:
            return True
        self._lines.append(yard_lines(frame_bgr))  # only needed past the cap
        return len(self._lines) == self._lines.maxlen and min(self._lines) >= self.s.yard_lines
