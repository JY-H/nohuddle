"""Basketball-only break detection.

Kept out of the core detector on purpose: these rules were tuned on basketball broadcasts and
made football worse (football replays keep the network logo up, so they never needed this, and the
extra evidence only added ways for an ad to look live).
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ReplaySettings:
    # Measured on real broadcasts: ads score ~0 against the game's colour signature, overlay-free
    # game footage ~0.55-0.65.
    match: float = 0.35
    cap: float = 45.0  # after this long without overlays, stop giving it the benefit of the doubt
    rate: float = 0.03  # how fast the signature follows the broadcast
    min_samples: int = 20  # frames of clearly-the-game before the signature is trusted
    back_after: int = 3  # frames of overlays in a row before a replay is over (one stray match isn't)


class ReplayByLook:
    """Replays often drop the scorebug *and* the logo, but they're still footage of the same game:
    same court, colours and lighting. Ads aren't. So keep a colour signature of the live broadcast
    and, when the overlays vanish, count footage that still matches it as the game."""

    def __init__(self, settings: ReplaySettings | None = None):
        self.s = settings or ReplaySettings()
        self._look: np.ndarray | None = None
        self._samples = 0
        self._sims: deque[float] = deque(maxlen=3)
        self._gone_since: float | None = None
        self._overlay_streak = 0

    def observe(self, frame_bgr: np.ndarray, *, any_overlay: bool, clearly_game: bool, live: bool, t: float) -> bool:
        """Evidence that this frame is live play despite no overlays. The signature only learns from
        frames that are clearly the game - otherwise an ad that briefly looked like an overlay would
        teach it the ad's colours."""
        hsv = cv2.cvtColor(cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        self._overlay_streak = self._overlay_streak + 1 if any_overlay else 0
        if any_overlay:
            if self._overlay_streak >= self.s.back_after:
                self._gone_since = None
                self._sims.clear()
            if live and clearly_game:
                self._look = hist if self._look is None else (1 - self.s.rate) * self._look + self.s.rate * hist
                self._samples += 1
            return False
        self._gone_since = self._gone_since or t
        if self._samples < self.s.min_samples or t - self._gone_since > self.s.cap:
            return False
        self._sims.append(float(cv2.compareHist(self._look, hist, cv2.HISTCMP_CORREL)))
        return float(np.mean(self._sims)) >= self.s.match
