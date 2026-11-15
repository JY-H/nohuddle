"""Commercial-break detection by learning each broadcast's persistent overlays.

A live broadcast nearly always has *something* burned in on top of the picture: the scorebug,
the network logo, a down-and-distance box. Their positions and designs differ by network, and
any one of them may disappear for a while (a side-panel shot, a full-screen stat graphic, a
replay) - but commercials have none of them.

So instead of matching a hard-coded scorebug template, each game gets its own detector that:
  1. keeps a rolling history of downscaled frames (one every few seconds, ~10 minutes),
  2. finds pixels that look the same in most of that history *and* sit on edges/text -
     i.e. graphics composited over a constantly-moving picture,
  3. groups them into regions (scorebug, logo, ...), and
  4. calls the current frame "broadcast" if ANY region still matches.

Raw per-frame answers are then smoothed over time: a break is only declared once every overlay
has been gone for `break_after` seconds, so short full-screen graphics don't trigger a switch,
while the return to live (`live_after`) is fast so you don't miss the snap.
"""

from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from . import gameclock
from .gameclock import ClockSettings, GameClock

W, H = 384, 216


class State(StrEnum):
    LEARNING = "learning"  # not enough history yet; treated as live by the scheduler
    LIVE = "live"
    BREAK = "break"
    STALLED = "stalled"  # stream isn't playing


@dataclass
class DetectorSettings:
    sample_every: float = 4.0  # seconds between frames added to the history
    history_len: int = 150  # ~10 minutes at 4s
    min_history: int = 12  # frames needed before the first model is built
    min_history_span: float = 60.0  # ...spanning at least this many seconds
    refit_every: float = 20.0  # seconds between rebuilding the overlay model
    # An overlay's outline is an edge in the same place in nearly every broadcast frame, even when
    # it's translucent; scenery only manages that while the camera holds still.
    edge_persistence: float = 0.7  # share of history frames a pixel must be an edge in
    region_match: float = 0.35  # correlation with the learned edge pattern needed to count as present
    min_region_px: int = 20  # persistent edge pixels needed for a region
    max_region_area: float = 0.15  # bounding box larger than this share of the frame = scenery
    # Seconds with no overlay (neither scorebug nor logo) before declaring a break. Measured on real
    # broadcasts: 4s gives the same false switches as 8s (replays/transitions with no graphics),
    # but leaves the screen 4s sooner; 3s starts switching away during replays.
    break_after: float = 4.0
    live_after: float = 2.0  # seconds with an overlay back before declaring live
    # The network logo stays up through halftime shows; the scorebug doesn't. Replays and stat
    # graphics hide the scorebug for well under this long.
    intermission_after: float = 120.0
    intermission_logo_for: float = 20.0  # ...with the logo continuously up for this long
    # Only relearn while the scorebug is (recently) on screen, so a long halftime or a stretch of
    # ads can't overwrite what the game looks like.
    refit_needs_primary_within: float = 30.0
    # Minutes of video with nothing persistent at all (a halftime show, a long run of ads) isn't
    # game coverage, even if we never got to see the game's overlays.
    no_overlays_break_after: float = 180.0
    recent_fit_frames: int = 30  # when nothing's been learned yet, fit on just the last ~2 minutes
    read_clock_every: float = 2.0  # seconds between scorebug clock reads (OCR)
    read_clock_every_idle: float = 6.0  # ...when no overlay is showing
    clock_disagree_relearn: float = 20.0  # clock says live but no overlays for this long -> relearn
    stuck_break_relearn: float = 480.0  # "break" (not halftime) this long -> relearn
    long_break_halftime: float = 300.0  # a break this long right after the 2nd quarter = halftime
    network_hits: int = 3  # frames a known network's overlays must show up in before adopting it...
    network_hits_span: float = 4.0  # ...spread over at least this long
    prior_history_span: float = 180.0  # with a network's model to go on, learn our own more patiently
    # While learning, no clock reading for this long (on a readable scorebug) means it's off: hidden.
    # Some scorebugs only read cleanly every few frames, so not too tight.
    learning_clock_window: float = 12.0
    # A game still learning is hidden (it could be mid-ad) until its scorebug confirms live play.
    # If its scorebug can't be read at all, show it anyway after this long - better than never.
    unverified_grace: float = 45.0
    clearly_game_after: float = 20.0  # overlays steadily up this long count as clearly the game
    # Coming back from a break on a game whose clock can be read: the clock must agree, or the
    # overlays must stay up this long (OCR can miss a few reads). A blip in an ad isn't enough.
    overlay_only_return: float = 6.0


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def edges(grey: np.ndarray) -> np.ndarray:
    return cv2.Canny(grey, 30, 90) > 0


class OverlayModel:
    """Persistent-overlay regions, from a map of how often each pixel is an edge."""

    def __init__(self, persistence: np.ndarray, s: DetectorSettings, threshold: float | None = None):
        self.persistence = persistence.astype(np.float32)
        mask = self.persistence >= (s.edge_persistence if threshold is None else threshold)
        mask[:3, :] = mask[-3:, :] = mask[:, :3] = mask[:, -3:] = False  # player borders

        # Merge nearby edge pixels (letters of one scorebug, one logo) into regions.
        grouped = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(grouped)
        template = cv2.GaussianBlur(self.persistence, (0, 0), 1.0)
        self.regions: list[tuple[slice, slice, np.ndarray]] = []  # (rows, cols, zero-mean template)
        sizes = []
        for label in range(1, n):
            x, y, w, h, _area = stats[label]
            size = int(mask[y:y + h, x:x + w].sum())
            if w * h > s.max_region_area * W * H or size < s.min_region_px:
                continue
            rows, cols = slice(y, y + h), slice(x, x + w)
            t = template[rows, cols] - template[rows, cols].mean()
            self.regions.append((rows, cols, t / (np.linalg.norm(t) + 1e-6)))
            sizes.append(size)
        # The biggest overlay is the scorebug; the others are logos and the like.
        self.primary = int(np.argmax(sizes)) if sizes else -1

    @classmethod
    def from_frames(cls, frames: list[np.ndarray], s: DetectorSettings) -> "OverlayModel":
        return cls(np.mean([edges(g) for g in frames], axis=0), s)

    def scores(self, grey: np.ndarray) -> list[float]:
        """Per region: correlation between the learned edge pattern and the current frame's edges.

        Correlating the *shape* (rather than counting edge pixels) keeps busy scenes - water,
        trees, crowds - from looking like an overlay just because they're full of edges.
        """
        e = cv2.GaussianBlur(edges(grey).astype(np.float32), (0, 0), 1.0)
        out = []
        for rows, cols, t in self.regions:
            cur = e[rows, cols] - e[rows, cols].mean()
            norm = np.linalg.norm(cur)
            out.append(float((t * cur).sum() / norm) if norm > 1e-6 else 0.0)
        return out

    def boxes(self) -> list[tuple[int, int, int, int]]:
        """Region bounding boxes (x, y, w, h) in model coordinates, for debugging overlays."""
        return [(int(c.start), int(r.start), int(c.stop - c.start), int(r.stop - r.start)) for r, c, _ in self.regions]


class BreakDetector:
    def __init__(self, settings: DetectorSettings | None = None, clock_settings: ClockSettings | None = None,
                 read_clock: bool = gameclock.OCR_AVAILABLE,
                 networks: Callable[[], list[tuple[str, "OverlayModel"]]] | None = None,
                 extras: list | None = None):
        self.s = settings or DetectorSettings()
        # Sport-specific evidence (see sports.py): objects with observe(frame, any_overlay=,
        # clearly_game=, live=, t=) -> bool, saying "live play" when the core rules see none.
        self.extras = extras or []
        self.networks = networks  # known networks' overlay models, to start from
        self.network: str | None = None  # which one this broadcast looks like
        self.network_votes: Counter = Counter()  # network names read off the screen
        self._from_network = False  # current model came from the library, not this game
        self._network_hits: dict[str, tuple[float, int]] = {}
        self._first_seen: float | None = None
        self.clock = GameClock(clock_settings)
        self.read_clock = read_clock
        self._last_read = -1e9
        self._primary_since: float | None = None
        self._clock_disagrees_since: float | None = None
        self._stuck_since: float | None = None
        self._overlay_since: float | None = None
        self._recent_overlays: deque[tuple[float, bool]] = deque()
        self.relearned = 0  # times the model was thrown away as wrong
        self.history: deque[tuple[float, np.ndarray]] = deque(maxlen=self.s.history_len)
        self.model: OverlayModel | None = None
        self.model_version = 0  # bumped on every refit, so callers know when to save it
        self._last_sample = -1e9
        self._last_fit = -1e9
        self.state = State.LEARNING
        self.since = 0.0  # when the current state began
        self.detail = ""  # why we're on a break: "ads" or "no scorebug"
        self._present_since: float | None = None
        self._absent_since: float | None = None
        self._primary_seen_at = -1e9
        self.last_scores: list[float] = []
        self.override: tuple[State, float] | None = None  # (state, expires_at)
        self._override_saw_absent = False

    def load(self, persistence: np.ndarray, t: float) -> None:
        """Start from a previously learned model (e.g. after a restart mid-game)."""
        self.model = OverlayModel(persistence, self.s)
        self._primary_seen_at = t

    def showable(self, t: float) -> bool:
        """Can this game be put on screen right now?"""
        if self.state == State.LIVE:
            return True
        if self.state != State.LEARNING or self.clock.readable or self._first_seen is None:
            return False
        return t - max(self.since, self._first_seen) >= self.s.unverified_grace

    def observe(self, frame_bgr: np.ndarray, t: float) -> State:
        if self._first_seen is None:
            self._first_seen = t
        grey = preprocess(frame_bgr)
        learned = self.model is not None and bool(self.model.regions)
        # While learning, a readable scorebug that hasn't been read for a while means we're probably
        # looking at ads: leave those frames out. Otherwise learn from everything (overlays only need
        # to be there most of the time).
        no_recent_clock = self.clock.readable and not self.clock.live(t, within=self.s.learning_clock_window)
        keep = (not no_recent_clock) if not learned else self.state != State.BREAK
        # Break frames stay out of the history so a long ad break can't wash out the overlays.
        if keep and t - self._last_sample >= self.s.sample_every:
            self.history.append((t, grey))
            self._last_sample = t
        self._maybe_refit(t)
        learned = self.model is not None and bool(self.model.regions)
        if not learned and self.networks:
            learned = self._try_networks(grey, t)

        self.last_scores = self.model.scores(grey) if learned else []
        any_overlay = any(sc >= self.s.region_match for sc in self.last_scores)
        if learned and self.last_scores[self.model.primary] >= self.s.region_match:
            self._primary_seen_at = t
            self._primary_since = self._primary_since if self._primary_since is not None else t
        else:
            self._primary_since = None

        # Read the scorebug: is this *now* (not a highlight, another game, or halftime)? Read less
        # often when nothing's on screen - but still read, since the learned overlays can be wrong.
        # Read often when there's a scorebug to read, when overlays and clock disagree, or while
        # learning (the clock is then what decides whether the game is shown).
        busy = any_overlay or not learned or self.clock.live(t, within=15.0)
        every = self.s.read_clock_every if busy else self.s.read_clock_every_idle
        if self.read_clock and t - self._last_read >= every:
            self._last_read = t
            texts = gameclock.read_text(frame_bgr)
            self.clock.update(gameclock.parse(texts), t, [x.s for x in texts])
            self.network_votes.update(gameclock.network_votes([x.s for x in texts]))
        self._overlay_since = (self._overlay_since or t) if any_overlay else None
        # Logo steadily up (not one stray frame that resembles it) while the scorebug's been gone.
        intermission = (any_overlay and t - self._overlay_since >= self.s.intermission_logo_for
                        and t - self._primary_seen_at >= self.s.intermission_after)
        if intermission or (self.state == State.BREAK and t - self.since >= self.s.long_break_halftime):
            self.clock.note_intermission(t)
        halftime = self.clock.in_halftime(t, t - self._primary_since if self._primary_since is not None else 0.0)
        # A highlight or another game's scorebug during play is part of the broadcast: keep showing
        # the game (those readings just don't count as the clock). Only around halftime do old
        # highlights mean "not live".
        stale = self.clock.verdict(t)
        stale_hides = stale is not None and self.clock.around_halftime()
        clock_live = self.clock.live(t, within=self.s.read_clock_every + 0.5)  # just the latest reading
        clearly_game = self.clock.live(t, within=3.0) or (
            self._overlay_since is not None and t - self._overlay_since >= self.s.clearly_game_after)
        extra_votes = [x.observe(frame_bgr, any_overlay=any_overlay, clearly_game=clearly_game,
                                 live=self.state == State.LIVE, t=t) for x in self.extras]

        if learned and self.clock.live(t, within=self.s.read_clock_every * 2) and not any_overlay:
            # The scorebug reads as live play but none of the learned overlays are there: they're
            # wrong (learned from a bumper, an ad, a halftime set). Start again.
            self._clock_disagrees_since = self._clock_disagrees_since or t
            if t - self._clock_disagrees_since >= self.s.clock_disagree_relearn:
                self._relearn(t)
                learned = False
        else:
            self._clock_disagrees_since = None
        if self.state == State.BREAK and not halftime:
            self._stuck_since = self._stuck_since or t
            if learned and t - self._stuck_since >= self.s.stuck_break_relearn:
                self._relearn(t)  # no real ad break lasts this long; the learned overlays must be wrong
                learned = False
        else:
            self._stuck_since = None

        if not learned:
            if self.override and self._override_holds(False, t):
                return self._set(self.override[0], t)
            if halftime or stale_hides:
                self.detail = "halftime" if halftime else stale
                return self._set(State.BREAK, t)
            if clock_live:
                return self._set(State.LIVE, t)
            if any(extra_votes):
                # Still learning, but the sport's own evidence says it's the game (e.g. a highlight
                # package with no scorebug): keep it on screen.
                return self._set(State.LIVE, t)
            if self.clock.readable and not self.clock.live(t, within=self.s.learning_clock_window):
                # Nothing learned yet, but this scorebug can be read and it isn't on screen: an ad,
                # most likely. Don't show it while we wait.
                self.detail = "ads"
                return self._set(State.BREAK, t)
            if self.history and self.history[-1][0] - self.history[0][0] >= self.s.no_overlays_break_after:
                self.detail = "no overlays"
                return self._set(State.BREAK, t)
            return self._set(State.LEARNING, t)

        present = (any_overlay or clock_live or any(extra_votes)) and not intermission and not halftime and not stale_hides
        if present:
            self._absent_since = None
            self._present_since = self._present_since if self._present_since is not None else t
        else:
            self._present_since = None
            self._absent_since = self._absent_since if self._absent_since is not None else t
            self.detail = "halftime" if halftime else (stale if stale_hides else None) or ("no scorebug" if intermission else "ads")

        if self.override and self._override_holds(present, t):
            return self._set(self.override[0], t)
        self.override = None

        # Graphics after a score (scoring drive, stats) keep covering the scorebug for a moment,
        # so "mostly up lately" counts, not just "up without a gap".
        self._recent_overlays.append((t, any_overlay))
        while self._recent_overlays and t - self._recent_overlays[0][0] > self.s.overlay_only_return + 2:
            self._recent_overlays.popleft()
        mostly_up = (len(self._recent_overlays) >= 3 and t - self._recent_overlays[0][0] >= self.s.overlay_only_return
                     and np.mean([up for _, up in self._recent_overlays]) >= 0.7)
        confirmed = not self.clock.readable or self.clock.live(t, within=6.0) or (present and mostly_up)
        if self.state == State.BREAK:
            if present and t - self._present_since >= self.s.live_after and confirmed:
                return self._set(State.LIVE, t)
            return self.state
        if not present and t - self._absent_since >= self.s.break_after:
            return self._set(State.BREAK, t)
        if present and confirmed and self.state in (State.LEARNING, State.STALLED):
            return self._set(State.LIVE, t)
        return self.state

    def _try_networks(self, grey: np.ndarray, t: float) -> bool:
        """Still learning: does this look like a network we already know? Adopt it if so."""
        for name, model in self.networks():
            if any(sc >= self.s.region_match for sc in model.scores(grey)):
                first, n = self._network_hits.get(name, (t, 0))
                self._network_hits[name] = (first, n + 1)
                if n + 1 >= self.s.network_hits and t - first >= self.s.network_hits_span:
                    self.model, self.network, self._from_network = model, name, True
                    self._primary_seen_at = t
                    self._network_hits.clear()
                    return True
            else:
                self._network_hits.pop(name, None)
        return False

    def _relearn(self, t: float) -> None:
        self.model = None
        self.history.clear()
        self.relearned += 1
        self._from_network = False
        self._clock_disagrees_since = None
        self._set(State.LEARNING, t)

    def mark_stalled(self, t: float) -> State:
        return self._set(State.STALLED, t)

    def force(self, state: State, t: float, hold: float = 900.0) -> None:
        """Manual correction from the user.

        "Break" holds until overlays disappear and come back (the break we were told about ends),
        or `hold` seconds - long enough for a halftime show.
        "Live" (i.e. "this is NOT a break") cancels a manual break; otherwise it means the learned
        overlays are wrong for this broadcast, so throw them away and learn again.
        """
        self._present_since = self._absent_since = None
        if state == State.LIVE and self.override:
            self.override = None
            self._set(State.LIVE, t)
            return
        if state == State.LIVE:
            self.override = None
            self._relearn(t)
            self.clock.reset_position(t)  # whatever made it look like a break/highlight was wrong
            return
        self.override = (state, t + hold)
        self._override_saw_absent = False
        self.detail = "marked by you"
        self._set(state, t)

    def _override_holds(self, present: bool, t: float) -> bool:
        forced, expires = self.override
        if t >= expires:
            return False
        if not present:
            self._override_saw_absent = True
            return True
        return not (self._override_saw_absent and t - self._present_since >= self.s.live_after)

    def _set(self, state: State, t: float) -> State:
        if state != self.state:
            self.state, self.since = state, t
        return state

    def _maybe_refit(self, t: float) -> None:
        if len(self.history) < self.s.min_history:
            return
        span = self.history[-1][0] - self.history[0][0]
        if span < (self.s.prior_history_span if self._from_network else self.s.min_history_span):
            return
        if self.model is not None:
            if t - self._last_fit < self.s.refit_every:
                return
            if self.model.regions and t - self._primary_seen_at > self.s.refit_needs_primary_within:
                return
        first = self.model is None or not self.model.regions
        frames = [g for _, g in self.history]
        self.model = OverlayModel.from_frames(frames[-self.s.recent_fit_frames:] if first else frames, self.s)
        self._from_network = False
        self.model_version += 1
        self._last_fit = t
        if first:
            self._primary_seen_at = t
