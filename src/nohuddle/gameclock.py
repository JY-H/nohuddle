"""Read the game clock off the scorebug, to tell live play from highlights and halftime.

Edge matching knows a scorebug is on screen, but not *which moment* it's from: halftime shows
and recaps replay earlier plays with the scorebug from that time still burned in. Reading the
quarter and clock gives a "game time elapsed" that only moves forward during live play, so:

- a scorebug showing a time well behind the furthest point seen is a highlight, not live play
  (a replay of the play that just happened stays within the tolerance);
- "2ND 0:00" / "HALFTIME" starts halftime, and only a 3rd-quarter clock ends it;
- a scorebug with neither of this game's team names is another game's highlight.

OCR uses macOS's Vision framework (local, ~50ms a frame). Elsewhere it's simply unavailable and
the detector falls back to timing rules.
"""

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import cv2
import numpy as np

MAX_CLOCK_SECONDS = 20 * 60  # longest period in any sport we read (college basketball halves)
SUFFIX = {1: "ST", 2: "ND", 3: "RD", 4: "TH"}
QUARTER_RE = re.compile(r"(?<![\d:])([1-4])([A-Z0-9]{2})(?![A-Z0-9])")
# "6:34", "6.34" (OCR), "6:4631" (clock run into the play clock); not "9:30 PM".
CLOCK_RE = re.compile(r"(?<![\d:.])(\d{1,2})[:.;](\d{2})(?![:.]\d)(?!\s*[AP]\.?M)")
# Under a minute some scorebugs drop the minutes ("3RD :47"). Only trusted right after the quarter,
# since a lone ":25" is usually the play clock.
SUBMINUTE_AFTER_QUARTER_RE = re.compile(r"[^\d:]{0,3}[:;](\d{2})(?![\d:.])")
OT_RE = re.compile(r"(?<![A-Z])(\d?OT)(?![A-Z])")
# "2ND & 10", "2ND DOWN", "1ST HALF" - and "2nd.84", which is "2nd & 4" with the & read as an 8.
NOT_A_QUARTER_AFTER = re.compile(r"[\s.,:;]*(&|AND\b|DOWNS?\b|HALF\b|TEAM\b|8\s*\d{1,2}\b(?![:.]))")
# Team-stat panels (halftime, end of quarter) are full of things that look like a scorebug:
# "3rd DOWNS" reads as a quarter and time of possession ("14:56") as a game clock.
STAT_PANEL_RE = re.compile(r"TOTAL\s*YAR|TURNOVERS|POSSESSION|TIME OF POS|\bT\.?O\.?P\b|(?:1ST|3RD|4TH|FIRST|THIRD|FOURTH)\s*DOWNS|PENALTIES|RUSHING\s*YARDS|PASSING\s*YARDS|\bSACKS\b|TACKLES")  # "2ND & 10", "2ND DOWN", "1ST HALF"
# Drive summaries ("SCORING DRIVE · 8 PLAYS · 75 YARDS · 2:20 TIME") show a duration, not the clock.
DRIVE_SUMMARY_RE = re.compile(r"SCORING\s*DRIVE|DRIVE\s*SUMMARY|THIS\s*DRIVE")
WORD_RE = re.compile(r"[A-Z][A-Z.'-]{2,}")
STOPWORDS = {
    "QUARTER", "QTR", "DOWN", "GOAL", "HALF", "HALFTIME", "FINAL", "TIMEOUT", "FLAG", "REVIEW",
    "DRIVE", "PLAYS", "YARDS", "YDS", "TIME", "SCORING", "POSS", "THE", "AND", "LIVE", "REPLAY",
    "SPORTS", "NETWORK", "SATURDAY", "SUNDAY", "MONDAY", "THURSDAY", "NIGHT", "FOOTBALL", "NOON",
    "BIG", "FOX", "CBS", "NBC", "ESPN", "ABC", "TNT", "PRIME", "VIDEO", "SUMMARY", "GAME", "RUSH",
    "PASS", "REC", "TD", "END", "OF", "HOLDING", "PENALTY", "OFFSIDE", "FALSE", "START", "INTERFERENCE",
    "TOUCHDOWN", "FIELD", "INJURY", "CHALLENGE", "TURNOVER", "SACK", "INTERCEPTION", "FUMBLE", "SAFETY",
    "TWO", "MINUTE", "WARNING", "KICKOFF", "PUNT", "ATT", "COMP", "CAR", "YD", "AVG",
}


@dataclass
class Text:
    s: str
    x: float  # centre, as a fraction of the frame
    y: float


@dataclass
class Reading:
    quarter: int | None = None  # 1-4, 5 = overtime
    clock: int | None = None  # seconds left in the quarter
    halftime: bool = False
    words: set[str] = field(default_factory=set)

    def elapsed(self, period: int) -> int | None:
        """Game seconds elapsed, for periods of `period` seconds (overtime counts on from the end)."""
        if self.quarter is None or self.clock is None or self.clock > period:
            return None
        return (self.quarter - 1) * period + (period - self.clock)


def _quarter_match(s: str) -> re.Match | None:
    for m in QUARTER_RE.finditer(s):
        digit, suffix = int(m.group(1)), m.group(2)
        if NOT_A_QUARTER_AFTER.match(s, m.end()):
            continue
        # OCR mangles these ("2N0", "2KD"): accept one wrong character in the suffix.
        if sum(a != b for a, b in zip(suffix, SUFFIX[digit])) <= 1:
            return m
    return None


def _quarter_in(s: str) -> int | None:
    if m := _quarter_match(s):
        return int(m.group(1))
    if OT_RE.search(s):
        return 5
    return None


# OCR sometimes returns Cyrillic look-alikes ("1Sт", "25РЛ"); map them to Latin first.
LOOKALIKES = str.maketrans("тТоОрРнНаАеЕсСвВмМкКхХуУ", "TTOOPPHHAAEECCBBMMKKXXYY")
# Down and distance ("1ST & 10"), including OCR's "1ST 8.10" (& read as 8). Only treated as such
# when a real clock is elsewhere on screen, since "2ND 8.47" can also be a misread clock.
DOWN_DISTANCE_RE = re.compile(r"(?<!\d)[1-4]\s*(?:ST|ND|RD|TH|S[T7]|N[D0O]|R[D0O])\W{0,2}(?:&|8)\W{0,2}(?:\d{1,2}|GOAL|GL)\b")


def _drop_down_and_distance(texts: list[Text]) -> list[Text]:
    clock_texts = {id(t) for t in texts if CLOCK_RE.search(t.s)}
    out = []
    for t in texts:
        if DOWN_DISTANCE_RE.search(t.s) and clock_texts - {id(t)}:
            t = Text(DOWN_DISTANCE_RE.sub(" ", t.s), t.x, t.y)
        out.append(t)
    return out


def parse(texts: list[Text]) -> Reading:
    """Turn recognised text into a reading. Pairs the quarter with the clock next to it."""
    r = Reading()
    texts = _drop_down_and_distance([Text(t.s.translate(LOOKALIKES).upper(), t.x, t.y) for t in texts])
    if len({m.group(0)[:5] for m in STAT_PANEL_RE.finditer(" | ".join(t.s for t in texts))}) >= 2:
        r.halftime = any("HALFTIME" in t.s for t in texts)
        return r  # a stats panel: nothing on it is the game clock
    quarters = [(t, q) for t in texts if (q := _quarter_in(t.s.upper())) is not None]
    clocks = []
    for t in texts:
        for m in CLOCK_RE.finditer(t.s.upper()):
            mins, secs = int(m.group(1)), int(m.group(2))
            if secs < 60 and mins * 60 + secs <= MAX_CLOCK_SECONDS:
                clocks.append((t, mins * 60 + secs))
        if (qm := _quarter_match(t.s.upper())) and (sm := SUBMINUTE_AFTER_QUARTER_RE.match(t.s.upper(), qm.end())):
            if int(sm.group(1)) < 60:
                clocks.append((t, int(sm.group(1))))
    for t in texts:
        up = t.s.upper()
        if "HALFTIME" in up or re.fullmatch(r"\s*HALF\s*", up):
            r.halftime = True
        r.words |= {w.strip(".'-") for w in WORD_RE.findall(up)} - STOPWORDS
    if not quarters:
        # Some scorebugs print the quarter too small to read; a lone clock can still be paired
        # with the quarter seen moments ago (see GameClock.update).
        if len(clocks) == 1:
            r.clock = clocks[0][1]
    if clocks and DRIVE_SUMMARY_RE.search(" | ".join(t.s for t in texts)):
        # Only a clock printed together with the quarter can be trusted next to a drive summary.
        clocks = [c for c in clocks if quarters and c[0] is quarters[0][0]]
    if quarters and not clocks:
        # Under a minute, a lone ":53" anywhere on screen is the game clock if it's the only one
        # (play clocks are shown without a colon). Otherwise only the one right next to the quarter.
        lone = [(t, int(m.group(1))) for t in texts if (m := re.fullmatch(r"\s*[:;](\d{2})\.?\s*", t.s))
                and int(m.group(1)) < 60]
        if len(lone) == 1:
            clocks.append(lone[0])
    if quarters and not clocks:
        # "3RD" and ":47" read as separate pieces: take the one read right next to the quarter
        # (FOX shows ":26" before "2ND", ESPN ":47" after "3RD").
        qt = quarters[0][0]
        i = next(i for i, t in enumerate(texts) if t is qt)
        for nb in (texts[i + 1] if i + 1 < len(texts) else None, texts[i - 1] if i > 0 else None):
            if nb and (m := re.fullmatch(r"\s*[:;](\d{2})\s*", nb.s)) and int(m.group(1)) < 60 \
                    and ((nb.x - qt.x) ** 2 + (nb.y - qt.y) ** 2) ** 0.5 < 0.1:
                clocks.append((nb, int(m.group(1))))
                break
    if quarters:
        qt, r.quarter = quarters[0]
        if clocks:
            # Prefer a clock in the same text as the quarter ("2ND 5:47"), else the nearest one.
            # Other clocks on screen (drive time, kickoff time) sit further away.
            ct, r.clock = min(clocks, key=lambda c: (c[0] is not qt, (c[0].x - qt.x) ** 2 + (c[0].y - qt.y) ** 2))
            if ct is not qt and ((ct.x - qt.x) ** 2 + (ct.y - qt.y) ** 2) ** 0.5 > 0.25:
                r.clock = None
    return r


def similar(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= 0.75)


# Logo text as OCR tends to see it. Order matters: "FOX NFL" is FOX, not NFL Network.
NETWORK_PATTERNS = [
    ("FOX", r"\bFOX\b"),
    ("CBS", r"\bCBS"),  # often read as one word: "CBSSPORTS"
    ("NBC", r"\bNBC\b|PEACOCK|SUNDAY NIGHT FOOTBALL"),
    ("ESPN", r"\bESPN|\bE5PN|MONDAY NIGHT FOOTBALL"),
    ("ABC", r"\babc\b"),
    ("Prime Video", r"PRIME\s*VIDEO|\bTNF\b|THURSDAY NIGHT FOOTBALL"),
    ("Netflix", r"NETFLIX"),
    ("NFL Network", r"NFL\s*NETWORK|\bNFLN\b"),
    ("TNT", r"\bTNT\b"),
]


def network_votes(texts: list[str]) -> Counter:
    joined = " | ".join(texts)
    votes = Counter()
    for name, pattern in NETWORK_PATTERNS:
        # ABC's logo is lowercase; everything else is matched case-insensitively.
        if re.search(pattern, joined if name == "ABC" else joined.upper()):
            votes[name] += 1
    return votes


# ---------- OCR ----------

try:
    import objc
    import Quartz
    import Vision
    from Foundation import NSData

    OCR_AVAILABLE = True
except ImportError:  # not macOS / pyobjc missing
    OCR_AVAILABLE = False


def read_text(frame_bgr: np.ndarray) -> list[Text]:
    if not OCR_AVAILABLE:
        return []
    ok, png = cv2.imencode(".png", frame_bgr)
    if not ok:
        return []
    with objc.autorelease_pool():
        data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
        src = Quartz.CGImageSourceCreateWithData(data, None)
        img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(False)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
        handler.performRequests_error_([req], None)
        out = []
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if not cand:
                continue
            box = obs.boundingBox()  # normalised, origin bottom-left
            out.append(Text(str(cand[0].string()), box.origin.x + box.size.width / 2,
                            1 - (box.origin.y + box.size.height / 2)))
        return out


# ---------- tracking ----------

@dataclass
class ClockSettings:
    rewind_tolerance: float = 150.0  # game seconds behind the furthest point before it's "old"
    ahead_slack: float = 60.0  # game clock can't outrun real time; allow this much for misc. error
    confirm_window: float = 30.0  # wall seconds within which two halftime hints must come
    run_gap: float = 15.0  # readings further apart than this don't continue a run
    run_slack: float = 10.0  # game seconds a run's clock may run ahead of real time (OCR jitter)
    resync_after: float = 30.0  # a steadily running clock for this long is live play, full stop
    quarter_memory: float = 180.0  # how long a read quarter is assumed for quarter-less readings
    resume_slack: float = 30.0  # game seconds a lone reading may differ from where we were and still count
    big_jump: float = 60.0  # moving the position further ahead than this needs a running clock
    verdict_ttl: float = 8.0  # how long a single reading's verdict stands
    min_halftime: float = 12 * 60  # fallback when the clock can't be read: NFL ~13 min
    halftime_resume_scorebug: float = 30.0  # ...and the scorebug back this long
    max_halftime: float = 35 * 60  # give up on a halftime that never ends (missed the 3rd-qtr read)
    min_team_readings: int = 15  # readings needed before trusting the learned team names
    kickoff: float = 0.0  # wall time the game started, if known
    period_seconds: int = 15 * 60  # length of a quarter (or half); set per sport, see sports.py
    periods_per_half: int = 2  # quarters (2) or halves (1)
    # 60 minutes of game clock take ~3 hours (even the quickest games ~2h40), so by wall-clock time
    # since kickoff the game can't be further along than this fraction (plus pregame slack).
    max_pace: float = 0.6
    pace_slack: float = 300.0
    halftime_window: tuple[float, float] = (60 * 60, 130 * 60)  # when halftime can happen, after kickoff


class GameClock:
    def __init__(self, settings: ClockSettings | None = None):
        self.s = settings or ClockSettings()
        self.max_elapsed: int | None = None
        self.halftime_since: float | None = None
        self.halftime_by_clock = False  # started from an actual reading (vs. the timing fallback)
        self.halftime_done = False  # there's only one
        self.readable = False  # ever produced a quarter+clock for this game
        self.team_counts: dict[str, int] = {}
        self.team_readings = 0
        self.max_at = 0.0  # wall time max_elapsed was last raised
        # A "run": consecutive readings whose clock moves the way a live clock does (stopped, or
        # counting down no faster than real time). Live play makes long runs; highlights don't.
        self._run_start: tuple[float, int] | None = None
        self._run_last: tuple[float, int] | None = None
        self._run_n = 0
        self._run_pending: tuple[float, int] | None = None  # a stray reading that may start a new run
        self._quarter: tuple[float, int, int] | None = None  # (wall time, quarter, clock) last read
        self._halftime_hint: float | None = None
        self._verdict: tuple[float, str] | None = None  # (wall time, "live" | "highlight" | "other game")
        self.display = ""  # e.g. "Q2 1:21", the last live reading
        self._confirmed_live_at = -1e9
        self.log: deque[dict] = deque(maxlen=30)  # recent readings, for debugging

    # -- state for persistence --
    def to_json(self) -> dict:
        return {"period_seconds": self.s.period_seconds,
                "max_elapsed": self.max_elapsed, "max_at": self.max_at, "halftime_since": self.halftime_since,
                "halftime_by_clock": self.halftime_by_clock, "halftime_done": self.halftime_done,
                "readable": self.readable,
                "team_counts": self.team_counts, "team_readings": self.team_readings}

    def load(self, d: dict) -> None:
        if d.get("period_seconds", 15 * 60) != self.s.period_seconds:
            return  # saved with a different period length: its game-time numbers don't apply
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @property
    def teams(self) -> list[str]:
        if self.team_readings < self.s.min_team_readings:
            return []
        return [w for w, c in sorted(self.team_counts.items(), key=lambda kv: -kv[1])
                if c >= 0.3 * self.team_readings][:4]

    @property
    def _period(self) -> int:
        return self.s.period_seconds

    @property
    def _half(self) -> int:
        return self.s.period_seconds * self.s.periods_per_half

    def _max_possible(self, t: float) -> float:
        """The furthest the game can be by now, going by kickoff time (inf if unknown)."""
        if not self.s.kickoff or t < self.s.kickoff:
            return float("inf")
        return self.s.max_pace * (t - self.s.kickoff) + self.s.pace_slack

    def update(self, r: Reading, t: float, raw: list[str] | None = None) -> None:
        if raw is not None:
            self.log.append({"t": round(t, 1), "raw": raw[:12], "quarter": r.quarter, "clock": r.clock,
                             "max_elapsed": self.max_elapsed})
        possible = self._max_possible(t)
        if self.max_elapsed is not None and self.max_elapsed > possible:
            # Our position is further along than the game can be (a misread got trusted): forget it,
            # and anything that followed from it.
            self.log.append({"t": round(t, 1), "resync": None, "was": self.max_elapsed, "why": "impossible"})
            self.max_elapsed, self._quarter = None, None
            self._run_start = self._run_last = None
            self._run_n = 0
            if possible < self._half:
                self.halftime_since, self.halftime_done = None, False
        if r.halftime:
            if self._halftime_hint is not None and t - self._halftime_hint <= self.s.confirm_window:
                self._start_halftime(t, by_clock=True)
            self._halftime_hint = t
        if r.quarter is None and r.clock is not None and self._quarter is not None:
            qt, q, last_clock = self._quarter
            # Same quarter if it was read recently and the clock hasn't jumped back up.
            if t - qt <= self.s.quarter_memory and r.clock <= last_clock + 30:
                r = Reading(q, r.clock, r.halftime, r.words)
        if r.quarter is None and r.clock is not None and self.max_elapsed is not None:
            # Quarter unreadable for a while (some scorebugs print it tiny): if the clock is still
            # counting down within the quarter the game was last in, it's that quarter.
            q = min(5, self.max_elapsed // self._period + 1)
            if r.clock <= self._period - self.max_elapsed % self._period + 30:
                r = Reading(q, r.clock, r.halftime, r.words)
        if r.quarter is not None and r.clock is not None and self._quarter and r.quarter != self._quarter[1]:
            # A clock that continues the one running right now, to the second, is the live game
            # whatever the quarter said: misreads ("2nd & 4" as a quarter) beat coincidences.
            alt = Reading(self._quarter[1], r.clock, r.halftime, r.words)
            if self._continues_run(t, alt.elapsed(self._period)):
                r = alt
        e = r.elapsed(self._period)
        if e is None:
            return
        if e > possible:
            return  # further along than the game can be yet: a misread (e.g. down-and-distance)
        self.readable = True
        self._quarter = (t, r.quarter, r.clock)
        run_for = self._extend_run(t, e)

        if self.max_elapsed is None:
            if self._run_n < 2:
                self._verdict = None  # one reading could be anything; wait for a second
                return
            self._raise_max(e, t)
        elif e > self.max_elapsed + (t - self.max_at) + self.s.ahead_slack:
            # Further ahead than the game clock could have run since we last saw it: another
            # game's scorebug (look-ins, NFL highlights) or a misread. Never trust it outright...
            teams = self.teams
            other = teams and not any(similar(w, team) for w in r.words for team in teams)
            self._verdict = (t, "other game") if other and run_for > 0 else None
            if other or run_for < self.s.resync_after:
                return
            self._raise_max(e, t, resync=True)  # ...unless it keeps running steadily: we were wrong
        elif e < self.max_elapsed - self.s.rewind_tolerance:
            if run_for < self.s.resync_after:
                # Two consistent readings behind us: a highlight. A single one: probably a misread.
                self._verdict = (t, "highlight") if run_for > 0 else None
                return
            self._raise_max(e, t, resync=True)  # a minute of steadily running clock isn't a highlight

        self._verdict = (t, "live")
        q = "OT" if r.quarter == 5 else f"Q{r.quarter}"
        self.display = f"{q} {r.clock // 60}:{r.clock % 60:02d}"
        # Two agreeing readings, or one that picks up right where the game was: coming back from
        # a break, the next reading often follows a run of stat graphics with no clock on them.
        if self._run_n < 2 and not self._resumes(e, t):
            return
        self._confirmed_live_at = t
        if e > self.max_elapsed and (e - self.max_elapsed <= self.s.big_jump or self._clock_running(e)):
            self._raise_max(e, t)  # big jumps (e.g. a misread right after a break) need a running clock
        for w in r.words:
            self.team_counts[w] = self.team_counts.get(w, 0) + 1
        self.team_readings += 1
        if r.quarter is not None and r.quarter <= self.s.periods_per_half and self.halftime_done:
            self.halftime_done = False  # still in the first half: whatever said otherwise was wrong
        if r.quarter == self.s.periods_per_half and r.clock == 0:
            self._start_halftime(t, by_clock=True)
        elif r.quarter is not None and r.quarter > self.s.periods_per_half and (
                self.halftime_since is None or self._clock_running(e)):
            # Second half underway - but only believe it once the 3rd-quarter clock actually runs:
            # a frozen "3RD 15:00" or a graphic can show up during the halftime show.
            self.halftime_since = None
            self.halftime_done = True

    def _resumes(self, e: int, t: float) -> bool:
        if self.max_elapsed is None:
            return False
        return self.max_elapsed - self.s.resume_slack <= e <= self.max_elapsed + min(t - self.max_at, 300) + self.s.resume_slack

    def _clock_running(self, e: int) -> bool:
        return self._run_start is not None and self._run_n >= 2 and e - self._run_start[1] >= 3

    def _continues(self, last: tuple[float, int] | None, t: float, e: int | None) -> bool:
        if last is None or e is None:
            return False
        lt, le = last
        return t - lt <= self.s.run_gap and 0 <= e - le <= (t - lt) + self.s.run_slack

    def _continues_run(self, t: float, e: int | None) -> bool:
        return self._continues(self._run_last, t, e)

    def _extend_run(self, t: float, e: int) -> float:
        """Track runs of live-looking readings; returns how long the current run has lasted (0 for a
        reading that doesn't belong to it). One stray reading (a misread) doesn't end a run; if the
        next reading agrees with the stray one instead, that's a new run of two."""
        if self._continues_run(t, e):
            self._run_last = (t, e)
            self._run_n += 1
            self._run_pending = None
            return t - self._run_start[0]
        if self._continues(self._run_pending, t, e):
            self._run_start, self._run_last, self._run_n = self._run_pending, (t, e), 2
            self._run_pending = None
            return t - self._run_start[0]
        if self._run_last is not None and self._run_pending is None and t - self._run_last[0] <= self.s.run_gap:
            self._run_pending = (t, e)
            return 0.0
        self._run_start = self._run_last = (t, e)
        self._run_n = 1
        self._run_pending = None
        return 0.0

    def around_halftime(self) -> bool:
        """End of the first half reached, second half not started: old-scorebug highlights here are
        the halftime show, not live play."""
        return (self.max_elapsed is not None and not self.halftime_done
                and self.max_elapsed >= self._half - 30)

    def reset_position(self, t: float) -> None:
        """Forget where we think the game is (the user says what's on screen is live)."""
        self.log.append({"t": round(t, 1), "resync": None, "was": self.max_elapsed, "why": "reset by you"})
        self.max_elapsed, self._quarter, self._verdict = None, None, None
        self._run_start = self._run_last = None
        self._run_n = 0
        self._run_pending = None

    def _raise_max(self, e: int, t: float, resync: bool = False) -> None:
        if resync:
            self.log.append({"t": round(t, 1), "resync": e, "was": self.max_elapsed})
            self.max_elapsed = e
        else:
            self.max_elapsed = max(self.max_elapsed or 0, e)
        self.max_at = t

    def note_intermission(self, t: float) -> None:
        """The scorebug has been gone for minutes with only the logo up. If the last thing we
        read was the 2nd quarter (or we can't read this scorebug at all), call it halftime."""
        if self.halftime_since is not None:
            return
        if not self.readable:
            # Can't read this scorebug: go by the time of day instead, so a long injury stoppage
            # in the 1st quarter isn't mistaken for halftime.
            lo, hi = self.s.halftime_window
            if not self.s.kickoff or lo <= t - self.s.kickoff <= hi:
                self._start_halftime(t, by_clock=False)
        elif self.max_elapsed is not None and self._half - self._period <= self.max_elapsed <= self._half:
            self._start_halftime(t, by_clock=True)

    def _start_halftime(self, t: float, by_clock: bool) -> None:
        if self.halftime_since is None and not self.halftime_done and \
                (self.max_elapsed is None or self.max_elapsed <= self._half):
            self.halftime_since = t
            self.halftime_by_clock = by_clock

    def in_halftime(self, t: float, scorebug_for: float) -> bool:
        """`scorebug_for`: how long the scorebug has been continuously on screen."""
        if self.halftime_since is None:
            return False
        # With a readable clock, halftime ends on a 3rd-quarter reading (see update()); without
        # one, fall back to a minimum halftime length plus the scorebug being back for a while.
        timed_out = t - self.halftime_since > self.s.max_halftime
        fallback_done = (
            not (self.halftime_by_clock and self.readable)
            and t - self.halftime_since >= self.s.min_halftime
            and scorebug_for >= self.s.halftime_resume_scorebug
        )
        ended = timed_out or fallback_done
        if ended:
            self.halftime_since = None
            self.halftime_done = True
            # Halftime is over, so whatever the clock said, we're in the second half now.
            self._raise_max(self._half, t)
        return not ended

    def live(self, t: float, within: float = 3.0) -> bool:
        """Two agreeing, current readings just now: the scorebug says this is live play."""
        return t - self._confirmed_live_at <= within and self.halftime_since is None

    def verdict(self, t: float) -> str | None:
        """"highlight" / "other game" if the scorebug just read as not-live, else None."""
        if self._verdict is None or t - self._verdict[0] > self.s.verdict_ttl:
            return None
        return None if self._verdict[1] == "live" else self._verdict[1]

