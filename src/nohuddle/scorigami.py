"""Scorigami: a final score that has never happened in NFL history (Jon Bois's term).

For each NFL game on the watchlist: whether the score right now would be one if the game ended,
and the chance that the final score will be. History is bundled (scorigami_scores.json, rebuilt by
tools/scorigami_data.py) and kept current without rebuilding it: once a day the games played since
are read from nflverse's schedule (cached in the data folder), and anything newer still - today's
finals - from ESPN's public scoreboard, which also gives the live scores and clocks.

The chance comes from simulating the rest of the game a drive at a time: about 22 drives a game
between the two teams, each ending in a touchdown (with its extra point, two-point try or miss), a
field goal, a safety, a defensive score or nothing, at league-average rates. It knows nothing about
the teams or the game situation (who has the ball, a team protecting a lead), so it's a rough guide.
"""

import csv
import io
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
import numpy as np

log = logging.getLogger("nohuddle")

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
REFRESH_EVERY = 24 * 3600
RETRY_AFTER = 3600  # after a failed refresh
GAP_DAYS = 10  # days between our history and this week fetched from ESPN (nflverse covers the rest)
HISTORY = Path(__file__).parent / "scorigami_scores.json"
EASTERN = ZoneInfo("America/New_York")  # the NFL's (and ESPN's scoreboard dates') time zone

QUARTER = 900
DRIVES_PER_GAME = 21.6  # both teams together
# What a drive ends in, for the team with the ball (+) or its opponent (-).
DRIVE_POINTS = np.array([7, 6, 8, 3, 0, -2, -7])
DRIVE_ODDS = np.array([0.198, 0.016, 0.011, 0.160, 0.599, 0.004, 0.012])  # TD+XP, TD missed, TD+2, FG, none, safety, D/ST TD
# Regulation ends tied: overtime (each team gets the ball; a tie after 10 minutes stands).
OT_MARGIN = np.array([0, 3, 6, 7])
OT_ODDS = np.array([0.08, 0.55, 0.27, 0.10])
SIMULATIONS = 20000


def load_history(path: Path = HISTORY) -> tuple[str, dict[tuple[int, int], tuple[int, str]]]:
    """(the last game date covered, {(winner's points, loser's points): (times, last date)})."""
    d = json.loads(path.read_text())
    return d["through"], {(w, l): (n, last) for w, l, n, last in d["scores"]}


def seconds_left(period: int, clock: float) -> float:
    """Regulation time left; 0 in overtime."""
    return (4 - period) * QUARTER + clock if period <= 4 else 0.0


def simulate(a: int, b: int, left: float, rng: np.random.Generator, n: int = SIMULATIONS) -> np.ndarray:
    """Final scores (n x 2, team a and team b) from a-b with `left` seconds of regulation to go."""
    drives = left / (4 * QUARTER) * DRIVES_PER_GAME
    count = np.floor(drives).astype(int) + (rng.random(n) < drives % 1)  # e.g. 10.4 -> 10 or 11
    first = rng.random(n) < 0.5  # who has the ball next: unknown
    final = np.tile(np.array([a, b], dtype=int), (n, 1))
    for i in range(int(count.max(initial=0))):
        going = i < count
        pts = DRIVE_POINTS[rng.choice(len(DRIVE_POINTS), size=n, p=DRIVE_ODDS)]
        offense = np.where(first ^ (i % 2 == 1), 0, 1)  # drives alternate
        scorer = np.where(pts >= 0, offense, 1 - offense)
        final[np.arange(n), scorer] += np.where(going, np.abs(pts), 0)
    tied = final[:, 0] == final[:, 1]
    margin = OT_MARGIN[rng.choice(len(OT_MARGIN), size=n, p=OT_ODDS)]
    winner = (rng.random(n) < 0.5).astype(int)
    final[np.arange(n), winner] += np.where(tied, margin, 0)
    return final


def chances(a: int, b: int, left: float, seen: set[tuple[int, int]], rng: np.random.Generator) -> dict:
    """The chance the final score is a scorigami, and the likeliest such finals."""
    final = simulate(a, b, left, rng)
    keys = np.maximum(final[:, 0], final[:, 1]) * 1000 + np.minimum(final[:, 0], final[:, 1])
    values, counts = np.unique(keys, return_counts=True)
    new = [(int(v), int(c)) for v, c in zip(values, counts) if (v // 1000, v % 1000) not in seen]
    new.sort(key=lambda vc: -vc[1])
    return {"p": sum(c for _, c in new) / len(final),
            "likeliest": [{"score": f"{v // 1000}–{v % 1000}", "p": c / len(final)} for v, c in new[:3]]}


class Scorigami:
    def __init__(self, cache: Path, ttl: float = 15.0):
        self.through, self.history = load_history()
        self._cache = cache  # games since the bundled history, from nflverse
        self._ttl = ttl
        self._board: tuple[float, list[dict]] | None = None
        self._past_days: dict[str, list[dict]] = {}
        self._refresh_tried = 0.0
        self._known = self.history  # history plus finals since, as of the last board
        self._rng = np.random.default_rng()
        self._bundled_through = self.through
        self._apply_cache()

    def _apply_cache(self) -> None:
        if not self._cache.exists():
            return
        self.through, self.history = load_history()
        for day, a, b in json.loads(self._cache.read_text())["games"]:
            if day > self._bundled_through:
                n, _ = self.history.get((max(a, b), min(a, b)), (0, day))
                self.history[(max(a, b), min(a, b))] = (n + 1, day)
                self.through = max(self.through, day)

    async def _refresh(self) -> None:
        """Once a day: games played since the bundled history, from nflverse's schedule."""
        if self._cache.exists() and time.time() - self._cache.stat().st_mtime < REFRESH_EVERY:
            return
        if time.time() - self._refresh_tried < RETRY_AFTER:
            return
        self._refresh_tried = time.time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
                async with s.get(NFLVERSE) as r:
                    r.raise_for_status()
                    text = await r.text()
            games = [(g["gameday"], int(g["home_score"]), int(g["away_score"]))
                     for g in csv.DictReader(io.StringIO(text))
                     if g["home_score"] != "" and g["gameday"] > self._bundled_through]
        except Exception as e:
            log.warning("Scorigami: couldn't refresh the score history: %s", type(e).__name__)
            return  # ESPN's scoreboard still covers what's missing
        self._cache.parent.mkdir(parents=True, exist_ok=True)
        self._cache.write_text(json.dumps({"games": games}))
        self._board = None
        self._apply_cache()

    async def _events(self) -> list[dict]:
        """ESPN's scoreboard: this week's games (live, and finals our history doesn't have yet), plus
        any days between our history and this week (nflverse can lag a day or two)."""
        await self._refresh()
        if self._board and time.time() - self._board[0] < self._ttl:
            return self._board[1]
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            events = await _scoreboard(s, {})  # no date: the current week
            day = date.fromisoformat(self.through) + timedelta(days=1)
            today = datetime.now(EASTERN).date().isoformat()
            first = min([_day(e) for e in events] + [today])
            for _ in range(GAP_DAYS):
                if day.isoformat() >= first:
                    break
                key = f"{day:%Y%m%d}"  # a date gives that day's games; past days' don't change
                if key not in self._past_days:
                    self._past_days[key] = await _scoreboard(s, {"dates": key})
                events = events + self._past_days[key]
                day += timedelta(days=1)
        self._board = (time.time(), events)
        return events

    async def board(self, games: list[tuple[str, list[str]]]) -> dict:
        """`games`: (title, its teams' abbreviations) for each NFL game being watched."""
        events = await self._events()
        history = dict(self.history)
        latest = []  # scores (and scorigamis) since our bundled history
        for e in events:
            st, sides = e["status"], _sides(e)
            day = _day(e)
            if st["type"]["state"] == "post" and day > self.through and sides:
                key = _key(sides)
                n, _ = history.get(key, (0, day))
                if n == 0:
                    latest.append({"score": f"{key[0]}–{key[1]}", "game": e.get("shortName", ""), "date": day})
                history[key] = (n + 1, day)
        self._known = history
        seen = set(history)
        out = []
        for title, teams in games:
            e = max((e for e in events if {t for t, _ in _sides(e)} == set(teams) and len(teams) == 2),
                    key=lambda e: e["date"], default=None)  # the latest, should they have met twice
            if e is None:
                out.append({"title": title, "found": False})
                continue
            sides, st = _sides(e), e["status"]
            state, period, clock = st["type"]["state"], st.get("period", 0), float(st.get("clock", 0))
            # The game's own final counts as history once it's in: judge it against what came before.
            if state == "post":
                key = _key(sides)
                n, last = history.get(key, (1, None))
                n -= 1
                now = {"times": n, "last": last if n else None}
                res = {"p": 1.0 if n == 0 else 0.0, "likeliest": []}
            else:
                key = _key(sides)
                n, last = history.get(key, (0, None))
                now = {"times": n, "last": last}
                left = 4 * QUARTER if state == "pre" else seconds_left(period, clock)
                res = chances(sides[0][1], sides[1][1], left, seen, self._rng)
            out.append({"title": title, "found": True, "state": state, "status": st["type"].get("shortDetail", ""),
                        "teams": [{"abbrev": t, "score": s} for t, s in sides], "now": now, **res})
        return {"distinct": len(history), "latest": latest, "games": out}


    def table(self) -> list[list]:
        """Every final score so far: [winner's points, loser's points, times, last date]."""
        return [[w, l, n, last] for (w, l), (n, last) in sorted(self._known.items())]


async def _scoreboard(s: aiohttp.ClientSession, params: dict) -> list[dict]:
    async with s.get(SCOREBOARD, params=params) as r:
        r.raise_for_status()
        return (await r.json()).get("events", [])


def _day(e: dict) -> str:
    """The game's date where the NFL keeps time (a late Monday game ends on Monday there)."""
    return datetime.fromisoformat(e["date"].replace("Z", "+00:00")).astimezone(EASTERN).date().isoformat()


def _sides(e: dict) -> list[tuple[str, int]]:
    comp = (e.get("competitions") or [{}])[0]
    return [(c["team"]["abbreviation"], int(c.get("score") or 0)) for c in comp.get("competitors", [])]


def _key(sides: list[tuple[str, int]]) -> tuple[int, int]:
    a, b = sides[0][1], sides[1][1]
    return max(a, b), min(a, b)
