"""What differs between sports, in one place.

The core detector and game clock are sport-agnostic. Each league maps to a profile with its period
structure, its fallback halftime length, and any sport-specific evidence to plug into the detector.
Logic tuned for one sport goes in its own module (e.g. basketball.py) and is only attached here, so
it can't change how other sports behave.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from .basketball import ReplayByLook


@dataclass(frozen=True)
class Sport:
    name: str
    period_seconds: int  # length of a quarter (or a half, for sports played in halves)
    periods_per_half: int  # 2 = quarters, 1 = halves
    min_halftime: float  # fallback halftime length when the clock can't be read
    extras: tuple[Callable[[], object], ...] = field(default=())  # sport-specific detector evidence

    def make_extras(self) -> list:
        return [make() for make in self.extras]


NFL = Sport("NFL", 15 * 60, 2, 12 * 60)  # halftime ~13 min
COLLEGE_FOOTBALL = Sport("college football", 15 * 60, 2, 18 * 60)  # halftime usually 20 min
NBA = Sport("NBA", 12 * 60, 2, 13 * 60, extras=(ReplayByLook,))  # halftime 15 min
WNBA = Sport("WNBA", 10 * 60, 2, 13 * 60, extras=(ReplayByLook,))
COLLEGE_BASKETBALL = Sport("college basketball", 20 * 60, 1, 13 * 60, extras=(ReplayByLook,))
# Anything else (hockey, soccer, ...): just the core rules, with football-like timing defaults.
OTHER = Sport("other", 15 * 60, 2, 18 * 60)

BY_LEAGUE = {"nfl": NFL, "cfb": COLLEGE_FOOTBALL, "nba": NBA, "wnba": WNBA, "ncaab": COLLEGE_BASKETBALL}


def sport_for(league: str) -> Sport:
    return BY_LEAGUE.get(league, OTHER)
