"""Rebuild src/nohuddle/scorigami_scores.json: every NFL (and AFL) final score so far.

    uv run python tools/scorigami_data.py

Games through 1998 come from FiveThirtyEight's NFL Elo data (1920-2020), 1999 onward from
nflverse's schedule (kept current through the season). Where the two overlap they agree game for
game, and the result matches nflscorigami.com's table. Scores since the file was built are added
at runtime from ESPN's scoreboard (see scorigami.py).
"""

import csv
import io
import json
import urllib.request
from pathlib import Path

ELO = "https://raw.githubusercontent.com/fivethirtyeight/nfl-elo-game/master/data/nfl_games.csv"
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
OUT = Path(__file__).parent.parent / "src" / "nohuddle" / "scorigami_scores.json"


def rows(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode())))


def main() -> None:
    games = [(r["date"], int(r["score1"]), int(r["score2"])) for r in rows(ELO) if int(r["season"]) < 1999]
    games += [(r["gameday"], int(r["home_score"]), int(r["away_score"]))
              for r in rows(NFLVERSE) if r["home_score"] != ""]
    scores: dict[tuple[int, int], list] = {}
    for date, a, b in sorted(games):
        entry = scores.setdefault((max(a, b), min(a, b)), [0, date])
        entry[0] += 1
        entry[1] = date
    OUT.write_text(json.dumps({
        "through": max(d for d, _, _ in games),
        "scores": [[w, l, n, last] for (w, l), (n, last) in sorted(scores.items())],
    }, separators=(",", ":")))
    print(f"{len(games)} games, {len(scores)} distinct final scores, through {max(d for d, _, _ in games)}")


if __name__ == "__main__":
    main()
