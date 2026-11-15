"""Your ESPN fantasy football matchup, live, for the drawer next to the games.

Reads your own league from ESPN's fantasy API with your ESPN session (espn_s2 / SWID), which is
what that API accepts - ESPN's Fantasycast *website* wants a full MyDisney login instead.
"""

import time
from datetime import date

import aiohttp

API = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league}"
SLOTS = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST", 17: "K", 23: "FLEX", 20: "BN", 21: "IR"}
SLOT_ORDER = [0, 2, 4, 6, 23, 16, 17, 20, 21]
BENCH = {20, 21}
PRO_TEAMS = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN",
    11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WSH", 29: "CAR",
    30: "JAX", 33: "BAL", 34: "HOU",
}


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


class Fantasy:
    def __init__(self, league_id: str, team_id: str, espn_s2: str, swid: str, cache_seconds: float = 30.0):
        self.league_id, self.team_id = league_id, int(team_id)
        self._cookies = {"espn_s2": espn_s2, "SWID": swid}
        self._cache: tuple[float, dict] | None = None
        self._cache_seconds = cache_seconds

    async def matchup(self) -> dict:
        if self._cache and time.time() - self._cache[0] < self._cache_seconds:
            return self._cache[1]
        base = API.format(season=current_season(), league=self.league_id)
        async with aiohttp.ClientSession(cookies=self._cookies, headers={"User-Agent": "Mozilla/5.0"},
                                         timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.get(base + "?view=mTeam&view=mSettings") as r:
                r.raise_for_status()
                league = await r.json(content_type=None)
            period, scoring = league["status"]["currentMatchupPeriod"], league["scoringPeriodId"]
            async with s.get(base + f"?view=mMatchupScore&view=mBoxscore&view=mRoster&scoringPeriodId={scoring}") as r:
                r.raise_for_status()
                box = await r.json(content_type=None)
        teams = {t["id"]: t for t in league.get("teams", [])}
        game = next((m for m in box.get("schedule", []) if m.get("matchupPeriodId") == period
                     and self.team_id in (m.get("home", {}).get("teamId"), m.get("away", {}).get("teamId"))), None)
        if game is None:
            result = {"week": period, "me": None, "opp": None}
        else:
            mine = "home" if game["home"]["teamId"] == self.team_id else "away"
            result = {"week": period,
                      "me": self._side(game.get(mine, {}), teams),
                      "opp": self._side(game.get("away" if mine == "home" else "home", {}), teams)}
        self._cache = (time.time(), result)
        return result

    @staticmethod
    def _side(side: dict, teams: dict) -> dict | None:
        if not side:
            return None  # a bye
        team = teams.get(side.get("teamId"), {})
        players = []
        for e in side.get("rosterForCurrentScoringPeriod", {}).get("entries", []):
            pe, slot = e.get("playerPoolEntry", {}), e.get("lineupSlotId")
            p = pe.get("player", {})
            proj = next((s.get("appliedTotal") for s in p.get("stats", []) if s.get("statSourceId") == 1), None)
            players.append({
                "slot": SLOTS.get(slot, "?"), "starter": slot not in BENCH, "name": p.get("fullName", "?"),
                "team": PRO_TEAMS.get(p.get("proTeamId"), ""), "points": round(pe.get("appliedStatTotal") or 0, 1),
                "projected": round(proj, 1) if proj is not None else None,
                "_order": SLOT_ORDER.index(slot) if slot in SLOT_ORDER else 99,
            })
        players.sort(key=lambda x: x.pop("_order"))
        return {
            "name": team.get("name") or team.get("abbrev", "?"), "abbrev": team.get("abbrev", ""),
            "points": round(side.get("totalPointsLive", side.get("totalPoints")) or 0, 1),
            "projected": round(side.get("totalProjectedPointsLive", side.get("totalProjectedPoints")) or 0, 1),
            "win_probability": side.get("winProbability"),
            "players": players,
        }
