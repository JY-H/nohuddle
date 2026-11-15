"""Your ESPN fantasy football matchup, live, for the drawer next to the games.

Reads your own league from ESPN's fantasy API with your ESPN session (espn_s2 / SWID), which is
what that API accepts - ESPN's Fantasycast *website* wants a full MyDisney login instead.
"""

import json
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

NICKNAMES = {
    "Falcons": "ATL", "Bills": "BUF", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE", "Cowboys": "DAL",
    "Broncos": "DEN", "Lions": "DET", "Packers": "GB", "Titans": "TEN", "Colts": "IND", "Chiefs": "KC",
    "Raiders": "LV", "Rams": "LAR", "Dolphins": "MIA", "Vikings": "MIN", "Patriots": "NE", "Saints": "NO",
    "Giants": "NYG", "Jets": "NYJ", "Eagles": "PHI", "Cardinals": "ARI", "Steelers": "PIT", "Chargers": "LAC",
    "49ers": "SF", "Seahawks": "SEA", "Buccaneers": "TB", "Commanders": "WSH", "Panthers": "CAR",
    "Jaguars": "JAX", "Ravens": "BAL", "Texans": "HOU",
}


def pro_teams_in(title: str) -> set[str]:
    """The NFL teams in a game's title, e.g. "Detroit Lions vs New England Patriots" -> {DET, NE}."""
    return {abbrev for word in title.replace("-", " ").split() if (abbrev := NICKNAMES.get(word))}


# ESPN stat IDs, each checked against real box scores (e.g. completions + incompletions = attempts,
# receptions <= targets, FG made + missed = attempted). IDs we couldn't confirm are left out.
PASS_ATT, PASS_CMP, PASS_YDS, PASS_TD, PASS_INT = 0, 1, 3, 4, 20
RUSH_ATT, RUSH_YDS, RUSH_TD = 23, 24, 25
REC, REC_YDS, REC_TD, TARGETS = 53, 42, 43, 58
FG_MADE, FG_ATT, XP_MADE, XP_ATT = 83, 84, 86, 87
DEF_INT, DEF_FR, DEF_SACK, DEF_PA = 95, 96, 99, 120


def stat_line(position: int | None, stats: dict[int, float]) -> str:
    """A Fantasycast-style line, e.g. "26/43, 228 yds, 1 TD, 1 INT" or "7 rec (10 tgt), 51 yds"."""
    g = lambda k: int(round(stats.get(k, 0)))
    parts = []
    if position == 16:  # D/ST
        if DEF_PA in stats:
            parts.append(f"{g(DEF_PA)} PA")
        parts += [f"{g(k)} {label}" for k, label in ((DEF_SACK, "sack"), (DEF_INT, "INT"), (DEF_FR, "FR")) if g(k)]
        return ", ".join(parts)
    if position == 5:  # K
        if g(FG_ATT):
            parts.append(f"FG {g(FG_MADE)}/{g(FG_ATT)}")
        if g(XP_ATT):
            parts.append(f"XP {g(XP_MADE)}/{g(XP_ATT)}")
        return ", ".join(parts)
    if g(PASS_ATT):
        parts.append(f"{g(PASS_CMP)}/{g(PASS_ATT)}, {g(PASS_YDS)} yds"
                     + (f", {g(PASS_TD)} TD" if g(PASS_TD) else "") + (f", {g(PASS_INT)} INT" if g(PASS_INT) else ""))
    if g(RUSH_ATT):
        parts.append(f"{g(RUSH_ATT)} car, {g(RUSH_YDS)} yds" + (f", {g(RUSH_TD)} TD" if g(RUSH_TD) else ""))
    if g(TARGETS) or g(REC):
        parts.append(f"{g(REC)} rec ({g(TARGETS)} tgt), {g(REC_YDS)} yds" + (f", {g(REC_TD)} TD" if g(REC_TD) else ""))
    return " · ".join(parts)


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


class Fantasy:
    """Reads are cached: the league (teams, current week) for an hour, scores for 30s - and only
    what's being looked at is fetched, filtered server-side (the unfiltered week is ~7 MB)."""

    def __init__(self, league_id: str, team_id: str, espn_s2: str, swid: str, cache_seconds: float = 30.0):
        self.league_id, self.team_id = league_id, int(team_id)
        self._cookies = {"espn_s2": espn_s2, "SWID": swid}
        self._cache: dict[object, tuple[float, object]] = {}
        self._cache_seconds = cache_seconds

    async def _get(self, query: str, filter_: dict | None = None) -> dict:
        base = API.format(season=current_season(), league=self.league_id)
        headers = {"User-Agent": "Mozilla/5.0"}
        if filter_:
            headers["x-fantasy-filter"] = json.dumps(filter_)
        async with aiohttp.ClientSession(cookies=self._cookies, headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.get(base + query) as r:
                r.raise_for_status()
                return await r.json(content_type=None)

    async def _cached(self, key: object, ttl: float, fetch) -> object:
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        value = await fetch()
        self._cache[key] = (time.time(), value)
        return value

    async def _league(self) -> dict:
        async def fetch():
            d = await self._get("?view=mTeam&view=mSettings")
            return {"period": d["status"]["currentMatchupPeriod"], "scoring": d["scoringPeriodId"],
                    "teams": {t["id"]: t for t in d.get("teams", [])}}
        return await self._cached("league", 3600, fetch)

    async def _week(self, team_id: int | None) -> tuple[dict, list[dict]]:
        """This week's matchups (all of them, or just `team_id`'s), with rosters and live points."""
        lg = await self._league()
        flt = {"schedule": {"filterMatchupPeriodIds": {"value": [lg["period"]]}}}
        if team_id is not None:
            flt["schedule"]["filterTeamIds"] = {"value": [team_id]}

        async def fetch():
            d = await self._get(f"?view=mMatchupScore&view=mBoxscore&scoringPeriodId={lg['scoring']}", flt)
            return [m for m in d.get("schedule", []) if m.get("matchupPeriodId") == lg["period"]]
        return lg, await self._cached(("week", team_id), self._cache_seconds, fetch)

    async def matchup(self, team_id: int | None = None) -> dict:
        """`team_id`'s matchup this week (yours by default), plus the league's teams to pick from."""
        team_id = team_id or self.team_id
        lg, games = await self._week(team_id)
        game = next((m for m in games if team_id in (m.get("home", {}).get("teamId"), m.get("away", {}).get("teamId"))), None)
        result = {"week": lg["period"], "team_id": team_id, "my_team_id": self.team_id,
                  "teams": sorted(({"id": i, "name": t.get("name") or t.get("abbrev", "?")} for i, t in lg["teams"].items()),
                                  key=lambda t: (t["id"] != self.team_id, t["name"].lower())),
                  "me": None, "opp": None}
        if game is not None:
            mine = "home" if game["home"]["teamId"] == team_id else "away"
            result["me"] = self._side(game.get(mine, {}), lg["teams"], lg["scoring"])
            result["opp"] = self._side(game.get("away" if mine == "home" else "home", {}), lg["teams"], lg["scoring"])
        return result

    async def started_in(self, pro_teams: set[str]) -> dict:
        """Every fantasy team's starters who play for `pro_teams` (e.g. the game on screen)."""
        lg, games = await self._week(None)
        players = []
        for m in games:
            for key in ("home", "away"):
                side = self._side(m.get(key, {}), lg["teams"], lg["scoring"])
                if side is None:
                    continue
                players += [{**p, "fantasy_team": side["name"], "fantasy_team_id": side["id"],
                             "mine": side["id"] == self.team_id}
                            for p in side["players"] if p["starter"] and p["team"] in pro_teams]
        players.sort(key=lambda p: (not p["mine"], p["team"], -p["points"]))
        return {"week": lg["period"], "players": players}

    @staticmethod
    def _side(side: dict, teams: dict, scoring: int) -> dict | None:
        if not side:
            return None  # a bye
        team = teams.get(side.get("teamId"), {})
        players = []
        for e in side.get("rosterForCurrentScoringPeriod", {}).get("entries", []):
            pe, slot = e.get("playerPoolEntry", {}), e.get("lineupSlotId")
            p = pe.get("player", {})
            proj = next((s.get("appliedTotal") for s in p.get("stats", []) if s.get("statSourceId") == 1), None)
            actual = next((s.get("stats") or {} for s in p.get("stats", [])
                           if s.get("statSourceId") == 0 and s.get("scoringPeriodId") == scoring), {})
            players.append({
                "slot": SLOTS.get(slot, "?"), "starter": slot not in BENCH, "name": p.get("fullName", "?"),
                "team": PRO_TEAMS.get(p.get("proTeamId"), ""), "points": round(pe.get("appliedStatTotal") or 0, 1),
                "projected": round(proj, 1) if proj is not None else None,
                "line": stat_line(p.get("defaultPositionId"), {int(k): v for k, v in actual.items()}),
                "_order": SLOT_ORDER.index(slot) if slot in SLOT_ORDER else 99,
            })
        players.sort(key=lambda x: x.pop("_order"))
        return {
            "id": side.get("teamId"), "name": team.get("name") or team.get("abbrev", "?"), "abbrev": team.get("abbrev", ""),
            "points": round(side.get("totalPointsLive", side.get("totalPoints")) or 0, 1),
            "projected": round(side.get("totalProjectedPointsLive", side.get("totalProjectedPoints")) or 0, 1),
            "win_probability": side.get("winProbability"),
            "players": players,
        }
