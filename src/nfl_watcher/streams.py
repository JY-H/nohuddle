"""Discover games on the configured stream site and resolve their embeddable player URLs."""

import base64
import html
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import aiohttp

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"

CARD_RE = re.compile(
    r'<article class="stream-card.*?<a href="([^"]+)".*?<h3 class="card-title">([^<]+)</h3>'
    r'.*?data-starts="(\d+)" data-ends="(\d+)".*?</article>',
    re.S,
)
ENCODED_RE = re.compile(r'data-e="([A-Za-z0-9+/=]+)"')
BUTTON_RE = re.compile(r'<button[^>]*class="[^"]*stream-btn[^"]*"[^>]*data-e="([A-Za-z0-9+/=]+)"[^>]*>(.*?)</button>', re.S)


@dataclass
class Game:
    id: str  # slug-ish key, e.g. "cfb/2026-11-14/miami-fl-vs-duke"
    title: str
    url: str
    league: str
    starts_at: int = 0
    ends_at: int = 0
    embeds: list[str] = field(default_factory=list)

    @property
    def is_live(self) -> bool:
        now = time.time()
        return self.starts_at <= now <= self.ends_at + 3600

    def to_json(self) -> dict:
        return {
            "id": self.id, "title": self.title, "url": self.url, "league": self.league,
            "starts_at": self.starts_at, "ends_at": self.ends_at, "is_live": self.is_live,
            "embeds": self.embeds,
        }


def game_id_from_url(url: str) -> str:
    path = urlparse(url).path
    return path.removeprefix("/live/").strip("/")


def deobfuscate(encoded: str) -> str | None:
    """The site stores embed URLs base64-encoded and XORed with a single byte.

    Rather than hard-coding the key, derive it from the fact that every URL starts with "http".
    """
    raw = base64.b64decode(encoded)
    if not raw:
        return None
    key = raw[0] ^ ord("h")
    url = bytes(b ^ key for b in raw).decode("utf-8", "replace")
    return url if url.startswith("http") else None


class StreamSite:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get(self, url: str) -> str:
        if self._session is None:
            self._session = aiohttp.ClientSession(headers={"User-Agent": UA}, timeout=aiohttp.ClientTimeout(total=20))
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.text()

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    async def list_games(self, league: str) -> list[Game]:
        page = await self._get(f"{self.base_url}/{league}")
        games: dict[str, Game] = {}
        for href, title, starts, ends in CARD_RE.findall(page):
            url = href if href.startswith("http") else self.base_url + href
            gid = game_id_from_url(url)
            games.setdefault(gid, Game(gid, html.unescape(title).strip(), url, league, int(starts), int(ends)))
        return sorted(games.values(), key=lambda g: (not g.is_live, g.starts_at, g.title))

    async def resolve(self, game: Game) -> Game:
        """Fill in the game's embeddable player URLs (one per mirror the site offers)."""
        page = await self._get(game.url)
        embeds = [deobfuscate(e) for e, _label in BUTTON_RE.findall(page)]
        if not embeds:
            embeds = [deobfuscate(e) for e in ENCODED_RE.findall(page)]
        game.embeds = list(dict.fromkeys(u for u in embeds if u))
        m = re.search(r"<h1[^>]*>([^<]+)", page)
        if m and m.group(1).strip():
            game.title = html.unescape(m.group(1)).strip()
        return game

    def game_from_url(self, url: str) -> Game:
        gid = game_id_from_url(url)
        league = gid.split("/", 1)[0]
        slug = gid.rsplit("/", 1)[-1]
        return Game(gid, slug.replace("-", " ").title(), url, league)
