"""Runs the multiview browser window, watches every stream, and switches the main screen."""

import asyncio
import base64
import hashlib
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web
from playwright.async_api import BrowserContext, Frame, Page, Route, async_playwright

from . import scheduler
from .detector import W, H, BreakDetector, DetectorSettings, State
from .gameclock import ClockSettings
from .health import PlaybackHealth
from .networks import NetworkLibrary
from .streams import Game, StreamSite

log = logging.getLogger("nfl_watcher")

CONFIG_DIR = Path.home() / ".config" / "nfl-watcher"
STATIC_DIR = Path(__file__).parent / "static"

# Poll intervals (seconds). See README for the reasoning.
POLL_ACTIVE = 1.0
POLL_AWAITED = 1.0  # a higher-priority game on a break: we want to jump back the moment it returns
POLL_OTHER = 4.0
RELOADS_BEFORE_MIRROR = 2
MIRROR_COOLDOWN = 120.0  # don't hop mirrors for lag more often than this
NETWORK_LEARN_EVERY = 600.0  # fold a game's proven overlays into its network's entry this often
# Only NFL games share learned overlays: other leagues (college especially) are on different
# networks and graphics packages, so each of their games learns on its own.
LIBRARY_LEAGUES = {"nfl"}

# Runs in every frame (including the stream players) before any page script.
ANTI_POPUP_JS = """
(() => {
  const noop = () => null;
  try { Object.defineProperty(window, 'open', { value: noop, writable: false, configurable: false }); }
  catch (e) { window.open = noop; }
  const stop = (e) => {
    const a = e.target && e.target.closest && e.target.closest('a[target], form[target]');
    if (a && a.target && a.target !== '_self') { e.preventDefault(); e.stopImmediatePropagation(); }
  };
  document.addEventListener('click', stop, true);
  document.addEventListener('submit', stop, true);
})();
"""

# Grab a downscaled frame from the player's <video> and enforce mute state.
GRAB_JS = """
({ width, muted }) => {
  const v = document.querySelector('video');
  if (!v || v.paused) {
    try { const j = window.jwplayer && window.jwplayer(); if (j && j.play) j.play(); } catch (e) {}
    if (v) v.play().catch(() => {});
  }
  if (!v) return { ready: false };
  if (!v.__nw) { v.__nw = { waits: 0 }; v.addEventListener('waiting', () => v.__nw.waits++); }
  v.muted = muted;
  if (!muted && v.volume === 0) v.volume = 1;
  if (!v.videoWidth) return { ready: false, ct: v.currentTime, waits: v.__nw.waits };
  const h = Math.round(width * v.videoHeight / v.videoWidth);
  const c = document.createElement('canvas');
  c.width = width; c.height = h;
  c.getContext('2d').drawImage(v, 0, 0, width, h);
  return { ready: true, ct: v.currentTime, waits: v.__nw.waits, img: c.toDataURL('image/jpeg', 0.8) };
}
"""

SET_MUTED_JS = """(muted) => { const v = document.querySelector('video'); if (v) { v.muted = muted; if (!muted && v.volume === 0) v.volume = 1; } }"""

AD_HOST_PATTERNS = [
    "popads", "popcash", "propellerads", "adsterra", "exoclick", "juicyads", "hilltopads",
    "onclickads", "clickadu", "adcash", "doubleclick", "googlesyndication", "sharethis",
]
EMBED_AD_DOMAIN_RE = re.compile(r'"(?:adserverDomain|cdnDomain)"\s*:\s*"([a-z0-9.-]+)"', re.I)


@dataclass
class Watched:
    game: Game
    detector: BreakDetector
    embed_index: int = 0
    nonce: int = 0  # bumped to make the page reload the player
    last_poll: float = 0.0
    health: PlaybackHealth = field(default_factory=PlaybackHealth)
    reloads: int = 0
    saved_version: int = 0
    network_learned_at: float = 0.0
    clock_saved_at: float = 0.0
    mirror_changed_at: float = 0.0
    laggy: bool = False
    last_frame: np.ndarray | None = None
    # The last ~30s of frames as received (JPEG), so a report can show what just happened.
    recent_jpegs: deque = field(default_factory=lambda: deque(maxlen=15))

    @property
    def embed_url(self) -> str | None:
        e = self.game.embeds
        return e[self.embed_index % len(e)] if e else None


class Controller:
    def __init__(self, site_url: str | None, port: int, league: str | None, mute: bool = False):
        self.mute = mute
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.config_path = CONFIG_DIR / "config.json"
        cfg = json.loads(self.config_path.read_text()) if self.config_path.exists() else {}
        self.site = StreamSite(site_url or cfg.get("site", ""))
        self.port = port
        self.league = league or cfg.get("league", "nfl")
        self.auto = cfg.get("auto", True)
        self.mode = cfg.get("mode", "single")  # "single" or "multi"
        s = cfg.get("settings", {})
        self.settings = DetectorSettings(
            break_after=s.get("break_after", DetectorSettings.break_after),
            live_after=s.get("live_after", DetectorSettings.live_after),
        )
        self.library = NetworkLibrary(CONFIG_DIR / "networks" / "nfl", self.settings)
        self.watched: dict[str, Watched] = {}
        self.priority: list[str] = []
        for g in cfg.get("watchlist", []):
            game = Game(g["id"], g["title"], g["url"], g.get("league", ""), g.get("starts_at", 0), g.get("ends_at", 0), g.get("embeds", []))
            self._add(game, g.get("embed_index", 0))
        self.active: str | None = None
        self.pinned: str | None = None
        self.events: list[dict] = []
        self.page: Page | None = None
        self.ctx: BrowserContext | None = None
        self._frames: dict[str, Frame] = {}
        self._frames_dirty = True
        self._blocked_hosts: set[str] = set()

    # ---------- persistence ----------

    def save(self) -> None:
        cfg = {
            "site": self.site.base_url, "league": self.league, "auto": self.auto, "mode": self.mode,
            "settings": {"break_after": self.settings.break_after, "live_after": self.settings.live_after},
            "watchlist": [{**self.watched[g].game.to_json(), "embed_index": self.watched[g].embed_index} for g in self.priority],
        }
        self.config_path.write_text(json.dumps(cfg, indent=2))

    def _add(self, game: Game, embed_index: int = 0) -> None:
        if game.id in self.watched:
            return
        w = Watched(game, self._new_detector(game, embed_index), embed_index)
        clock_path = self._clock_path(game.id)
        if clock_path.exists():
            try:
                w.detector.clock.load(json.loads(clock_path.read_text()))
            except Exception as e:
                log.warning("Ignoring saved clock for %s: %s", game.title, e)
        self.watched[game.id] = w
        self.priority.append(game.id)

    def _new_detector(self, game: Game, embed_index: int) -> BreakDetector:
        clock = ClockSettings(
            kickoff=float(game.starts_at),
            # NFL halftime is ~13 minutes; college is usually 20.
            min_halftime=(12 if game.league == "nfl" else 18) * 60,
        )
        shares = game.league in LIBRARY_LEAGUES
        d = BreakDetector(self.settings, clock, networks=self.library.candidates if shares else None)
        path = self._model_path(game.id, embed_index)
        if path.exists():
            try:
                d.load(np.load(path), time.time())
                log.info("Loaded learned overlays for %s", game.title)
            except Exception as e:
                log.warning("Ignoring saved overlays for %s: %s", game.title, e)
        return d

    def _clock_path(self, gid: str) -> Path:
        return CONFIG_DIR / "models" / f"{hashlib.sha1(gid.encode()).hexdigest()[:16]}.clock.json"

    def _model_path(self, gid: str, embed_index: int) -> Path:
        """Learned overlays are kept per game *and* source: mirrors can be scaled or cropped differently."""
        key = hashlib.sha1(f"{gid}#{embed_index}".encode()).hexdigest()[:16]
        return CONFIG_DIR / "models" / f"{key}.npy"

    def _save_model(self, w: Watched) -> None:
        d = w.detector
        if time.time() - w.clock_saved_at > 15:
            path = self._clock_path(w.game.id)
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(d.clock.to_json()))
            w.clock_saved_at = time.time()
        # Only keep overlays that have proven themselves over a minute of live play.
        proven = d.state == State.LIVE and time.time() - d.since >= 60
        if d.model is None or not d.model.regions or not proven or d.model_version == w.saved_version:
            return
        path = self._model_path(w.game.id, w.embed_index)
        path.parent.mkdir(exist_ok=True)
        np.save(path, d.model.persistence.astype(np.float16))
        w.saved_version = d.model_version
        if w.game.league in LIBRARY_LEAGUES and time.time() - w.network_learned_at >= NETWORK_LEARN_EVERY:
            hint = d.network_votes.most_common(1)[0][0] if d.network_votes else None
            d.network = self.library.learn(d.model.persistence, hint)
            w.network_learned_at = time.time()
            log.info("%s looks like %s", w.game.title, d.network)

    # ---------- browser ----------

    async def run(self) -> None:
        runner = web.AppRunner(self._web_app(), access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", self.port).start()
        url = f"http://127.0.0.1:{self.port}/"
        log.info("Control panel at %s", url)

        async with async_playwright() as pw:
            self.ctx = await pw.chromium.launch_persistent_context(
                str(CONFIG_DIR / "chrome-profile"),
                channel="chrome",
                headless=False,
                no_viewport=True,
                service_workers="block",
                args=["--autoplay-policy=no-user-gesture-required", "--window-size=1600,1000",
                      "--disable-features=BackgroundVideoTrackOptimization"],
                ignore_default_args=[] if self.mute else ["--mute-audio"],
            )
            await self._install_popup_defenses(self.ctx)
            self.page = self.ctx.pages[0] if self.ctx.pages else await self.ctx.new_page()
            for p in self.ctx.pages[1:]:
                await p.close()
            for ev in ("frameattached", "framedetached", "framenavigated"):
                self.page.on(ev, lambda *_: setattr(self, "_frames_dirty", True))
            self.page.on("framenavigated", self._guard_main_frame)
            await self.page.goto(url)
            # Refresh sources (mirrors come and go) and titles for the saved watchlist.
            await asyncio.gather(*(self._resolve(w) for w in self.watched.values()))
            closed = asyncio.Event()
            self.page.on("close", lambda *_: closed.set())
            loop = asyncio.create_task(self._watch_loop())
            await closed.wait()
            loop.cancel()
        await self.site.close()
        await runner.cleanup()

    async def _install_popup_defenses(self, ctx: BrowserContext) -> None:
        await ctx.add_init_script(ANTI_POPUP_JS)

        def on_page(p: Page) -> None:
            if p is not self.page:
                log.info("Blocked popup: %s", p.url)
                asyncio.ensure_future(p.close())
        ctx.on("page", on_page)

        await ctx.route(re.compile("|".join(AD_HOST_PATTERNS)), lambda r: r.abort())
        # Player pages declare their ad servers in a config blob; read it and block them.
        await ctx.route(re.compile(r"/embed/"), self._inspect_embed)

    async def _inspect_embed(self, route: Route) -> None:
        if route.request.resource_type != "document":
            await route.fallback()
            return
        try:
            resp = await route.fetch()
            body = await resp.text()
        except Exception:
            await route.fallback()
            return
        for host in set(EMBED_AD_DOMAIN_RE.findall(body)) - self._blocked_hosts:
            self._blocked_hosts.add(host)
            log.info("Blocking embed ad host %s", host)
            await self.ctx.route(re.compile(rf"^https?://([^/]*\.)?{re.escape(host)}/"), lambda r: r.abort())
        await route.fulfill(response=resp, body=body)

    def _guard_main_frame(self, frame: Frame) -> None:
        if frame is self.page.main_frame and not frame.url.startswith(f"http://127.0.0.1:{self.port}"):
            log.info("Blocked navigation to %s", frame.url)
            asyncio.ensure_future(self.page.goto(f"http://127.0.0.1:{self.port}/"))

    async def _game_frames(self) -> dict[str, Frame]:
        if not self._frames_dirty:
            return self._frames
        frames = {}
        for f in self.page.main_frame.child_frames:
            try:
                el = await f.frame_element()
                gid = await el.get_attribute("data-game")
            except Exception:
                continue
            if gid:
                frames[gid] = f
        self._frames, self._frames_dirty = frames, False
        return frames

    # ---------- watching ----------

    async def _resolve(self, w: Watched) -> None:
        try:
            await self.site.resolve(w.game)
            self.save()
        except Exception as e:
            log.warning("Couldn't resolve %s: %s", w.game.url, e)

    def _poll_interval(self, gid: str) -> float:
        if gid == self.active:
            return POLL_ACTIVE
        if self.active in self.priority and gid in self.priority and \
                self.priority.index(gid) < self.priority.index(self.active) and \
                self.watched[gid].detector.state == State.BREAK:
            return POLL_AWAITED
        return POLL_OTHER

    async def _watch_loop(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("watch loop error")
            await asyncio.sleep(0.25)

    async def _tick(self) -> None:
        frames = await self._game_frames()
        now = time.time()
        due = [g for g in self.priority if now - self.watched[g].last_poll >= self._poll_interval(g)]
        await asyncio.gather(*(self._poll(g, frames.get(g), now) for g in due))
        self._reschedule()

    async def _poll(self, gid: str, frame: Frame | None, now: float) -> None:
        w = self.watched.get(gid)
        if w is None:
            return
        w.last_poll = now
        res = None
        if frame is not None and not frame.is_detached():
            try:
                res = await frame.evaluate(GRAB_JS, {"width": 640, "muted": gid != self.active})
            except Exception:
                self._frames_dirty = True
        res = res or {}
        w.health.update(now, res.get("ct"), res.get("waits", 0))
        if w.health.stalled(now):
            w.detector.mark_stalled(now)
            self._recover(w, now)
            return
        w.reloads = 0
        w.laggy = w.health.laggy()
        if w.laggy and len(w.game.embeds) > 1 and now - w.mirror_changed_at > MIRROR_COOLDOWN:
            self._switch_mirror(w, (w.embed_index + 1) % len(w.game.embeds), now, "laggy")
            return
        if not res.get("ready"):
            return
        jpeg = base64.b64decode(res["img"].split(",", 1)[1])
        w.recent_jpegs.append((now, jpeg))
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        await asyncio.to_thread(w.detector.observe, img, now)
        w.last_frame = cv2.resize(img, (W, H))
        self._save_model(w)

    def _recover(self, w: Watched, now: float) -> None:
        """Reload a stuck player; after a couple of failed reloads, try the next mirror."""
        if not w.game.embeds:
            return
        w.reloads += 1
        if w.reloads > RELOADS_BEFORE_MIRROR and len(w.game.embeds) > 1:
            self._switch_mirror(w, (w.embed_index + 1) % len(w.game.embeds), now, "not playing")
            return
        w.nonce += 1
        w.health.reset(now)  # give the reload time to start

    def _switch_mirror(self, w: Watched, index: int, now: float, why: str) -> None:
        w.embed_index = index
        w.nonce += 1
        w.reloads = 0
        w.laggy = False
        w.mirror_changed_at = now
        w.health.reset(now)
        # A different source may be cropped or scaled differently; use what was learned for it.
        # The game clock is the same whichever source it comes from.
        clock = w.detector.clock
        w.detector = self._new_detector(w.game, index)
        w.detector.clock = clock
        w.saved_version = 0
        self._event(f"{w.game.title}: {why}, switched to source {index + 1} of {len(w.game.embeds)}")
        self.save()

    def _reschedule(self) -> None:
        if not self.priority:
            self.active = None
            return
        states = {g: self.watched[g].detector.state for g in self.priority}
        if self.pinned not in self.priority:
            self.pinned = None
        if not self.auto and self.active in self.priority:
            return
        new = scheduler.choose(self.priority, self._showable(), self.pinned)
        if new != self.active:
            old = self.active
            self.active = new
            if new is None:
                self._event("All games on break")
            elif old in self.watched:
                why = {State.BREAK: "on a break", State.STALLED: "stalled"}.get(states[old])
                why = f"{self.watched[old].game.title} {why}" if why else f"{self.watched[new].game.title} is back"
                self._event(f"→ {self.watched[new].game.title} ({why})")
            asyncio.ensure_future(self._apply_audio())

    async def _apply_audio(self) -> None:
        frames = await self._game_frames()
        for gid, f in frames.items():
            try:
                await f.evaluate(SET_MUTED_JS, gid != self.active)
            except Exception:
                pass

    def _event(self, msg: str) -> None:
        log.info(msg)
        self.events = (self.events + [{"t": time.time(), "msg": msg}])[-20:]

    # ---------- HTTP API ----------

    def _showable(self) -> set[str]:
        now = time.time()
        return {g for g in self.priority if self.watched[g].detector.showable(now)}

    def _visible(self) -> list[str]:
        """Games on screen, in display order: the active game first. Breaks are never shown."""
        if self.active is None:
            return []
        if self.mode == "single":
            return [self.active]
        showable = self._showable()
        rest = [g for g in self.priority if g != self.active and g in showable]
        return [self.active] + rest

    def _state_json(self) -> dict:
        now = time.time()
        return {
            "now": now, "active": self.active, "pinned": self.pinned, "auto": self.auto, "mode": self.mode,
            "visible": self._visible(),
            "site": self.site.base_url, "league": self.league, "events": self.events,
            "settings": {"break_after": self.settings.break_after, "live_after": self.settings.live_after},
            "games": [
                {
                    **self.watched[g].game.to_json(),
                    "state": self.watched[g].detector.state, "since": self.watched[g].detector.since,
                    "detail": self.watched[g].detector.detail,
                    "clock": self.watched[g].detector.clock.display,
                    "network": self.watched[g].detector.network,
                    "overridden": self.watched[g].detector.override is not None,
                    "embed_index": self.watched[g].embed_index, "embed_url": self.watched[g].embed_url,
                    "nonce": self.watched[g].nonce, "laggy": self.watched[g].laggy,
                    "play_ratio": self.watched[g].health.play_ratio(),
                    "regions": len(self.watched[g].detector.model.regions) if self.watched[g].detector.model else 0,
                }
                for g in self.priority
            ],
        }

    def _web_app(self) -> web.Application:
        app = web.Application()
        r = app.router
        r.add_get("/", lambda _: web.FileResponse(STATIC_DIR / "index.html"))
        r.add_static("/static", STATIC_DIR)
        r.add_get("/api/state", lambda _: web.json_response(self._state_json()))
        r.add_get("/api/games", self._api_games)
        r.add_post("/api/watch", self._api_watch)
        r.add_post("/api/unwatch", self._api_unwatch)
        r.add_post("/api/order", self._api_order)
        r.add_post("/api/pin", self._api_pin)
        r.add_post("/api/auto", self._api_auto)
        r.add_post("/api/override", self._api_override)
        r.add_post("/api/mirror", self._api_mirror)
        r.add_post("/api/settings", self._api_settings)
        r.add_get("/api/debug/{gid:.+}.png", self._api_debug)
        r.add_get("/api/debug/{gid:.+}.json", self._api_debug_json)
        r.add_post("/api/report", self._api_report)
        return app

    async def _api_games(self, req: web.Request) -> web.Response:
        league = req.query.get("league", self.league)
        if not self.site.base_url:
            return web.json_response({"error": "Set your stream site under Settings (or run with --site)."}, status=400)
        try:
            games = await self.site.list_games(league)
        except Exception as e:
            return web.json_response({"error": f"Couldn't load {self.site.base_url}/{league}: {e}"}, status=502)
        self.league = league
        return web.json_response({"games": [g.to_json() for g in games]})

    async def _api_watch(self, req: web.Request) -> web.Response:
        body = await req.json()
        url = body["url"].strip()
        game = self.site.game_from_url(url)
        game.title = body.get("title") or game.title
        game.starts_at, game.ends_at = body.get("starts_at", 0), body.get("ends_at", 0)
        try:
            await self.site.resolve(game)
        except Exception as e:
            return web.json_response({"error": f"Couldn't load {url}: {e}"}, status=502)
        if not game.embeds:
            return web.json_response({"error": "No player found on that page (stream may not have started)."}, status=422)
        self._add(game)
        self.save()
        return web.json_response({"ok": True})

    async def _api_unwatch(self, req: web.Request) -> web.Response:
        gid = (await req.json())["id"]
        self.watched.pop(gid, None)
        self.priority = [g for g in self.priority if g != gid]
        self.save()
        return web.json_response({"ok": True})

    async def _api_order(self, req: web.Request) -> web.Response:
        order = (await req.json())["ids"]
        self.priority = [g for g in order if g in self.watched] + [g for g in self.priority if g not in order]
        self.save()
        return web.json_response({"ok": True})

    async def _api_pin(self, req: web.Request) -> web.Response:
        self.pinned = (await req.json()).get("id")
        if self.pinned:
            self.active = self.pinned
            await self._apply_audio()
        self._reschedule()
        return web.json_response({"ok": True})

    async def _api_auto(self, req: web.Request) -> web.Response:
        body = await req.json()
        if "auto" in body:
            self.auto = bool(body["auto"])
        if body.get("mode") in ("single", "multi"):
            self.mode = body["mode"]
        self.save()
        return web.json_response({"ok": True})

    async def _api_override(self, req: web.Request) -> web.Response:
        body = await req.json()
        w = self.watched.get(body["id"])
        if w:
            w.detector.force(State(body["state"]), time.time())
            self._reschedule()
        return web.json_response({"ok": True})

    async def _api_mirror(self, req: web.Request) -> web.Response:
        body = await req.json()
        w = self.watched.get(body["id"])
        if w and w.game.embeds:
            self._switch_mirror(w, int(body["index"]) % len(w.game.embeds), time.time(), "chosen manually")
        return web.json_response({"ok": True})

    async def _api_settings(self, req: web.Request) -> web.Response:
        body = await req.json()
        for key in ("break_after", "live_after"):
            if key in body:
                setattr(self.settings, key, max(0.5, float(body[key])))
        if body.get("site"):
            self.site.base_url = body["site"].rstrip("/")
        self.save()
        return web.json_response({"ok": True})

    def _debug_info(self, w: Watched) -> dict:
        c = w.detector.clock
        return {"id": w.game.id, "title": w.game.title, "network": w.detector.network,
                "network_votes": dict(w.detector.network_votes),
                "state": w.detector.state, "detail": w.detector.detail,
                "since": w.detector.since, "clock_display": c.display, "clock": c.to_json(), "teams": c.teams,
                "readings": list(c.log), "scores": w.detector.last_scores,
                "regions": w.detector.model.boxes() if w.detector.model else [],
                "relearned": w.detector.relearned, "source": w.embed_index, "laggy": w.laggy,
                "play_ratio": w.health.play_ratio()}

    def _debug_png(self, w: Watched) -> bytes | None:
        """The latest frame with the learned overlay regions drawn on it."""
        if w.last_frame is None:
            return None
        img = w.last_frame.copy()
        model = w.detector.model
        if model:
            for (x, y, bw, bh), sc in zip(model.boxes(), w.detector.last_scores):
                color = (0, 220, 0) if sc >= self.settings.region_match else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 1)
                cv2.putText(img, f"{sc:.2f}", (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        return cv2.imencode(".png", img)[1].tobytes()

    async def _api_debug_json(self, req: web.Request) -> web.Response:
        w = self.watched.get(req.match_info["gid"])
        if w is None:
            raise web.HTTPNotFound()
        return web.json_response(self._debug_info(w))

    async def _api_debug(self, req: web.Request) -> web.Response:
        w = self.watched.get(req.match_info["gid"])
        png = self._debug_png(w) if w else None
        if png is None:
            raise web.HTTPNotFound()
        return web.Response(body=png, content_type="image/png", headers={"Cache-Control": "no-store"})

    async def _api_report(self, req: web.Request) -> web.Response:
        """Save everything about a game (and the overall state) for later diagnosis."""
        body = await req.json()
        w = self.watched.get(body.get("id"))
        if w is None:
            raise web.HTTPNotFound()
        slug = w.game.id.rsplit("/", 1)[-1]
        folder = CONFIG_DIR / "reports" / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
        (folder / "frames").mkdir(parents=True)
        report = {"note": body.get("note", ""), "saved_at": time.time(), "game": self._debug_info(w),
                  "app": self._state_json()}
        (folder / "report.json").write_text(json.dumps(report, indent=2, default=str))
        if png := self._debug_png(w):
            (folder / "detector.png").write_bytes(png)
        for t, jpeg in list(w.recent_jpegs):
            (folder / "frames" / f"{t:.1f}.jpg").write_bytes(jpeg)
        log.info("Saved report to %s", folder)
        return web.json_response({"ok": True, "path": str(folder)})
