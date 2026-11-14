# nfl-watcher

Watch several games at once and never sit through a commercial. Pick your games, put them in priority order, and the app shows the highest-priority game that isn't on a break, switching back to your main game the moment it returns.

```
uv run nfl-watcher            # opens a Chrome window with the multiview + control panel
uv run nfl-watcher --league cfb --site https://<stream site>   # remembered after the first run
```

Needs [uv](https://docs.astral.sh/uv/) and Google Chrome (the app drives your installed Chrome, so there's no separate browser download).

## Using it

- **Add games** from the list (pick a league) or paste a game page URL.
- **Drag** the watchlist to set priority. #1 is your main game.
- **Single** shows one game; **Multiview** shows every game that's live, with the top one big and unmuted. Games on a break are never shown; if everything's on a break you get a "all games on break" screen.
- **Watch now** / click a tile / keys `1`–`9` pins a game (manual control). **Resume auto** or `A` unpins.
- **It's a break** / **Not a break** fix a wrong call. "Not a break" on a detected break makes it relearn that game.
- **Source** picks a mirror; the app also switches mirrors itself when a stream is laggy or dead.
- 👁 shows what the detector sees (green = overlay visible, red = learned but not visible).
- Keys: `S` panel, `M` single/multi, `F` fullscreen.

Settings, the watchlist, learned overlays, the network library, saved reports and the app's Chrome profile live in `data/` in the checkout (gitignored; it holds your stream site and broadcast frames, so keep it out of git). Set `NFL_WATCHER_DATA` to use another folder; installed elsewhere, the default is `~/.config/nfl-watcher/`.

## How break detection works

Live broadcasts have graphics composited over the picture: the scorebug, the network logo, down-and-distance boxes. Commercials don't. Every network's graphics look different and any one can disappear for a while (ESPN hides the scorebug during some plays; stat graphics and replays cover it), so rather than hard-coding templates, each game learns its own:

1. Every 4s a frame goes into a ~10 minute history (break frames excluded, so long ad breaks can't wash the model out).
2. Pixels that are an **edge in ≥70% of frames** are overlay outlines — even translucent logos keep their edges, while moving scenery doesn't.
3. Those pixels are grouped into regions; each frame, each region's edge pattern is **correlated** with the current edges (busy ad scenes don't correlate with a logo's shape).
4. **Any** region matching = broadcast. All gone (scorebug *and* logo) for 4s = **break**; back for 2s = **live**.
5. The largest region is the scorebug. If only the logo has been there for 2+ minutes, it's an **intermission** (halftime show) and counts as a break. Minutes with no overlays at all also count as a break.

Learned overlays are saved per game and source, so restarting mid-game (or mid-halftime) doesn't start from scratch.

### Game clock (macOS)

Edge matching knows *a* scorebug is on screen but not *when* it's from — halftime shows replay first-quarter plays with the old scorebug burned in. So every ~2s the scorebug is read with macOS's built-in text recognition (Vision; local, ~50ms) to get the quarter and clock:

- **Highlights:** a scorebug more than 2½ minutes of game clock behind the furthest point seen is a highlight, not live play. A replay of the play that just happened is within that.
- **Halftime:** starts at `2ND 0:00`, `HALFTIME`, or when the scorebug vanishes for minutes after the 2nd quarter; it only ends when the scorebug reads the 3rd quarter.
- **Other games:** a scorebug far *ahead* of this game, without this game's team names, is another game's highlight (NFL broadcasts show these).
- Readings only move the "furthest point" when two in a row agree, so OCR slips don't derail it.
- **Fallback** when the clock can't be read: halftime (only possible 60–130 min after kickoff) lasts at least 12 min for NFL / 18 min for college, plus 30s of scorebug before it counts as live again.

Polling: the main game and a higher-priority game you're waiting on every 1s, the rest every 4s.

Stream health: no video progress for 15s = stalled (reload, then next mirror); less than 0.85s of video per second or 3+ rebuffers in a minute = laggy (next mirror, at most every 2 minutes). Games with a single source are just marked laggy.

## Popups

Stream iframes never receive clicks (`pointer-events: none`), `window.open` is disabled in every frame, any new tab/window is closed immediately, navigation away from the app is reverted, and the ad servers each player declares in its config are blocked.

## Tools

- `tools/replay.py DIR --sheet out.jpg` — run the detector over recorded frames (`NNNNN.jpg`, named by second) and make a colour-coded contact sheet.
- `tools/regions.py DIR LEARN_UNTIL OUT.jpg T...` — draw learned regions and scores on chosen frames.
- `uv run pytest`
