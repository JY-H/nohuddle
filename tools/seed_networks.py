"""Seed the network library from folders of stills, one folder per game.

    uv run python tools/seed_networks.py GAME_DIR [GAME_DIR ...] [--dry-run] [--sheet OUT.jpg]

Stills should be spread across a broadcast (different plays, shots and scores), so that only the
overlays stay put. Each folder becomes an overlay map, the network is read off the logo, and the
map is folded into ~/.config/nfl-watcher/networks/ like a game watched live would be.
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from nfl_watcher import gameclock
from nfl_watcher.app import CONFIG_DIR
from nfl_watcher.detector import DetectorSettings, OverlayModel, edges, preprocess
from nfl_watcher.networks import NetworkLibrary, similarity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report, but don't change the library")
    ap.add_argument("--sheet", type=Path, help="write each game's learned overlays drawn on a still")
    args = ap.parse_args()

    settings = DetectorSettings()
    games = []
    for d in args.dirs:
        files = sorted(d.glob("*.jpg"))
        if len(files) < 8:
            print(f"{d.name}: only {len(files)} stills, skipping")
            continue
        frames = [cv2.imread(str(f)) for f in files]
        # If the player ignored the seeks, every still is the same frame and *everything* looks
        # like a permanent overlay. Require the stills to actually differ.
        small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (96, 54)).astype(np.int16) for f in frames]
        spread = np.median([np.abs(a - b).mean() for i, a in enumerate(small) for b in small[i + 1:]])
        if spread < 15:
            print(f"{d.name}: stills barely differ (median difference {spread:.1f}); the seeks probably failed, skipping")
            continue
        persistence = np.mean([edges(preprocess(f)) for f in frames], axis=0)
        votes = Counter()
        for f in frames:
            votes.update(gameclock.network_votes([t.s for t in gameclock.read_text(f)]))
        hint = votes.most_common(1)[0][0] if votes else None
        model = OverlayModel(persistence, settings, NetworkLibrary.edge_threshold)
        print(f"{d.name}: network={hint} votes={dict(votes)} regions={len(model.regions)} boxes={model.boxes()}")
        games.append((d.name, persistence, hint, model, frames[len(frames) // 2]))

    print("\nsimilarity between games:")
    for i, (a, pa, *_rest) in enumerate(games):
        row = " ".join(f"{similarity(pa, pb, NetworkLibrary.edge_threshold):.2f}" for _, pb, *_r in games)
        print(f"  {a[:28]:28s} {row}")

    if args.sheet:
        tiles = []
        for name, _p, hint, model, still in games:
            img = cv2.resize(still, (384, 216))
            for x, y, w, h in model.boxes():
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 220, 0), 1)
            cv2.putText(img, f"{hint} {name[:30]}", (4, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(img)
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.sheet), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))

    if args.dry_run:
        return
    library = NetworkLibrary(CONFIG_DIR / "networks", settings)
    # Games whose network name could be read go first, so unnamed ones join a named entry.
    for name, persistence, hint, _model, _still in sorted(games, key=lambda g: g[2] is None):
        print(f"{name} -> {library.learn(persistence, hint)}")
    print("\nlibrary:", {n.name: n.games for n in library.networks.values()})


if __name__ == "__main__":
    main()
