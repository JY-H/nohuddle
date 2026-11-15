"""Run the full NFL setup (network library, game clock, sport plug-ins) over recorded windows.

    uv run python tools/replay_nfl.py WINDOW_DIR [WINDOW_DIR ...] [--league nfl] [--sheet OUT.jpg]

Each WINDOW_DIR holds NNNNN.jpg frames one second apart. Prints every state change, so in an
ad-free replay every "break" outside halftime is a false switch worth looking at.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from nohuddle.app import CONFIG_DIR
from nohuddle.detector import BreakDetector, DetectorSettings, State
from nohuddle.gameclock import ClockSettings
from nohuddle.networks import NetworkLibrary
from nohuddle.sports import sport_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--league", default="nfl")
    ap.add_argument("--sheet", type=Path, help="stills at each state change")
    args = ap.parse_args()
    settings = DetectorSettings()
    library = NetworkLibrary(CONFIG_DIR / "networks" / "nfl", settings)
    sport = sport_for(args.league)
    tiles = []
    for d in args.dirs:
        clock = ClockSettings(period_seconds=sport.period_seconds, periods_per_half=sport.periods_per_half,
                              min_halftime=sport.min_halftime)
        det = BreakDetector(settings, clock, networks=library.candidates if args.league == "nfl" else None,
                            extras=sport.make_extras())
        prev, changes, counts = None, [], {}
        for f in sorted(d.glob("*.jpg")):
            t = float(f.stem)
            img = cv2.imread(str(f))
            st = det.observe(img, t)
            counts[st.value] = counts.get(st.value, 0) + 1
            if st != prev:
                label = f"{int(t)//60}:{int(t)%60:02d} {st.value} {det.detail if st != State.LIVE else ''} {det.clock.display}"
                changes.append(label)
                if args.sheet:
                    th = cv2.resize(img, (288, 162))
                    cv2.putText(th, f"{d.parent.name[:10]}/{d.name} {label}"[:44], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    tiles.append(th)
            prev = st
        print(f"== {d.parent.name}/{d.name}: network={det.network} seconds={counts}")
        for c in changes:
            print("   ", c)
    if args.sheet and tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.sheet), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


if __name__ == "__main__":
    main()
