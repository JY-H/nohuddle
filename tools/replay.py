"""Replay a folder of recorded frames (NNNNN.jpg = seconds) through the detector.

    uv run python tools/replay.py RECORDING_DIR [--sheet OUT.jpg]

Prints a timeline of state changes and optionally writes a contact sheet (one thumbnail every
N seconds, bordered green=live, orange=break, blue=learning) for eyeballing mistakes.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from nohuddle.detector import BreakDetector, State

COLORS = {State.LIVE: (80, 200, 60), State.BREAK: (0, 160, 255), State.LEARNING: (255, 140, 60), State.STALLED: (60, 60, 230)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--sheet", type=Path)
    ap.add_argument("--every", type=int, default=5, help="seconds between contact-sheet thumbnails")
    args = ap.parse_args()

    det = BreakDetector()
    files = sorted(args.dir.glob("*.jpg"))
    prev, thumbs = None, []
    for f in files:
        t = float(f.stem)
        img = cv2.imread(str(f))
        st = det.observe(img, t)
        if st != prev:
            print(f"{int(t) // 60:02d}:{int(t) % 60:02d}  {st.value:9s} regions={len(det.model.regions) if det.model else 0}")
            prev = st
        if args.sheet and int(t) % args.every == 0:
            th = cv2.resize(img, (192, 108))
            raw = any(sc >= det.s.region_match for sc in det.last_scores) if det.model else None
            th = cv2.copyMakeBorder(th, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=COLORS[st])
            cv2.putText(th, f"{int(t)//60}:{int(t)%60:02d} {'+' if raw else '-' if raw is not None else '?'}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            thumbs.append(th)
    if args.sheet and thumbs:
        cols = 10
        while len(thumbs) % cols:
            thumbs.append(np.zeros_like(thumbs[0]))
        rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
        cv2.imwrite(str(args.sheet), np.vstack(rows))


if __name__ == "__main__":
    main()
