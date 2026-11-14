"""Learn overlays from the first N seconds of a recording, then draw them on chosen frames.

    uv run python tools/regions.py RECORDING_DIR LEARN_UNTIL_S OUT.jpg T1 T2 ...
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from nfl_watcher.detector import BreakDetector, preprocess

d, until, out, times = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], [int(x) for x in sys.argv[4:]]
det = BreakDetector()
for f in sorted(d.glob("*.jpg")):
    if float(f.stem) > until:
        break
    det.observe(cv2.imread(str(f)), float(f.stem))
m = det.model
tiles = []
for t in times:
    img = cv2.resize(cv2.imread(str(d / f"{t:05d}.jpg")), (768, 432))
    scores = m.scores(preprocess(cv2.imread(str(d / f"{t:05d}.jpg"))))
    for (x, y, w, h), sc in zip(m.boxes(), scores):
        c = (0, 220, 0) if sc >= det.s.region_match else (0, 0, 255)
        cv2.rectangle(img, (x * 2, y * 2), ((x + w) * 2, (y + h) * 2), c, 1)
        cv2.putText(img, f"{sc:.2f}", (x * 2, max(10, y * 2 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    cv2.putText(img, f"t={t}", (4, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tiles.append(img)
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
