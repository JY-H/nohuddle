"""A library of what each network's broadcast overlays look like, shared across games.

A game's own overlays take a minute or so to learn (and can't be learned at all if you tune in
during halftime or an ad break). But every FOX game has the same FOX logo and scorebug frame, so
overlays learned from one game make a good starting point for the next:

- When a new game starts, each known network's overlays are checked against its frames; once one
  shows up steadily, the game starts from that network's model instead of from nothing.
- When a game's own model has proven itself, it's folded back into its network's entry (or starts
  a new entry). Averaging across games keeps what's common to the network - logo, scorebug frame -
  and fades what's specific to one game, like team names.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectorSettings, OverlayModel
from .gameclock import network_votes  # noqa: F401 (re-exported for callers)

def similarity(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    """How much two overlay maps' strong edges coincide (Dice overlap, with 1px slack)."""
    ma = (a >= threshold).astype(np.uint8)
    mb = (b >= threshold).astype(np.uint8)
    total = int(ma.sum() + mb.sum())
    if total == 0:
        return 0.0
    k = np.ones((3, 3), np.uint8)
    hits = int((ma & cv2.dilate(mb, k)).sum() + (mb & cv2.dilate(ma, k)).sum())
    return hits / total


@dataclass
class Network:
    name: str
    persistence: np.ndarray
    games: int  # how many games have been folded in
    model: OverlayModel


class NetworkLibrary:
    match_threshold = 0.65  # same network measured 0.74-0.86 on NFL broadcasts; FOX vs CBS 0.46-0.55
    # Network maps are averages over many games (or sparse stills), so a lower bar than a single
    # game's live history still only keeps overlays.
    edge_threshold = 0.5
    max_weight = 20  # older games' share of the average stops growing past this many

    def __init__(self, folder: Path, settings: DetectorSettings):
        self.folder = folder
        self.s = settings
        self.networks: dict[str, Network] = {}
        if folder.exists():
            for meta_path in folder.glob("*.json"):
                try:
                    meta = json.loads(meta_path.read_text())
                    persistence = np.load(meta_path.with_suffix(".npy")).astype(np.float32)
                except Exception:
                    continue
                self._put(meta["name"], persistence, meta.get("games", 1))

    def _put(self, name: str, persistence: np.ndarray, games: int) -> Network:
        net = Network(name, persistence, games, OverlayModel(persistence, self.s, self.edge_threshold))
        self.networks[name] = net
        return net

    def _stem(self, net: Network) -> str:
        return re.sub(r"[^a-z0-9]+", "-", net.name.lower()).strip("-")

    def _delete(self, net: Network) -> None:
        for ext in (".npy", ".json"):
            (self.folder / f"{self._stem(net)}{ext}").unlink(missing_ok=True)

    def _save(self, net: Network) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        stem = self._stem(net)
        np.save(self.folder / f"{stem}.npy", net.persistence.astype(np.float16))
        (self.folder / f"{stem}.json").write_text(json.dumps({"name": net.name, "games": net.games}))

    def candidates(self) -> list[tuple[str, OverlayModel]]:
        return [(n.name, n.model) for n in self.networks.values() if n.model.regions]

    def match(self, persistence: np.ndarray) -> tuple[str | None, float]:
        best, best_sim = None, 0.0
        for net in self.networks.values():
            sim = similarity(net.persistence, persistence, self.edge_threshold)
            if sim > best_sim:
                best, best_sim = net.name, sim
        return (best, best_sim) if best_sim >= self.match_threshold else (None, best_sim)

    def learn(self, persistence: np.ndarray, hint: str | None = None) -> str:
        """Fold a proven game model into its network's entry (or start a new one)."""
        name, _ = self.match(persistence)
        if name is None:
            # A new look. Same network as an existing entry (e.g. its college vs. NFL scorebug)
            # gets its own entry rather than being blurred together.
            base = hint or "Network"
            name = base if base not in self.networks and hint else \
                next(f"{base} ({i})" for i in range(2, 100) if f"{base} ({i})" not in self.networks)
            net = self._put(name, persistence.astype(np.float32), 1)
        else:
            net = self.networks[name]
            w = min(net.games, self.max_weight)
            merged = (net.persistence * w + persistence) / (w + 1)
            if hint and name.startswith("Network") and hint not in self.networks:
                # First time we've read this look's network name off the screen: use it.
                self._delete(net)
                del self.networks[name]
                name = hint
            net = self._put(name, merged, net.games + 1)
        self._save(net)
        return name
