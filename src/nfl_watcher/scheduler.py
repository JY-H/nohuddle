"""Decide which game should be on the main screen (and have the audio)."""

from .detector import State

WATCHABLE = (State.LIVE, State.LEARNING)


def choose(priority: list[str], states: dict[str, State], pinned: str | None) -> str | None:
    """Pick the game to show.

    - A pinned game always wins (manual control).
    - Otherwise the highest-priority game that isn't on a break or stalled.
    - If nothing is watchable, show nothing: commercials are never put on screen.
    """
    if pinned in priority:
        return pinned
    for gid in priority:
        if states.get(gid, State.LEARNING) in WATCHABLE:
            return gid
    return None
