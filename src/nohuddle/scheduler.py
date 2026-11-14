"""Decide which game should be on the main screen (and have the audio)."""

from .detector import State

# States a game can ever be on screen in. A LEARNING game additionally has to qualify through
# BreakDetector.showable() - normally only once its scorebug has confirmed it's live.
WATCHABLE = (State.LIVE, State.LEARNING)


def choose(priority: list[str], showable: set[str], pinned: str | None) -> str | None:
    """Pick the game to show.

    - A pinned game always wins (manual control).
    - Otherwise the highest-priority game that can be shown (confirmed live, or a game we can't
      confirm at all after a grace period).
    - If nothing qualifies, show nothing: commercials are never put on screen.
    """
    if pinned in priority:
        return pinned
    for gid in priority:
        if gid in showable:
            return gid
    return None
