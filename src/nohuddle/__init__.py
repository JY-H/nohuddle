import argparse
import asyncio
import logging


def main() -> None:
    from .app import Controller

    parser = argparse.ArgumentParser(description="Watch several games at once and skip the commercials.")
    parser.add_argument("--site", default=None, help="stream site base URL (default: last used)")
    parser.add_argument("--league", default=None, help="league listed in the control panel (nfl, cfb, ...; default: last used)")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--mute", action="store_true", help="mute the browser entirely (for testing)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        asyncio.run(Controller(args.site, args.port, args.league, mute=args.mute).run())
    except KeyboardInterrupt:
        pass
