import argparse

import uvicorn

from battleship_server.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Run a single-process local Battleship server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--turn-timeout", type=float, default=None, metavar="SECONDS")
    args = parser.parse_args()
    try:
        app = create_app(turn_timeout=args.turn_timeout)
    except ValueError as error:
        parser.error(str(error))
    # Access logs include paths, not bearer credentials, invitation bodies, or placements.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
