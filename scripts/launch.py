"""Local convenience launcher. Clients still interact only through the HTTP SDK."""

import argparse
import getpass
import os
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from battleship_client import BattleshipClient, PlayerAPIError
from battleship_client.cli import save_session
from battleship_client.policies import create_policy

ROOT = Path(__file__).resolve().parents[1]


def stop(process: subprocess.Popen):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class Launcher:
    def __init__(self, url: str, stack: ExitStack):
        self.url = url.rstrip("/")
        self.stack = stack
        runs = ROOT / ".battleship" / "runs"
        runs.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = Path(tempfile.mkdtemp(prefix="game-", dir=runs))
        self.environment = dict(os.environ)
        self.environment["PYTHONPATH"] = os.pathsep.join(
            str(ROOT / "packages" / package / "src") for package in ("protocol", "client", "server"))

    def log(self, name: str):
        path = self.directory / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        return self.stack.enter_context(os.fdopen(fd, "w"))

    def spawn(self, arguments: list[str], *, log=None, interactive=False):
        process = subprocess.Popen(
            [sys.executable, "-m", *arguments], cwd=ROOT, env=self.environment,
            stdin=None if interactive else subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT if log is not None else None,
            start_new_session=not interactive,
        )
        self.stack.callback(stop, process)
        return process

    def ensure_server(self):
        with httpx.Client(timeout=0.5, trust_env=False) as http:
            try:
                response = http.get(self.url + "/openapi.json")
            except httpx.ConnectError:
                pass
            else:
                try:
                    schema = response.json()
                    valid = (response.status_code == 200
                             and schema.get("info", {}).get("title") == "Battleship player API"
                             and "/v1/matches" in schema.get("paths", {}))
                except (ValueError, AttributeError):
                    valid = False
                if not valid:
                    raise ValueError("This URL is occupied by another service; choose another PORT or URL.")
                print(f"Using existing server at {self.url}.", flush=True)
                return

            address = urlsplit(self.url)
            if (address.scheme != "http" or address.hostname not in ("127.0.0.1", "localhost", "::1")
                    or address.path not in ("", "/") or address.query or address.fragment
                    or address.username is not None or address.password is not None):
                raise ValueError("Server is unreachable. Automatic startup supports local HTTP URLs only.")
            port = address.port or 80
            process = self.spawn(["battleship_server", "--host", address.hostname, "--port", str(port)],
                                 log=self.log("server.log"))
            until = time.monotonic() + 10
            while time.monotonic() < until:
                if process.poll() is not None:
                    break
                try:
                    response = http.get(self.url + "/openapi.json")
                    if response.status_code == 200:
                        print(f"Started local server at {self.url}.", flush=True)
                        return
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
        raise RuntimeError(f"Could not start the server. See {self.directory / 'server.log'}")

    def player(self, session: Path, *, bot: str | None = None, seed: int = 42,
               weights: str | None = None, placement_weights: str | None = None):
        arguments = ["battleship_client", "--session", str(session)]
        if bot is None:
            return self.spawn([*arguments, "play"], interactive=True)
        arguments += ["bot", bot, "--seed", str(seed)]
        if weights is not None:
            arguments += ["--weights", weights]
        if placement_weights is not None:
            arguments += ["--placement-weights", placement_weights]
        return self.spawn(arguments, log=self.log(f"{session.stem}.log"))

    def create_game(self):
        host_path = self.directory / "human.session.json"
        bot_path = self.directory / "bot.session.json"
        with BattleshipClient(self.url) as host, BattleshipClient(self.url) as guest:
            room = host.create()
            guest.join(room.match_id, room.invitation)
            save_session(host_path, self.url, host.session)
            save_session(bot_path, self.url, guest.session)
        print(f"Created match {room.match_id}. Session files: {self.directory}", flush=True)
        return host_path, bot_path


def wait_for_players(players: list[subprocess.Popen]):
    remaining = set(players)
    while remaining:
        for player in list(remaining):
            result = player.poll()
            if result is not None:
                if result != 0:
                    raise RuntimeError("A player exited unexpectedly. Check the session directory logs.")
                remaining.remove(player)
        if remaining:
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description="Launch local games using the existing virtual environment")
    parser.add_argument("mode", choices=("play", "bots", "attach-bot"))
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--bot", choices=("hunt", "random", "weights"), default="hunt")
    parser.add_argument("--opponent", choices=("hunt", "random", "weights"), default="random")
    parser.add_argument("--weights")
    parser.add_argument("--opponent-weights")
    parser.add_argument("--placement-weights")
    parser.add_argument("--opponent-placement-weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--match")
    args = parser.parse_args()
    if args.mode == "attach-bot" and not args.match:
        parser.error("attach-bot requires --match")
    try:
        # Run before Launcher creates run directories or any HTTP/process work.
        create_policy(args.bot, args.seed, args.weights, args.placement_weights)
        if args.weights is not None:
            args.weights = str(Path(args.weights).resolve())
        if args.mode == "bots":
            create_policy(args.opponent, args.seed + 1, args.opponent_weights,
                          args.opponent_placement_weights)
            if args.opponent_weights is not None:
                args.opponent_weights = str(Path(args.opponent_weights).resolve())
        elif args.opponent_weights is not None:
            raise ValueError("--opponent-weights is only valid in bots mode")
        if args.mode != "bots" and args.opponent_placement_weights is not None:
            raise ValueError("--opponent-placement-weights is only valid in bots mode")
        for option in ("placement_weights", "opponent_placement_weights"):
            if getattr(args, option) is not None:
                setattr(args, option, str(Path(getattr(args, option)).resolve()))
        with ExitStack() as stack:
            launcher = Launcher(args.url, stack)
            if args.mode == "attach-bot":
                path = launcher.directory / "bot.session.json"
                with BattleshipClient(args.url) as client:
                    client.join(args.match, getpass.getpass("Invitation: "))
                    save_session(path, args.url, client.session)
                print(f"Bot attached. Session files: {launcher.directory}", flush=True)
                process = launcher.player(path, bot=args.bot, seed=args.seed, weights=args.weights,
                                          placement_weights=args.placement_weights)
                wait_for_players([process])
                print((launcher.directory / "bot.session.log").read_text(), end="")
            else:
                launcher.ensure_server()
                host, guest = launcher.create_game()
                bot = launcher.player(guest, bot=args.bot, seed=args.seed, weights=args.weights,
                                          placement_weights=args.placement_weights)
                if args.mode == "play":
                    human = launcher.player(host)
                    # Leaving the interactive UI should stop the managed bot too.
                    # A server that was already running is never stopped.
                    while human.poll() is None:
                        if bot.poll() not in (None, 0):
                            raise RuntimeError(f"Bot exited unexpectedly; see {launcher.directory}")
                        time.sleep(0.1)
                    if human.returncode != 0:
                        raise RuntimeError("Human client exited with an error.")
                else:
                    other = launcher.player(host, bot=args.opponent, seed=args.seed + 1,
                                            weights=args.opponent_weights,
                                            placement_weights=args.opponent_placement_weights)
                    wait_for_players([bot, other])
                    print((launcher.directory / "human.session.log").read_text(), end="")
    except KeyboardInterrupt:
        print("\nLauncher stopped. Existing servers and session files were left intact.")
    except (OSError, ValueError, ImportError, RuntimeError, EOFError, PlayerAPIError, httpx.HTTPError) as error:
        message = "Invalid session or request data." if isinstance(error, ValidationError) else str(error)
        print(f"Error: {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
