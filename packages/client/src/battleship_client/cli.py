"""Terminal player; all game access goes through the HTTP SDK."""

import argparse
import getpass
import json
import os
import re
import sys
from pathlib import Path

import httpx
from pydantic import ValidationError

from battleship_protocol import Coordinate, FLEET, Observation, PlayerSession, ShipPlacement
from battleship_client.policies import create_policy, play_policy
from battleship_client.sdk import BattleshipClient, PlayerAPIError


def parse_coordinate(text: str) -> Coordinate:
    match = re.fullmatch(r"([A-J])(10|[1-9])", text.strip().upper())
    if match is None:
        raise ValueError("Use a coordinate from A1 to J10.")
    return Coordinate(row=int(match[2]) - 1, column=ord(match[1]) - ord("A"))


def format_coordinate(cell: Coordinate) -> str:
    return f"{chr(ord('A') + cell.column)}{cell.row + 1}"


def render(observation: Observation) -> str:
    own = [["~" for _ in range(10)] for _ in range(10)]
    for ship in observation.own_fleet:
        for cell in ship.cells:
            own[cell.row][cell.column] = "S"
    for event in observation.incoming_shots:
        cell = event.coordinate
        own[cell.row][cell.column] = "o" if event.outcome == "miss" else "X"
    for ship in observation.own_fleet:
        if ship.sunk:
            for cell in ship.cells:
                own[cell.row][cell.column] = "#"
    symbols = {"unknown": ".", "miss": "o", "hit": "X", "sunk": "#"}
    target = [[symbols[cell] for cell in row] for row in observation.target_board]
    lines = [f"Match {observation.match_id} | player {observation.player + 1} | {observation.phase}",
             "       YOUR FLEET                    TARGET",
             "    A B C D E F G H I J         A B C D E F G H I J"]
    for r in range(10):
        lines.append(f"{r + 1:>2}  {' '.join(own[r])}     {r + 1:>2}  {' '.join(target[r])}")
    lines.append("S ship  ~ water  . unknown  o miss  X hit  # sunk")
    lines.append("Enemy ships afloat: " + ", ".join(observation.remaining_enemy_ships))
    if observation.shot_history:
        last = observation.shot_history[-1]
        detail = f" ({last.ship})" if last.ship else ""
        lines.append(f"Last shot: player {last.attacker + 1}, {format_coordinate(last.coordinate)}"
                     f", {last.outcome}{detail}")
    if observation.result:
        lines.append(f"Player {observation.result.winner + 1} wins: {observation.result.reason}.")
    elif observation.turn is not None:
        lines.append(f"Player {observation.turn + 1}'s turn.")
    else:
        lines.append(f"Fleet readiness: player 1={observation.ready[0]}, player 2={observation.ready[1]}")
    return "\n".join(lines)


def manual_fleet() -> list[ShipPlacement]:
    ships = []
    for name, length in FLEET.items():
        while True:
            try:
                text = input(f"{name} ({length} cells), start and direction (e.g. A1 H): ").split()
                if len(text) != 2 or text[1].upper() not in ("H", "V"):
                    raise ValueError("Enter a start cell and H or V.")
                ships.append(ShipPlacement(name=name, start=parse_coordinate(text[0]),
                                           orientation="horizontal" if text[1].upper() == "H" else "vertical"))
                break
            except ValueError as error:
                print(error)
    return ships


def play_human(client: BattleshipClient):
    observation = client.observe()
    while True:
        print(render(observation), flush=True)
        if observation.phase == "finished":
            return
        try:
            if not observation.ready[observation.player]:
                choice = input("Placement: [r]andom, [m]anual, or resign: ").strip().lower()
                if choice == "resign":
                    observation = client.resign()
                elif choice in ("r", "random"):
                    observation = client.place()
                elif choice in ("m", "manual"):
                    observation = client.place(manual_fleet())
                else:
                    print("Choose random, manual, or resign.")
            elif observation.phase == "playing" and observation.turn == observation.player:
                text = input("Shot (A1–J10), or resign: ").strip()
                observation = (client.resign() if text.lower() == "resign"
                               else client.fire(parse_coordinate(text)))
            else:
                print("Waiting for the other player… (Ctrl-C to leave; session is saved)", flush=True)
                observation = client.wait_for_change(observation.revision)
        except (ValueError, PlayerAPIError) as error:
            print(str(error))
            observation = client.observe()


def save_session(path: Path, url: str, session: PlayerSession):
    # Never overwrite another player's credentials or follow an existing symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump({"url": url, "session": session.model_dump(mode="json")}, output)
        output.write("\n")


def load_session(path: Path) -> tuple[str, PlayerSession]:
    data = json.loads(path.read_text())
    return data["url"], PlayerSession.model_validate(data["session"])


def main():
    parser = argparse.ArgumentParser(description="Battleship terminal player and bots")
    parser.add_argument("--url", help="Server URL (default: saved URL or http://127.0.0.1:8000)")
    parser.add_argument("--session", type=Path, default=Path(".battleship-session.json"),
                        help="Private credentials file; use a different file for each player")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create", help="Create a room and save player-one credentials")
    join = sub.add_parser("join", help="Join a room and save player-two credentials")
    join.add_argument("match_id")
    join.add_argument("--invitation", help="Invitation; omit to enter it privately at a prompt")
    show = sub.add_parser("show")
    show.add_argument("--json", action="store_true")
    place = sub.add_parser("place", help="Commit a fleet; automatic unless --manual is given")
    place.add_argument("--manual", action="store_true")
    fire = sub.add_parser("fire")
    fire.add_argument("coordinate")
    sub.add_parser("resign")
    sub.add_parser("play", help="Play interactively; resume from the saved session")
    bot = sub.add_parser("bot", help="Play a saved seat using a bot")
    bot.add_argument("kind", choices=("random", "hunt", "weights"))
    bot.add_argument("--seed", type=int)
    bot.add_argument("--weights", help="Weight artifact (.npz) or UTF-8 edits/response file")
    bot.add_argument("--placement-weights", help="Independent placement artifact or edits file")
    weights = sub.add_parser("weights", help="Manage fixed network weights (no session needed)")
    weight_commands = weights.add_subparsers(dest="weights_command", required=True)
    importer = weight_commands.add_parser("import", help="Validate edits and save an artifact")
    importer.add_argument("input", type=Path)
    importer.add_argument("--output", type=Path, required=True)
    importer.add_argument("--role", choices=("shooting", "placement"), default="shooting")
    args = parser.parse_args()
    try:
        if args.command == "weights":
            from battleship_client.weights import load_weight_source, save_weights
            if args.output.suffix.lower() != ".npz":
                raise ValueError("Weight artifact output must have a .npz extension")
            weights = load_weight_source(args.input, role=args.role)
            save_weights(args.output, weights)
            print(f"Saved {args.output}\nSHA-256: {weights.content_hash}")
            return
        if args.command == "bot":
            policy = create_policy(args.kind, args.seed, args.weights, args.placement_weights)
        if args.command in ("create", "join"):
            if args.session.exists():
                raise ValueError("Session file already exists; choose a new --session path.")
            url = args.url or "http://127.0.0.1:8000"
            with BattleshipClient(url) as client:
                if args.command == "create":
                    created = client.create()
                    save_session(args.session, url, client.session)
                    print(f"Match: {created.match_id}\nInvitation: {created.invitation}")
                else:
                    invitation = args.invitation or getpass.getpass("Invitation: ")
                    client.join(args.match_id, invitation)
                    save_session(args.session, url, client.session)
                print(f"Credentials saved to {args.session}. Keep this file private.")
            return
        url, session = load_session(args.session)
        with BattleshipClient(args.url or url, session=session) as client:
            if args.command == "play":
                play_human(client)
            elif args.command == "bot":
                print(render(play_policy(client, policy)))
            elif args.command == "show":
                observation = client.observe()
                print(observation.model_dump_json(indent=2) if args.json else render(observation))
            else:
                if args.command == "place":
                    observation = client.place(manual_fleet() if args.manual else None)
                elif args.command == "fire":
                    observation = client.fire(parse_coordinate(args.coordinate))
                else:
                    observation = client.resign()
                print(render(observation))
    except KeyboardInterrupt:
        print("\nLeft the client; saved credentials can resume the game.")
    except EOFError:
        print("Input closed; use play again to resume.", file=sys.stderr)
        raise SystemExit(1)
    except (OSError, ValueError, KeyError, ImportError, ValidationError, PlayerAPIError, httpx.HTTPError) as error:
        # Validation errors may include the session contents; do not print their repr.
        message = "Invalid session or request data." if isinstance(error, ValidationError) else str(error)
        print(f"Error: {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
