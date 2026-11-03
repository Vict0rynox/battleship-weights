"""Transport-independent rules and explicit per-player observations."""

import random
import time
from collections.abc import Callable, Sequence

from battleship_protocol import (
    BOARD_SIZE, FLEET, Coordinate, Observation, OwnShip, Result, ShipPlacement, ShotEvent,
)

Cell = tuple[int, int]
Fleet = dict[str, tuple[Cell, ...]]


class GameError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def placement_cells(ship: ShipPlacement) -> tuple[Cell, ...]:
    dr, dc = (0, 1) if ship.orientation == "horizontal" else (1, 0)
    return tuple((ship.start.row + dr * i, ship.start.column + dc * i)
                 for i in range(FLEET[ship.name]))


def validate_fleet(ships: Sequence[ShipPlacement]) -> Fleet:
    if len(ships) != len(FLEET) or {s.name for s in ships} != set(FLEET):
        raise GameError("invalid_fleet", "Supply each of the five ships exactly once.")
    fleet: Fleet = {}
    occupied: set[Cell] = set()
    for ship in ships:
        cells = placement_cells(ship)
        if any(not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE) for r, c in cells):
            raise GameError("invalid_fleet", "All ships must fit on the board.")
        if occupied.intersection(cells):
            raise GameError("invalid_fleet", "Ships must not overlap.")
        fleet[ship.name] = cells
        occupied.update(cells)
    return fleet


def random_fleet(rng: random.Random) -> Fleet:
    """Uniform over current legal placements, not over complete fleet layouts."""
    while True:
        fleet: Fleet = {}
        occupied: set[Cell] = set()
        for name, length in FLEET.items():
            candidates = []
            for dr, dc in ((0, 1), (1, 0)):
                for r in range(BOARD_SIZE - dr * (length - 1)):
                    for c in range(BOARD_SIZE - dc * (length - 1)):
                        cells = tuple((r + dr * i, c + dc * i) for i in range(length))
                        if not occupied.intersection(cells):
                            candidates.append(cells)
            if not candidates:
                break
            fleet[name] = rng.choice(candidates)
            occupied.update(fleet[name])
        if len(fleet) == len(FLEET):
            return fleet


class Game:
    """No credentials or network dependencies. Callers serialize access."""

    def __init__(self, match_id: str, *, rng: random.Random | None = None,
                 turn_timeout: float | None = None, clock: Callable[[], float] = time.monotonic,
                 shooting_only: bool = False):
        if turn_timeout is not None and (not 0 < turn_timeout < float("inf")):
            raise ValueError("turn_timeout must be finite and positive")
        self.match_id = match_id
        self.rng = rng if rng is not None else random.Random()
        self.turn_timeout = turn_timeout
        self.clock = clock
        # Local target practice only; the HTTP server always uses the default mode.
        self.shooting_only = shooting_only
        self.joined = False
        self.fleets: list[Fleet | None] = [None, None]
        self.shots: list[set[Cell]] = [set(), set()]
        self.history: list[ShotEvent] = []
        self.result: Result | None = None
        self.turn: int | None = None
        self.deadline: float | None = None
        self.revision = 0

    @property
    def phase(self) -> str:
        if self.result is not None:
            return "finished"
        if not self.joined:
            return "waiting"
        return "playing" if all(f is not None for f in self.fleets) else "placement"

    def _require_active(self):
        if self.result is not None:
            raise GameError("finished", "The match has finished.")

    def join(self):
        self._require_active()
        if self.joined:
            raise GameError("already_joined", "The second seat is occupied.")
        self.joined = True
        self.revision += 1
        self._start_if_ready()

    def place(self, player: int, ships: Sequence[ShipPlacement] | None = None):
        self._require_active()
        if self.fleets[player] is not None:
            raise GameError("already_placed", "Your fleet is already committed.")
        # Validate fully before committing; rejection never partially changes the fleet.
        fleet = random_fleet(self.rng) if ships is None else validate_fleet(ships)
        self.fleets[player] = fleet
        self.revision += 1
        self._start_if_ready()

    def _start_if_ready(self):
        if self.phase == "playing" and self.turn is None:
            self.turn = 0
            self._set_deadline()

    def _set_deadline(self):
        self.deadline = (self.clock() + self.turn_timeout
                         if self.turn_timeout is not None else None)

    def expire(self):
        if (self.phase == "playing" and self.deadline is not None
                and self.clock() >= self.deadline):
            self._finish(1 - self.turn, "timeout")

    def _finish(self, winner: int, reason: str):
        self.result = Result(winner=winner, reason=reason)
        self.turn = None
        self.deadline = None
        self.revision += 1

    def fire(self, player: int, coordinate: Coordinate):
        self.expire()
        self._require_active()
        if self.phase != "playing":
            raise GameError("not_playing", "Both players must join and place their fleets first.")
        if self.turn != player:
            raise GameError("not_your_turn", "Wait for your turn.")
        cell = (coordinate.row, coordinate.column)
        if cell in self.shots[player]:
            raise GameError("duplicate_shot", "You have already fired at this cell.")
        enemy = self.fleets[1 - player]
        self.shots[player].add(cell)
        hit_ship = next((name for name, cells in enemy.items() if cell in cells), None)
        sunk = hit_ship is not None and set(enemy[hit_ship]) <= self.shots[player]
        event = ShotEvent(
            number=len(self.history) + 1, attacker=player, coordinate=coordinate,
            outcome="sunk" if sunk else "hit" if hit_ship else "miss",
            ship=hit_ship if sunk else None,
            sunk_cells=[Coordinate(row=r, column=c) for r, c in enemy[hit_ship]] if sunk else [],
        )
        self.history.append(event)
        if all(set(cells) <= self.shots[player] for cells in enemy.values()):
            self._finish(player, "fleet_destroyed")
        else:
            self.turn = 0 if self.shooting_only else 1 - player
            self.revision += 1
            self._set_deadline()

    def resign(self, player: int):
        self.expire()
        self._require_active()
        self._finish(1 - player, "resignation")

    def observe(self, player: int) -> Observation:
        self.expire()
        # This allowlist is the only engine-to-player projection. No hidden fleet is serialized.
        board = [["unknown" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        remaining = list(FLEET)
        for event in self.history:
            if event.attacker != player:
                continue
            r, c = event.coordinate.row, event.coordinate.column
            board[r][c] = "miss" if event.outcome == "miss" else "hit"
            if event.outcome == "sunk":
                remaining.remove(event.ship)
                for cell in event.sunk_cells:
                    board[cell.row][cell.column] = "sunk"
        own_fleet = [OwnShip(
            name=name, length=FLEET[name],
            cells=[Coordinate(row=r, column=c) for r, c in cells],
            sunk=set(cells) <= self.shots[1 - player],
        ) for name, cells in (self.fleets[player] or {}).items()]
        may_shoot = self.phase == "playing" and self.turn == player
        return Observation(
            match_id=self.match_id, player=player, phase=self.phase, revision=self.revision,
            turn=self.turn, ready=[fleet is not None for fleet in self.fleets],
            own_fleet=own_fleet,
            incoming_shots=[e.model_copy(deep=True) for e in self.history if e.attacker != player],
            target_board=board, remaining_enemy_ships=remaining,
            shot_history=[e.model_copy(deep=True) for e in self.history],
            legal_shot_mask=[[may_shoot and (r, c) not in self.shots[player]
                              for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)],
            result=self.result.model_copy(deep=True) if self.result else None,
        )
