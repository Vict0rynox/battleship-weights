"""The public protocol. This package never imports the server or engine."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_VERSION = "1.0"
OBSERVATION_VERSION = "1.0"
NETWORK_VERSION = "1.0"
BOARD_SIZE = 10
FLEET = {"carrier": 5, "battleship": 4, "cruiser": 3, "submarine": 3, "destroyer": 2}

ShipName = Literal["carrier", "battleship", "cruiser", "submarine", "destroyer"]
CellState = Literal["unknown", "miss", "hit", "sunk"]
Player = Literal[0, 1]
ActionId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")]


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Coordinate(PublicModel):
    row: int = Field(ge=0, lt=BOARD_SIZE)
    column: int = Field(ge=0, lt=BOARD_SIZE)


class ShipPlacement(PublicModel):
    name: ShipName
    start: Coordinate
    orientation: Literal["horizontal", "vertical"]


class CreateMatch(PublicModel):
    action_id: ActionId


class JoinMatch(PublicModel):
    action_id: ActionId
    invitation: str = Field(min_length=20, max_length=128)


class PlacementRequest(PublicModel):
    action_id: ActionId
    mode: Literal["manual", "automatic"]
    ships: list[ShipPlacement] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def check_mode(self):
        if self.mode == "automatic" and self.ships:
            raise ValueError("Automatic placement takes no ships")
        if self.mode == "manual" and len(self.ships) != 5:
            raise ValueError("Manual placement needs five ships")
        return self


class ShotRequest(Coordinate):
    action_id: ActionId


class ResignRequest(PublicModel):
    action_id: ActionId


class PlayerSession(PublicModel):
    match_id: str
    player: Player
    token: str


class CreatedMatch(PlayerSession):
    invitation: str


class OwnShip(PublicModel):
    name: ShipName
    length: int
    cells: list[Coordinate]
    sunk: bool


class ShotEvent(PublicModel):
    number: int
    attacker: Player
    coordinate: Coordinate
    outcome: Literal["miss", "hit", "sunk"]
    ship: ShipName | None = None
    sunk_cells: list[Coordinate] = Field(default_factory=list)


class Result(PublicModel):
    winner: Player
    reason: Literal["fleet_destroyed", "resignation", "timeout"]


TargetRow = Annotated[list[CellState], Field(min_length=10, max_length=10)]
MaskRow = Annotated[list[bool], Field(min_length=10, max_length=10)]


class Observation(PublicModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    observation_version: Literal["1.0"] = OBSERVATION_VERSION
    match_id: str
    player: Player
    phase: Literal["waiting", "placement", "playing", "finished"]
    revision: int
    turn: Player | None
    ready: list[bool] = Field(min_length=2, max_length=2)
    own_fleet: list[OwnShip]
    incoming_shots: list[ShotEvent]
    target_board: list[TargetRow] = Field(min_length=10, max_length=10)
    remaining_enemy_ships: list[ShipName]
    shot_history: list[ShotEvent]
    legal_shot_mask: list[MaskRow] = Field(min_length=10, max_length=10)
    result: Result | None


class ErrorResponse(PublicModel):
    code: str
    message: str
