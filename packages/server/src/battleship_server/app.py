"""Authenticated HTTP boundary; no administrative state endpoints."""

import asyncio
import contextlib
import random
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from battleship_protocol import (
    Coordinate, CreatedMatch, CreateMatch, ErrorResponse, JoinMatch, Observation,
    PlacementRequest, PlayerSession, ResignRequest, ShotRequest,
)
from battleship_server.engine import Game, GameError


@dataclass
class Room:
    game: Game
    tokens: list[str]
    invitation: str | None
    lock: threading.RLock = field(default_factory=threading.RLock)
    actions: dict = field(default_factory=dict)
    joins: dict = field(default_factory=dict)


class MatchStore:
    def __init__(self, *, turn_timeout: float | None = None,
                 rng_factory: Callable[[], random.Random] = random.Random):
        if turn_timeout is not None and not 0 < turn_timeout < float("inf"):
            raise ValueError("turn_timeout must be finite and positive")
        self.turn_timeout = turn_timeout
        self.rng_factory = rng_factory
        self.rooms: dict[str, Room] = {}
        self.creates: dict[str, CreatedMatch] = {}
        self.lock = threading.RLock()

    def create(self, action_id: str) -> CreatedMatch:
        with self.lock:
            if action_id in self.creates:
                return self.creates[action_id]
            match_id = secrets.token_urlsafe(18)
            room = Room(
                game=Game(match_id, rng=self.rng_factory(), turn_timeout=self.turn_timeout),
                tokens=[secrets.token_urlsafe(32), secrets.token_urlsafe(32)],
                invitation=secrets.token_urlsafe(32),
            )
            self.rooms[match_id] = room
            response = CreatedMatch(match_id=match_id, player=0, token=room.tokens[0],
                                    invitation=room.invitation)
            self.creates[action_id] = response
            return response

    def room(self, match_id: str) -> Room:
        with self.lock:
            room = self.rooms.get(match_id)
        if room is None:
            raise GameError("not_found", "Match not found.")
        return room

    def expire(self):
        with self.lock:
            rooms = list(self.rooms.values())
        for room in rooms:
            with room.lock:
                room.game.expire()


def create_app(*, turn_timeout: float | None = None, store: MatchStore | None = None) -> FastAPI:
    store = store if store is not None else MatchStore(turn_timeout=turn_timeout)

    @asynccontextmanager
    async def lifespan(app):
        async def sweep():
            while True:
                await asyncio.sleep(0.1)
                store.expire()
        task = asyncio.create_task(sweep())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Battleship player API", version="1.0", lifespan=lifespan,
                  responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse},
                             404: {"model": ErrorResponse}, 409: {"model": ErrorResponse},
                             422: {"model": ErrorResponse}})
    bearer = HTTPBearer(auto_error=False)

    @app.exception_handler(GameError)
    async def game_error(request: Request, error: GameError):
        status = {"unauthorized": 401, "not_found": 404, "invalid_fleet": 400}.get(error.code, 409)
        return JSONResponse(status_code=status, content={"code": error.code, "message": error.message})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # FastAPI's default includes rejected inputs. Never echo secrets or fleet bodies.
        return JSONResponse(status_code=422, content={
            "code": "invalid_request", "message": "Request does not match the public API schema."})

    def authorize(match_id: str, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise GameError("unauthorized", "Valid player credentials are required.")
        room = store.room(match_id)
        for player, token in enumerate(room.tokens):
            if secrets.compare_digest(credentials.credentials.encode("utf-8"), token.encode("ascii")):
                return room, player
        raise GameError("unauthorized", "Valid player credentials are required.")

    def mutate(room: Room, player: int, operation: str, body, perform) -> Observation:
        fingerprint = (operation, body.model_dump_json())
        key = (player, body.action_id)
        with room.lock:
            room.game.expire()
            cached = room.actions.get(key)
            if cached is not None:
                previous, response = cached
                if previous != fingerprint:
                    raise GameError("action_conflict", "This action ID was used with different content.")
                if isinstance(response, GameError):
                    raise GameError(response.code, response.message)
                return response.model_copy(deep=True)
            try:
                perform()
                response = room.game.observe(player)
            except GameError as error:
                room.actions[key] = (fingerprint, error)
                raise
            room.actions[key] = (fingerprint, response.model_copy(deep=True))
            return response

    @app.post("/v1/matches", response_model=CreatedMatch)
    def create(body: CreateMatch):
        return store.create(body.action_id)

    @app.post("/v1/matches/{match_id}/join", response_model=PlayerSession)
    def join(match_id: str, body: JoinMatch):
        room = store.room(match_id)
        with room.lock:
            fingerprint = body.model_dump_json()
            cached = room.joins.get(body.action_id)
            if cached:
                if cached[0] != fingerprint:
                    raise GameError("action_conflict", "This action ID was used with different content.")
                return cached[1]
            if room.invitation is None or not secrets.compare_digest(
                    body.invitation.encode("utf-8"), room.invitation.encode("ascii")):
                raise GameError("unauthorized", "The invitation is invalid or already used.")
            room.game.join()
            room.invitation = None
            response = PlayerSession(match_id=match_id, player=1, token=room.tokens[1])
            room.joins[body.action_id] = (fingerprint, response)
            return response

    @app.get("/v1/matches/{match_id}/observation", response_model=Observation)
    def observation(auth=Depends(authorize)):
        room, player = auth
        with room.lock:
            return room.game.observe(player)

    @app.post("/v1/matches/{match_id}/placement", response_model=Observation)
    def placement(body: PlacementRequest, auth=Depends(authorize)):
        room, player = auth
        return mutate(room, player, "placement", body,
                      lambda: room.game.place(player, body.ships if body.mode == "manual" else None))

    @app.post("/v1/matches/{match_id}/shots", response_model=Observation)
    def shot(body: ShotRequest, auth=Depends(authorize)):
        room, player = auth
        coordinate = Coordinate(row=body.row, column=body.column)
        return mutate(room, player, "shot", body, lambda: room.game.fire(player, coordinate))

    @app.post("/v1/matches/{match_id}/resign", response_model=Observation)
    def resign(body: ResignRequest, auth=Depends(authorize)):
        room, player = auth
        return mutate(room, player, "resign", body, lambda: room.game.resign(player))

    return app
