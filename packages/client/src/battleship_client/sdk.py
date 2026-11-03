import time
import uuid
from collections.abc import Sequence

import httpx

from battleship_protocol import (
    Coordinate, CreatedMatch, CreateMatch, JoinMatch, Observation, PlacementRequest,
    PlayerSession, ResignRequest, ShipPlacement, ShotRequest,
)


class PlayerAPIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


class BattleshipClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", *,
                 session: PlayerSession | None = None, timeout: float = 5.0,
                 retries: int = 2, transport: httpx.BaseTransport | None = None):
        if retries < 0:
            raise ValueError("retries cannot be negative")
        self.session = session
        self.retries = retries
        self.http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport,
                                 follow_redirects=False, trust_env=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.http.close()

    @staticmethod
    def _action_id(action_id: str | None) -> str:
        return action_id if action_id is not None else uuid.uuid4().hex

    def _request(self, method: str, path: str, model, *, body=None, authenticated=False):
        headers = {}
        if authenticated:
            if self.session is None:
                raise ValueError("Create or join a match first")
            headers["Authorization"] = f"Bearer {self.session.token}"
        payload = body.model_dump(mode="json") if body is not None else None
        for attempt in range(self.retries + 1):
            try:
                response = self.http.request(method, path, json=payload, headers=headers)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
            else:
                if response.status_code not in (502, 503, 504) or attempt == self.retries:
                    break
            time.sleep(0.1 * 2 ** attempt)
        if response.is_error:
            try:
                error = response.json()
                code, message = error["code"], error["message"]
            except (ValueError, KeyError, TypeError):
                code, message = "http_error", "The server returned an unexpected error."
            raise PlayerAPIError(response.status_code, code, message)
        return model.model_validate_json(response.content)

    def create(self, *, action_id: str | None = None) -> CreatedMatch:
        response = self._request("POST", "/v1/matches", CreatedMatch,
                                 body=CreateMatch(action_id=self._action_id(action_id)))
        self.session = PlayerSession(match_id=response.match_id, player=response.player,
                                     token=response.token)
        return response

    def join(self, match_id: str, invitation: str, *, action_id: str | None = None) -> PlayerSession:
        response = self._request("POST", f"/v1/matches/{match_id}/join", PlayerSession,
                                 body=JoinMatch(action_id=self._action_id(action_id),
                                                invitation=invitation))
        self.session = response
        return response

    def _path(self, operation: str) -> str:
        if self.session is None:
            raise ValueError("Create or join a match first")
        return f"/v1/matches/{self.session.match_id}/{operation}"

    def observe(self) -> Observation:
        return self._request("GET", self._path("observation"), Observation, authenticated=True)

    def place(self, ships: Sequence[ShipPlacement] | None = None, *,
              action_id: str | None = None) -> Observation:
        body = PlacementRequest(action_id=self._action_id(action_id),
                                mode="automatic" if ships is None else "manual",
                                ships=list(ships) if ships is not None else [])
        return self._request("POST", self._path("placement"), Observation,
                             body=body, authenticated=True)

    def fire(self, coordinate: Coordinate, *, action_id: str | None = None) -> Observation:
        body = ShotRequest(action_id=self._action_id(action_id),
                           row=coordinate.row, column=coordinate.column)
        return self._request("POST", self._path("shots"), Observation,
                             body=body, authenticated=True)

    def resign(self, *, action_id: str | None = None) -> Observation:
        return self._request("POST", self._path("resign"), Observation,
                             body=ResignRequest(action_id=self._action_id(action_id)),
                             authenticated=True)

    def wait_for_change(self, revision: int, *, poll_interval: float = 0.25,
                        timeout: float | None = None) -> Observation:
        if poll_interval <= 0 or (timeout is not None and timeout < 0):
            raise ValueError("Invalid polling interval or timeout")
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            observation = self.observe()
            if observation.revision != revision or observation.phase == "finished":
                return observation
            if end is not None and time.monotonic() >= end:
                raise TimeoutError("No match change before polling timeout")
            delay = poll_interval if end is None else min(poll_interval, max(0, end - time.monotonic()))
            time.sleep(delay)
