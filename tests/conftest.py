import random

import pytest
from fastapi.testclient import TestClient

from battleship_protocol import Coordinate, FLEET, ShipPlacement
from battleship_server.app import MatchStore, create_app
from battleship_server.engine import Game


def make_fleet(column=0):
    # Adjacent rows deliberately exercise the ships-may-touch rule.
    return [ShipPlacement(name=name, start=Coordinate(row=i, column=column), orientation="horizontal")
            for i, name in enumerate(FLEET)]


@pytest.fixture
def fleet():
    return make_fleet()


@pytest.fixture
def game(fleet):
    game = Game("test-match", rng=random.Random(7))
    game.join()
    game.place(0, fleet)
    game.place(1, fleet)
    return game


@pytest.fixture
def api():
    store = MatchStore(rng_factory=lambda: random.Random(17))
    with TestClient(create_app(store=store)) as client:
        yield client, store


def make_room(client, *, automatic=True):
    import uuid
    created = client.post("/v1/matches", json={"action_id": uuid.uuid4().hex}).json()
    match_id = created["match_id"]
    joined = client.post(f"/v1/matches/{match_id}/join", json={
        "action_id": "join", "invitation": created["invitation"]}).json()
    headers = [{"Authorization": f"Bearer {seat['token']}"} for seat in (created, joined)]
    if automatic:
        for header in headers:
            assert client.post(f"/v1/matches/{match_id}/placement", headers=header,
                               json={"action_id": "place", "mode": "automatic"}).status_code == 200
    return match_id, headers, created
