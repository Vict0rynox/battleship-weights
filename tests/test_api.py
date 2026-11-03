import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import make_room


def test_creation_and_join_retries(api):
    client, _ = api
    created = client.post("/v1/matches", json={"action_id": "same"}).json()
    assert created == client.post("/v1/matches", json={"action_id": "same"}).json()
    path = f"/v1/matches/{created['match_id']}/join"
    body = {"action_id": "join", "invitation": created["invitation"]}
    joined = client.post(path, json=body)
    assert joined.status_code == 200
    assert client.post(path, json=body).json() == joined.json()
    assert joined.json()["token"] != created["token"]
    assert client.post(path, json={**body, "action_id": "new"}).status_code == 401
    assert client.post(path, json={**body, "invitation": "x" * 32}).status_code == 409


def test_authentication_and_cross_match_access(api):
    client, _ = api
    match_id, headers, _ = make_room(client)
    path = f"/v1/matches/{match_id}/observation"
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer invalid"}).status_code == 401
    for player in (0, 1):
        response = client.get(path, headers=headers[player])
        assert response.json()["player"] == player
        assert "token" not in response.text and "invitation" not in response.text
    another, _, _ = make_room(client)
    assert client.get(f"/v1/matches/{another}/observation", headers=headers[0]).status_code == 401
    # The protocol has no player-selection field on an action.
    assert client.post(f"/v1/matches/{match_id}/shots", headers=headers[0], json={
        "action_id": "impersonate", "row": 0, "column": 0, "player": 1}).status_code == 422


def test_idempotency_conflict_and_cached_errors(api):
    client, store = api
    match_id, headers, _ = make_room(client)
    path = f"/v1/matches/{match_id}/shots"
    body = {"action_id": "shot", "row": 0, "column": 0}
    first = client.post(path, json=body, headers=headers[0])
    assert first.status_code == 200
    assert client.post(path, json=body, headers=headers[0]).json() == first.json()
    assert len(store.room(match_id).game.history) == 1
    assert client.post(path, json={**body, "column": 1}, headers=headers[0]).status_code == 409
    # IDs are scoped to a player, so the other seat can use the same ID.
    assert client.post(path, json=body, headers=headers[1]).status_code == 200
    rejected = {"action_id": "bad", "row": 0, "column": 0}
    assert client.post(path, json=rejected, headers=headers[0]).json()["code"] == "duplicate_shot"
    assert client.post(path, json={**rejected, "column": 2}, headers=headers[0]).json()["code"] == "action_conflict"


def test_concurrent_shots_only_one_turn_is_consumed(api):
    client, store = api
    match_id, headers, _ = make_room(client)
    def fire(i):
        return client.post(f"/v1/matches/{match_id}/shots", headers=headers[0], json={
            "action_id": f"concurrent-{i}", "row": 0, "column": i}).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(fire, (0, 1)))
    assert sorted(statuses) == [200, 409]
    assert len(store.room(match_id).game.history) == 1


def test_concurrent_retries_replay_one_response(api):
    client, store = api
    match_id, headers, _ = make_room(client)
    def fire(_):
        return client.post(f"/v1/matches/{match_id}/shots", headers=headers[0], json={
            "action_id": "retry", "row": 0, "column": 0}).json()
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(fire, range(4)))
    assert all(response == responses[0] for response in responses)
    assert len(store.room(match_id).game.history) == 1


def test_error_bodies_do_not_echo_secrets_or_fleets(api):
    client, _ = api
    match_id, headers, created = make_room(client)
    secret = "SECRET-DO-NOT-ECHO"
    response = client.post(f"/v1/matches/{match_id}/placement", headers=headers[0], json={
        "action_id": "bad", "mode": "manual", "ships": secret})
    assert response.status_code == 422
    assert secret not in response.text and created["token"] not in response.text
    observation = client.get(f"/v1/matches/{match_id}/observation", headers=headers[0]).json()
    assert set(observation) == {"protocol_version", "observation_version", "match_id", "player",
                               "phase", "revision", "turn", "ready", "own_fleet", "incoming_shots",
                               "target_board", "remaining_enemy_ships", "shot_history",
                               "legal_shot_mask", "result"}
    assert not ({"rng", "seed", "fleets", "tokens", "invitation"} & set(observation))


def test_non_ascii_invitation_is_rejected_safely(api):
    client, _ = api
    created = client.post("/v1/matches", json={"action_id": "unicode"}).json()
    response = client.post(f"/v1/matches/{created['match_id']}/join", json={
        "action_id": "unicode-join", "invitation": "é" * 32})
    assert response.status_code == 401 and "é" not in response.text


def test_configured_timeout_expires_without_player_request(fleet):
    import time
    from fastapi.testclient import TestClient
    from battleship_server.app import MatchStore, create_app
    now = [0.0]
    store = MatchStore(turn_timeout=30)
    with TestClient(create_app(store=store)) as client:
        match_id, headers, _ = make_room(client, automatic=False)
        game = store.room(match_id).game
        game.clock = lambda: now[0]
        for header in headers:
            assert client.post(f"/v1/matches/{match_id}/placement", headers=header, json={
                "action_id": "place", "mode": "manual",
                "ships": [s.model_dump() for s in fleet]}).status_code == 200
        now[0] = 30
        until = time.monotonic() + 1
        while game.result is None and time.monotonic() < until:
            time.sleep(0.01)
        assert game.result.reason == "timeout" and game.result.winner == 1
        observation = client.get(f"/v1/matches/{match_id}/observation", headers=headers[0]).json()
        assert observation["phase"] == "finished"


def test_manual_placement_and_lifecycle(api, fleet):
    client, _ = api
    match_id, headers, _ = make_room(client, automatic=False)
    path = f"/v1/matches/{match_id}"
    for header in headers:
        placed = client.post(f"{path}/placement", headers=header, json={
            "action_id": "manual", "mode": "manual", "ships": [s.model_dump() for s in fleet]})
        assert placed.status_code == 200
    observed = client.get(f"{path}/observation", headers=headers[0]).json()
    assert observed["phase"] == "playing" and observed["turn"] == 0
    result = client.post(f"{path}/resign", headers=headers[1], json={"action_id": "resign"}).json()
    assert result["result"] == {"winner": 0, "reason": "resignation"}
    assert client.post(f"{path}/shots", headers=headers[0], json={
        "action_id": "late", "row": 9, "column": 9}).json()["code"] == "finished"


def test_openapi_exposes_only_player_routes(api):
    client, _ = api
    schema = client.get("/openapi.json").json()
    assert len(schema["paths"]) == 6
    text = json.dumps(schema)
    assert "HTTPBearer" in text
    assert "seed" not in text and "admin" not in text
