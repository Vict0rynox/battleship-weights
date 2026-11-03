import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from battleship_client.cli import format_coordinate, parse_coordinate, render, save_session
from battleship_client.policies import HuntTargetBot, RandomBot, play_policy
from battleship_client.sdk import BattleshipClient, PlayerAPIError
from battleship_protocol import Coordinate
from battleship_server.app import create_app


@pytest.fixture(scope="module")
def live_url():
    # A bound ephemeral socket avoids a port-selection race and uses real TCP/HTTP.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    end = time.monotonic() + 5
    while not server.started:
        if not thread.is_alive() or time.monotonic() >= end:
            pytest.fail("HTTP server did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


def test_sdk_http_lifecycle(live_url, fleet):
    with BattleshipClient(live_url) as first, BattleshipClient(live_url) as second:
        created = first.create()
        assert second.join(created.match_id, created.invitation).player == 1
        assert first.observe().phase == "placement"
        assert first.place(fleet).ready == [True, False]
        second.place()
        before = first.observe()
        shot = first.fire(Coordinate(row=9, column=9), action_id="retry")
        assert first.fire(Coordinate(row=9, column=9), action_id="retry") == shot
        assert first.wait_for_change(before.revision).revision == shot.revision
        with pytest.raises(PlayerAPIError) as error:
            first.fire(Coordinate(row=9, column=8))
        assert error.value.code == "not_your_turn"
        with pytest.raises(TimeoutError):
            first.wait_for_change(shot.revision, timeout=0.02, poll_interval=0.01)
        assert second.resign().result.winner == 0


def test_sdk_retries_preserve_action_id_after_lost_response():
    from battleship_protocol import CreatedMatch
    bodies = []
    response = CreatedMatch(match_id="match", player=0, token="secret", invitation="invite")
    def handle(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            raise httpx.ReadError("Response lost after server accepted request", request=request)
        return httpx.Response(200, json=response.model_dump())
    with BattleshipClient(transport=httpx.MockTransport(handle)) as client:
        assert client.create().match_id == "match"
    assert len(bodies) == 2 and bodies[0] == bodies[1]


def test_sdk_does_not_retry_application_errors():
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(409, json={"code": "finished", "message": "Match finished."})
    with BattleshipClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(PlayerAPIError):
            client.create()
    assert len(calls) == 1


def test_policy_placement_hook_and_terminal_result(live_url, fleet):
    class Policy:
        called = False
        def choose_placement(self, observation):
            self.called = True
            return fleet
        def choose_shot(self, observation):
            raise AssertionError("No shot should be requested after opponent resignation")
    with BattleshipClient(live_url) as first, BattleshipClient(live_url) as second:
        created = first.create()
        second.join(created.match_id, created.invitation)
        policy = Policy()
        results = []
        thread = threading.Thread(target=lambda: results.append(play_policy(first, policy, poll_interval=0.01)))
        thread.start()
        end = time.monotonic() + 3
        while not second.observe().ready[0]:
            assert time.monotonic() < end
            time.sleep(0.01)
        second.resign()
        thread.join(timeout=3)
        assert policy.called and not thread.is_alive()
        assert results[0].result.winner == 0


def test_seeded_bot_legality_and_reproducibility(game):
    observation = game.observe(0)
    for bot_class in (RandomBot, HuntTargetBot):
        first, second = bot_class(91), bot_class(91)
        for _ in range(20):
            a, b = first.choose_shot(observation), second.choose_shot(observation)
            assert a == b and observation.legal_shot_mask[a.row][a.column]
        with pytest.raises(ValueError):
            bot_class(1).choose_shot(game.observe(1))


def test_hunt_target_prioritizes_line_and_handles_touching_hits(game):
    observation = game.observe(0)
    observation.target_board[3][3] = "hit"
    observation.target_board[3][4] = "hit"
    observation.target_board[4][3] = "hit"  # Could belong to a touching ship.
    for r, c in ((3, 3), (3, 4), (4, 3)):
        observation.legal_shot_mask[r][c] = False
    assert HuntTargetBot(2).choose_shot(observation) in [
        Coordinate(row=3, column=2), Coordinate(row=3, column=5),
        Coordinate(row=2, column=3), Coordinate(row=5, column=3)]
    # Once sunk, disclosed cells no longer create target candidates.
    for r, c in ((3, 3), (3, 4), (4, 3)):
        observation.target_board[r][c] = "sunk"
    chosen = HuntTargetBot(2).choose_shot(observation)
    assert (chosen.row + chosen.column) % 2 == 0


def test_hunt_does_not_infer_water_next_to_sunk_ship(game):
    observation = game.observe(0)
    observation.target_board[3][3] = "sunk"
    observation.target_board[3][4] = "hit"
    for r in range(10):
        for c in range(10):
            observation.legal_shot_mask[r][c] = (r, c) == (3, 5)
    assert HuntTargetBot(1).choose_shot(observation) == Coordinate(row=3, column=5)


@pytest.mark.parametrize("text,expected", [("A1", (0, 0)), ("j10", (9, 9)), (" D5 ", (4, 3))])
def test_terminal_coordinates(text, expected):
    cell = parse_coordinate(text)
    assert (cell.row, cell.column) == expected
    assert parse_coordinate(format_coordinate(cell)) == cell


@pytest.mark.parametrize("text", ["A0", "K1", "A11", "1A", "A01", "A1 H", ""])
def test_bad_terminal_coordinates(text):
    with pytest.raises(ValueError):
        parse_coordinate(text)


def test_terminal_rendering_and_private_session(game, tmp_path):
    from battleship_protocol import PlayerSession
    observation = game.observe(0)
    text = render(observation)
    assert "YOUR FLEET" in text and "TARGET" in text and "Player 1's turn" in text
    path = tmp_path / "private.session.json"
    session = PlayerSession(match_id="id", player=0, token="private")
    save_session(path, "http://127.0.0.1:8000", session)
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        save_session(path, "http://127.0.0.1:8000", session)


def cli(*args, input=None):
    return subprocess.run([sys.executable, "-m", "battleship_client", *map(str, args)],
                          input=input, text=True, capture_output=True, timeout=10, check=True)


def test_terminal_commands_over_http(live_url, tmp_path):
    host, guest = tmp_path / "host.session.json", tmp_path / "guest.session.json"
    created = cli("--url", live_url, "--session", host, "create")
    lines = created.stdout.splitlines()
    match_id = lines[0].split(": ", 1)[1]
    invitation = lines[1].split(": ", 1)[1]
    cli("--url", live_url, "--session", guest, "join", match_id, "--invitation", invitation)
    manual_input = "A1 H\nA2 H\nA3 H\nA4 H\nA5 H\n"
    cli("--session", host, "place", "--manual", input=manual_input)
    cli("--session", guest, "place")
    assert "Last shot" in cli("--session", host, "fire", "J10").stdout
    observed = json.loads(cli("--session", guest, "show", "--json").stdout)
    assert observed["player"] == 1 and observed["turn"] == 1
    assert "Player 1 wins" in cli("--session", guest, "resign").stdout
    assert "Player 1 wins" in cli("--session", host, "play").stdout


def test_bot_vs_bot_in_separate_processes(live_url, tmp_path):
    host, guest = tmp_path / "host.session.json", tmp_path / "guest.session.json"
    with BattleshipClient(live_url) as first, BattleshipClient(live_url) as second:
        created = first.create()
        second.join(created.match_id, created.invitation)
        save_session(host, live_url, first.session)
        save_session(guest, live_url, second.session)
        processes = [subprocess.Popen([sys.executable, "-m", "battleship_client", "--session", str(path),
                                      "bot", kind, "--seed", "19"], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
                     for path, kind in ((host, "random"), (guest, "hunt"))]
        try:
            for process in processes:
                stdout, stderr = process.communicate(timeout=50)
                assert process.returncode == 0, stderr
                assert "wins: fleet_destroyed" in stdout
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        final = first.observe()
        assert final.phase == "finished" and len(final.shot_history) <= 199
        for player in (0, 1):
            cells = [(e.coordinate.row, e.coordinate.column) for e in final.shot_history if e.attacker == player]
            assert len(cells) == len(set(cells))


def test_player_source_has_no_server_dependency():
    import ast
    from importlib.metadata import requires
    import battleship_client
    import battleship_protocol
    for package in (battleship_client, battleship_protocol):
        for path in Path(package.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("battleship_server")
                elif isinstance(node, ast.Import):
                    assert not any(alias.name.startswith("battleship_server") for alias in node.names)
    assert not any("battleship-server" in dependency for dependency in requires("battleship-client"))


def assert_complete_legal_game(observation):
    assert observation.phase == "finished"
    assert observation.result.reason == "fleet_destroyed"
    assert len(observation.shot_history) <= 199
    for seat in (0, 1):
        cells = [(event.coordinate.row, event.coordinate.column)
                 for event in observation.shot_history if event.attacker == seat]
        assert len(cells) == len(set(cells))


def test_weight_player_sdk_game(live_url, tmp_path):
    from battleship_client.weights import WeightPlayer, load_weights, parse_edits, save_weights
    artifact = tmp_path / "sdk.npz"
    save_weights(artifact, parse_edits("W1[0,3]=.2\nW2[3,7]=.5\nW3[7,42]=2\nb3[0]=.1"))
    player = WeightPlayer(load_weights(artifact), 21)
    with BattleshipClient(live_url) as first, BattleshipClient(live_url) as second:
        room = first.create()
        second.join(room.match_id, room.invitation)
        results, errors = [], []
        def play(client, policy):
            try:
                results.append(play_policy(client, policy, poll_interval=0.001))
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=play, args=(client, policy), daemon=True)
                   for client, policy in ((first, player), (second, HuntTargetBot(22)))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        assert not errors, errors
        assert all(not thread.is_alive() for thread in threads)
        assert len(results) == 2
        assert_complete_legal_game(first.observe())


@pytest.mark.parametrize("kinds", [("weights", "hunt"), ("random", "weights"), ("weights", "weights")])
def test_weight_players_in_separate_processes(live_url, tmp_path, kinds):
    source = tmp_path / "synthetic response.txt"
    source.write_text("Synthetic only.\n```edits\nW1[400,1]=.1\nW2[1,2]=2\nW3[2,50]=3\nb3[0]=.2\n```\n")
    artifact = tmp_path / "model.npz"
    cli("weights", "import", source, "--output", artifact)
    with BattleshipClient(live_url) as first, BattleshipClient(live_url) as second:
        created = first.create()
        second.join(created.match_id, created.invitation)
        processes = []
        try:
            for seat, (client, kind, weight_path) in enumerate(zip((first, second), kinds, (artifact, source))):
                session = tmp_path / f"seat-{seat}.session.json"
                save_session(session, live_url, client.session)
                command = [sys.executable, "-m", "battleship_client", "--session", str(session),
                           "bot", kind, "--seed", str(seat + 1)]
                if kind == "weights":
                    command += ["--weights", str(weight_path)]
                processes.append(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            for process in processes:
                stdout, stderr = process.communicate(timeout=50)
                assert process.returncode == 0, stderr
                assert "wins: fleet_destroyed" in stdout
            assert_complete_legal_game(first.observe())
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


def test_make_weight_game_with_existing_server(live_url, tmp_path):
    source = tmp_path / "make synthetic.edits"
    source.write_text("b3[0]=.1")
    result = subprocess.run(["make", "bots", "BOT=weights", f"WEIGHTS={source}", "OPPONENT=weights",
                             f"OPPONENT_WEIGHTS={source}", f"URL={live_url}"],
                            cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=55)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Using existing server" in result.stdout
    assert "wins: fleet_destroyed" in result.stdout
    # The launcher leaves the externally managed server alive.
    assert httpx.get(live_url + "/openapi.json").status_code == 200


@pytest.mark.parametrize("kind", ["random", "hunt", "weights"])
def test_two_model_gameplay_and_resumed_placement(live_url, tmp_path, kind):
    from battleship_client.weights import PlacementWeightPlayer, load_weight_source
    placement = tmp_path / "placement source.edits"
    placement.write_text("b3[199]=.5")
    shooting = tmp_path / "shooting source.edits"
    shooting.write_text("b3[0]=.2")
    with BattleshipClient(live_url) as first, BattleshipClient(live_url) as second:
        room = first.create()
        second.join(room.match_id, room.invitation)
        # A resumed bot must preserve its already committed fleet.
        committed = PlacementWeightPlayer(load_weight_source(placement, role="placement"), 9).choose_placement(
            first.observe())
        first.place(committed)
        initial = first.observe().own_fleet
        processes = []
        try:
            for seat, client in enumerate((first, second)):
                path = tmp_path / f"seat-{seat}.json"
                save_session(path, live_url, client.session)
                command = [sys.executable, "-m", "battleship_client", "--session", str(path),
                           "bot", kind, "--seed", str(seat + 1), "--placement-weights", str(placement)]
                if kind == "weights":
                    command += ["--weights", str(shooting)]
                processes.append(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            for process in processes:
                stdout, stderr = process.communicate(timeout=50)
                assert process.returncode == 0, stderr
                assert "fleet_destroyed" in stdout
            final = first.observe()
            assert_complete_legal_game(final)
            assert [(s.name, s.cells) for s in initial] == [(s.name, s.cells) for s in final.own_fleet]
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


def test_make_two_independent_placement_models(live_url, tmp_path):
    first, other = tmp_path / "first placement.edits", tmp_path / "other placement.edits"
    first.write_text("b3[0]=.5")
    other.write_text("b3[199]=.5")
    result = subprocess.run(["make", "bots", "BOT=random", "OPPONENT=hunt",
                             f"PLACEMENT_WEIGHTS={first}", f"OPPONENT_PLACEMENT_WEIGHTS={other}",
                             f"URL={live_url}"], cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True, timeout=55)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "wins: fleet_destroyed" in result.stdout
