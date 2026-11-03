import random

import pytest
from pydantic import ValidationError

from battleship_protocol import Coordinate, FLEET, ShipPlacement
from battleship_server.engine import Game, GameError, random_fleet, validate_fleet
from conftest import make_fleet


def test_placement_and_waiting(fleet):
    game = Game("waiting")
    assert game.observe(0).phase == "waiting"
    game.place(0, fleet)
    assert game.observe(0).ready == [True, False]
    game.join()
    assert game.phase == "placement"
    game.place(1, fleet)
    assert game.phase == "playing" and game.turn == 0
    assert sum(len(cells) for cells in game.fleets[0].values()) == 17
    before = game.observe(0)
    with pytest.raises(GameError, match="already committed"):
        game.place(0)
    assert game.observe(0) == before


@pytest.mark.parametrize("kind", ["duplicate", "overlap", "outside", "missing"])
def test_invalid_fleets_do_not_commit(kind, fleet):
    ships = list(fleet)
    if kind == "duplicate":
        ships[-1] = ships[0]
    elif kind == "overlap":
        ships[1] = ShipPlacement(name="battleship", start=Coordinate(row=0, column=0), orientation="horizontal")
    elif kind == "outside":
        ships[0] = ShipPlacement(name="carrier", start=Coordinate(row=9, column=9), orientation="vertical")
    else:
        ships.pop()
    game = Game("bad")
    with pytest.raises(GameError):
        game.place(0, ships)
    assert game.fleets == [None, None] and game.revision == 0


def test_vertical_boundary_and_touching(fleet):
    ships = [ShipPlacement(name=name, start=Coordinate(row=10 - length, column=i), orientation="vertical")
             for i, (name, length) in enumerate(FLEET.items())]
    assert len(validate_fleet(ships)) == 5
    assert len(validate_fleet(fleet)) == 5


def test_seeded_random_fleets():
    rng1, rng2 = random.Random(19), random.Random(19)
    for _ in range(25):
        fleet = random_fleet(rng1)
        assert fleet == random_fleet(rng2)
        cells = [cell for ship in fleet.values() for cell in ship]
        assert len(set(cells)) == len(cells) == 17
        assert all(0 <= r < 10 and 0 <= c < 10 for r, c in cells)


def test_hit_changes_turn_and_hides_identity(game):
    game.fire(0, Coordinate(row=0, column=0))
    observation = game.observe(0)
    assert observation.turn == 1
    assert observation.target_board[0][0] == "hit"
    assert observation.shot_history[0].ship is None
    assert observation.shot_history[0].sunk_cells == []
    assert not any(any(row) for row in observation.legal_shot_mask)
    assert game.observe(1).incoming_shots[0].ship is None


def test_sinking_relabels_previous_hits_and_reveals_cells(game):
    for c in range(5):
        game.fire(0, Coordinate(row=0, column=c))
        if c < 4:
            game.fire(1, Coordinate(row=9, column=c))
    observation = game.observe(0)
    assert observation.target_board[0][:5] == ["sunk"] * 5
    assert "carrier" not in observation.remaining_enemy_ships
    last = observation.shot_history[-1]
    assert last.ship == "carrier" and len(last.sunk_cells) == 5
    assert all(e.ship is None for e in observation.shot_history[:-1])


def test_rejections_do_not_change_state(game):
    before = game.observe(0)
    with pytest.raises(GameError) as error:
        game.fire(1, Coordinate(row=9, column=9))
    assert error.value.code == "not_your_turn"
    assert before == game.observe(0)
    game.fire(0, Coordinate(row=9, column=9))
    game.fire(1, Coordinate(row=9, column=9))
    before = game.observe(0)
    with pytest.raises(GameError) as error:
        game.fire(0, Coordinate(row=9, column=9))
    assert error.value.code == "duplicate_shot"
    assert game.observe(0) == before


@pytest.mark.parametrize("kwargs", [{"row": -1, "column": 0}, {"row": 0, "column": 10},
                                    {"row": True, "column": 0}, {"row": 1.1, "column": 0}])
def test_coordinate_bounds(kwargs):
    with pytest.raises(ValidationError):
        Coordinate(**kwargs)


def test_victory_and_final_privacy(game):
    cells = [Coordinate(row=r, column=c) for ship in game.fleets[1].values() for r, c in ship]
    for i, cell in enumerate(cells):
        game.fire(0, cell)
        if i != len(cells) - 1:
            game.fire(1, Coordinate(row=8 + i // 10, column=i % 10))
    assert game.result.winner == 0 and game.result.reason == "fleet_destroyed"
    assert game.turn is None and game.phase == "finished"
    loser = game.observe(1)
    assert all(s == "unknown" for row in loser.target_board[:8] for s in row)
    assert not any(any(row) for row in loser.legal_shot_mask)
    for action in (lambda: game.fire(0, Coordinate(row=9, column=9)),
                   lambda: game.place(0), lambda: game.resign(1), lambda: game.join()):
        with pytest.raises(GameError):
            action()


def test_timeout_and_invalid_actions_do_not_extend_deadline(fleet):
    now = [0.0]
    game = Game("timer", turn_timeout=30, clock=lambda: now[0])
    game.join()
    game.place(0, fleet)
    assert game.deadline is None
    game.place(1, fleet)
    assert game.deadline == 30
    now[0] = 10
    with pytest.raises(GameError):
        game.fire(1, Coordinate(row=9, column=9))
    assert game.deadline == 30
    now[0] = 20
    game.fire(0, Coordinate(row=9, column=9))
    assert game.deadline == 50
    now[0] = 50
    observation = game.observe(1)
    assert observation.result.winner == 0 and observation.result.reason == "timeout"


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_bad_timeout(value):
    with pytest.raises(ValueError):
        Game("bad", turn_timeout=value)


def test_no_default_deadline_and_resignation(game):
    assert game.deadline is None
    game.resign(0)
    assert game.result.winner == 1 and game.result.reason == "resignation"


@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("hit", [False, True])
def test_hidden_layout_noninterference(finished, hit):
    games = [Game("same-public-id"), Game("same-public-id")]
    for index, game in enumerate(games):
        game.join()
        game.place(0, make_fleet())
        enemy = make_fleet(column=index * 5)
        enemy[0] = make_fleet()[0]  # A common carrier permits the same unresolved hit.
        game.place(1, enemy)
        game.fire(0, Coordinate(row=0, column=0) if hit else Coordinate(row=9, column=9))
        game.fire(1, Coordinate(row=9, column=9))
        if finished:
            game.resign(1)
    assert games[0].fleets[1] != games[1].fleets[1]
    assert games[0].observe(0).model_dump() == games[1].observe(0).model_dump()


def test_observation_is_detached(game):
    game.fire(0, Coordinate(row=0, column=0))
    observation = game.observe(0)
    observation.target_board[0][0] = "sunk"
    observation.own_fleet[0].cells.clear()
    observation.shot_history[0].sunk_cells.append(Coordinate(row=9, column=9))
    fresh = game.observe(0)
    assert fresh.target_board[0][0] == "hit"
    assert len(fresh.own_fleet[0].cells) == 5
    assert fresh.shot_history[0].sunk_cells == []
