"""Placement contract, artifact isolation, and independently composed players."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

import battleship_client.weights as module
from battleship_client.policies import ComposedPlayer, RandomBot, create_policy, placement_seed, play_policy
from battleship_client.weights import (
    EditsError, PlacementWeightPlayer, WeightPlayer, encode_placement, load_weights,
    parse_edits, placement_cells, placement_mask, save_weights, validate_placement_fleet,
)
from battleship_protocol import Coordinate, FLEET, ShipPlacement
from battleship_server.engine import Game, validate_fleet

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def zero():
    return parse_edits("", role="placement")


def test_shapes_encoding_and_numerical_contract(fleet):
    weights = parse_edits("W1[501, 3]=.5\nW1[0, 3]=.25\nb1[3]=-.1\n"
                          "W2[3, 7]=2\nb2[7]=.3\nW3[7, 199]=-3\nb3[199]=.2", role="placement")
    assert sum(t.size for t in weights.tensors.values()) == 107080
    encoded = encode_placement(fleet[:1], "battleship")
    assert encoded.dtype == np.float64 and encoded.shape == (505,)
    assert encoded.sum() == 6
    for c in range(5):
        assert encoded[c * 5] == 1
    np.testing.assert_array_equal(encoded[500:], [0, 1, 0, 0, 0])
    expected = np.zeros(200)
    expected[199] = -3 * np.tanh(2 * np.tanh(.5 + .25 - .1) + .3) + .2
    np.testing.assert_allclose(PlacementWeightPlayer(weights).logits(fleet[:1], "battleship"), expected)
    for index, ship in enumerate(fleet):
        x = encode_placement(fleet[:index], ship.name)
        assert x[500 + index] == 1 and x[500:].sum() == 1
        for prior_index, prior in enumerate(fleet[:index]):
            for row, column in placement_cells(prior):
                assert x[(row * 10 + column) * 5 + prior_index] == 1


def test_masks_match_authoritative_rules_and_zero_uniformity(fleet, zero):
    for index, name in enumerate(FLEET):
        prefix = fleet[:index]
        mask = placement_mask(prefix, name)
        probs = PlacementWeightPlayer(zero).placement_probabilities(prefix, name)
        assert mask.shape == (200,) and mask.any()
        np.testing.assert_allclose(probs[mask], 1 / mask.sum())
        assert np.all(probs[~mask] == 0)
        occupied = {cell for ship in prefix for cell in placement_cells(ship)}
        for action, legal in enumerate(mask):
            cell, direction = divmod(action, 2)
            row, column = divmod(cell, 10)
            ship = ShipPlacement(name=name, start=Coordinate(row=row, column=column),
                                 orientation="horizontal" if direction == 0 else "vertical")
            cells = placement_cells(ship)
            assert bool(legal) == (all(r < 10 and c < 10 for r, c in cells)
                                   and not occupied.intersection(cells))
    # Adjacent rows are legal, including a horizontal destroyer ending at J10.
    assert placement_mask(fleet[:1], "battleship")[2 * 10]
    assert placement_mask(fleet[:4], "destroyer")[2 * 98]
    assert not placement_mask([], "carrier")[2 * 99]
    assert not placement_mask([], "carrier")[2 * 99 + 1]


def test_many_seeded_fleets_and_public_input_boundary(zero):
    observation = Game("a").observe(0)
    altered = observation.model_copy(update={"match_id": "different", "player": 1,
                                             "incoming_shots": [object()], "target_board": [],
                                             "remaining_enemy_ships": [], "own_fleet": [object()]})
    for seed in range(100):
        a = PlacementWeightPlayer(zero, seed).choose_placement(observation)
        b = PlacementWeightPlayer(zero, seed).choose_placement(altered)
        assert a == b
        assert len(validate_fleet(a)) == 5
        assert len(validate_placement_fleet(a, complete=True)) == 17


def test_placement_lifecycle_and_failure_behavior(game, fleet, zero, monkeypatch):
    player = PlacementWeightPlayer(zero, 1)
    with pytest.raises(ValueError, match="uncommitted"):
        player.choose_placement(game.observe(0))
    with pytest.raises(ValueError, match="order"):
        encode_placement([], "destroyer")
    with pytest.raises(ValueError, match="order"):
        encode_placement(fleet[1:2], "battleship")
    with pytest.raises(ValueError, match="overlap"):
        validate_placement_fleet([fleet[0], fleet[1].model_copy(update={"start": fleet[0].start})])
    monkeypatch.setattr(player, "logits", lambda *args: np.full(200, np.nan))
    with pytest.raises(ValueError, match="logits"):
        player.placement_probabilities([], "carrier")
    monkeypatch.setattr(module, "placement_mask", lambda *args: np.zeros(200, dtype=bool))
    with pytest.raises(ValueError, match="No legal"):
        player.placement_probabilities([], "carrier")


def test_extreme_logits_action_decoding():
    # Kinetic example: select a vertical carrier at J6, index 119.
    weights = parse_edits("b3[:]=-1000000\nb3[119]=1000000", role="placement")
    ships = PlacementWeightPlayer(weights, 1).choose_placement(Game("a").observe(0))
    assert ships[0] == ShipPlacement(name="carrier", start=Coordinate(row=5, column=9), orientation="vertical")
    assert validate_fleet(ships)


def test_role_artifacts_hashes_and_legacy_compatibility(tmp_path, zero):
    artifact = tmp_path / "placement.npz"
    save_weights(artifact, zero)
    restored = load_weights(artifact, role="placement")
    assert restored.content_hash == zero.content_hash
    for array in restored.tensors.values():
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with np.load(artifact, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
    assert manifest == {"artifact_version": "1.1", "role": "placement", "network_version": "placement-1.0",
                        "observation_version": "1.0", "content_hash": zero.content_hash}
    with pytest.raises(ValueError):
        load_weights(artifact)
    with pytest.raises(ValueError):
        PlacementWeightPlayer(parse_edits(""))
    with pytest.raises(ValueError):
        WeightPlayer(zero)
    # Independently reproduce the original shooting v1.0 hash algorithm.
    shooting = parse_edits("b3[0]=1")
    digest = hashlib.sha256(b'{"artifact_version":"1.0","network_version":"1.0","observation_version":"1.0"}\n')
    for name, array in shooting.tensors.items():
        digest.update(json.dumps([name, list(array.shape), "<f8"], separators=(",", ":")).encode() + b"\n")
        digest.update(array.astype("<f8").tobytes())
    assert shooting.content_hash == digest.hexdigest()
    legacy = tmp_path / "shooting.npz"
    save_weights(legacy, shooting)
    with pytest.raises(ValueError):
        load_weights(legacy, role="placement")
    with pytest.raises(FileExistsError):
        save_weights(artifact, zero)
    link = tmp_path / "link.npz"
    link.symlink_to(artifact)
    with pytest.raises(FileExistsError):
        save_weights(link, zero)


def test_placement_archive_corruption(tmp_path, zero):
    path = tmp_path / "bad.npz"
    for corruption in ("hash", "version", "role", "dtype", "shape", "duplicate", "object"):
        arrays = dict(zero.tensors)
        manifest = {**module.NETWORK_SPECS["placement"].versions, "content_hash": zero.content_hash}
        if corruption == "hash":
            manifest["content_hash"] = "0" * 64
        elif corruption == "version":
            manifest["network_version"] = "placement-99"
        elif corruption == "role":
            manifest["role"] = "shooting"
        elif corruption == "dtype":
            arrays["b3"] = np.zeros(200, dtype=np.float32)
        elif corruption == "shape":
            arrays["b3"] = np.zeros(100)
        elif corruption == "object":
            arrays["b3"] = np.array([object()] * 200)
        np.savez(path, **arrays, manifest=np.array(json.dumps(manifest)))
        if corruption == "duplicate":
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("extra.npy", b"extra")
        with pytest.raises(ValueError):
            load_weights(path, role="placement")
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)})
    with pytest.raises(ValueError, match="shape"):
        module._read_npy(header.getvalue(), module.NETWORK_SPECS["placement"])


@pytest.mark.parametrize("edit", ["b3[200]=0", "W1[505,0]=1", "b3[:]=[1]", "import os",
                                  "b3[0]=1000000\nb3[0]+=1", "b3[0]=float(1)"])
def test_placement_parser_uses_same_restrictions(edit):
    with pytest.raises(EditsError):
        parse_edits(edit, role="placement")


def test_import_cli_and_independent_rng(tmp_path, game):
    source = tmp_path / "placement response.txt"
    source.write_text("```edits\nb3[199]=.5\n```")
    artifact = tmp_path / "placement weights.npz"
    result = subprocess.run([sys.executable, "-m", "battleship_client", "weights", "import", str(source),
                             "--role", "placement", "--output", str(artifact)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert load_weights(artifact, role="placement").tensors["b3"][199] == .5
    assert placement_seed(7) != 7 and placement_seed(7) == placement_seed(7)
    assert placement_seed(None) is None
    for kind in ("random", "hunt", "weights"):
        shooting_source = tmp_path / "shooting.edits"
        shooting_source.write_text("")
        path = shooting_source if kind == "weights" else None
        baseline = create_policy(kind, 7, path)
        composed = create_policy(kind, 7, path, artifact)
        assert isinstance(composed, ComposedPlayer)
        # Any number of placement decisions must leave the shot RNG untouched.
        composed.choose_placement(Game("a").observe(0))
        composed.choose_placement(Game("b").observe(0))
        for _ in range(10):
            assert composed.choose_shot(game.observe(0)) == baseline.choose_shot(game.observe(0))


def test_resumed_seat_skips_placement(game):
    class Client:
        def observe(self):
            return game.observe(0)
        def fire(self, shot):
            game.resign(1)
            return game.observe(0)
        def place(self, ships):
            pytest.fail("Committed fleet must not be replaced")
    class Placement:
        def choose_placement(self, observation):
            pytest.fail("Committed seat must not invoke placement inference")
    result = play_policy(Client(), ComposedPlayer(RandomBot(1), Placement()))
    assert result.phase == "finished"


@pytest.mark.parametrize("target", ["bot", "attach-bot", "play", "bots"])
def test_make_placement_paths(target):
    result = subprocess.run(["make", "-n", target, "MATCH=unused", "PLACEMENT_WEIGHTS=first path.npz",
                             "OPPONENT_PLACEMENT_WEIGHTS=other path.edits"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert '--placement-weights "first path.npz"' in result.stdout
    if target == "bots":
        assert '--opponent-placement-weights "other path.edits"' in result.stdout


def test_placement_preflight_before_session(tmp_path):
    result = subprocess.run([sys.executable, "-m", "battleship_client", "--session", str(tmp_path / "absent"),
                             "bot", "random", "--placement-weights", "missing-placement.npz"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1 and "missing-placement.npz" in result.stderr
    assert "absent" not in result.stderr
