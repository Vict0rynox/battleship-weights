"""Synthetic networks exercise the contract without any weight generation."""

import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest

import battleship_client.weights as weights_module
from battleship_client.weights import (
    EditsError, NetworkWeights, TENSOR_SHAPES, WeightPlayer, encode_observation,
    load_weight_source, load_weights, parse_edits, save_weights,
)


def test_documented_edits_and_order():
    weights = parse_edits("""# all forms in the contract
W1[12, 0] = 2.5
W1[0:4, 3] = [1, 0, -1, 0]
b1[0:4] = -1
W1[3:400:40, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
W3[:, 2] *= 2
b3[0:3] = [
  +1, # signed decimal
  .5,
  -2.,
]
b3[:3] += [1, 2, 3]
b3[0:3:2] -= 1
b3[1] *= -2
b3[2] = 9
""")
    w = weights.tensors
    assert sum(array.size for array in w.values()) == 81380
    assert all(array.dtype == np.float64 for array in w.values())
    assert w["W1"][12, 0] == 2.5
    np.testing.assert_array_equal(w["W1"][:4, 3], [1, 0, -1, 0])
    np.testing.assert_array_equal(w["b1"][:4], [-1] * 4)
    np.testing.assert_array_equal(w["W2"][:2, :3], [[1, -1, .5], [0, 2, 0]])
    np.testing.assert_array_equal(w["b3"][:3], [1, -5, 9])
    assert w["W1"][323, 0] == w["W1"][363, 0] == 1
    assert not w["W3"].any()


def test_full_response_reads_only_last_edits_and_reports_original_location():
    text = "explanation\n```edits\ninvalid earlier block\n```\ntext\n```edits\nb3[4] = 7\n```\nend"
    assert parse_edits(text).tensors["b3"][4] == 7
    with pytest.raises(EditsError, match="line 7, column"):
        parse_edits(text.replace("b3[4] = 7", "b3[100] = 7"))
    with pytest.raises(EditsError, match="unclosed"):
        parse_edits(text + "\n```edits\nb3[0] = 1")
    with pytest.raises(EditsError, match="no closed edits"):
        parse_edits("```python\nb3[0] = 1\n```")


@pytest.mark.parametrize("edit", [
    "import os", "__import__('os')", "W4[0] = 1", "b3 = 1", "b3[0] = 1; b3[1] = 2",
    "b3[0] = 1 + 2", "b3[0] = 1e2", "b3[0] = NaN", "b3[0] = inf", "b3[0] = float(1)",
    "b3[0] = True", "b3[0] = [x for x in []]", "b3[0] = (1)", "b3[0] /= 2",
    "b3[-1] = 1", "b3[100] = 1", "b3[0.0] = 1", "b3[0:101] = 1", "b3[-1:4] = 1",
    "b3[2:2] = 1", "b3[4:1] = 1", "b3[::0] = 1", "b3[::-1] = 1", "b3[100:] = 1",
    "b3[0, 1] = 1", "W1[0] = 1", "b3[0:3] = [1]", "b3[0] = [1]", "b3[:] = []",
    "W1[:2, :2] = [1, 2]", "W1[:2, :2] = [[1, 2], [3]]", "b3[0] = [[[1]]]",
    "b3[0] = 1000001", "b3[0] = -1000001", "b3[0] = 1000000.00000000001",
    "b3[0] = 1000000\nb3[0] += 1\nb3[0] = 0",
    "b3[0] = 1000000\nb3[0] *= 2", "b3[0] = -1000000\nb3[0] -= 1",
    "b3[999999999999999999999999] = 0", "b3[0] = 1\ntext after raw edits",
])
def test_invalid_edits_are_rejected_atomically(edit):
    baseline = parse_edits("b3[1] = 7")
    before = baseline.content_hash
    with pytest.raises(EditsError, match="line"):
        parse_edits("b3[1] = 5\n" + edit)
    assert baseline.content_hash == before
    assert not parse_edits("").tensors["b3"].any()


def test_import_resource_limits(monkeypatch, tmp_path):
    monkeypatch.setattr(weights_module, "MAX_TEXT_BYTES", 32)
    with pytest.raises(EditsError, match="exceeds"):
        parse_edits("#" * 33)
    with pytest.raises(EditsError, match="exceeds"):
        parse_edits("#" + "é" * 16)
    path = tmp_path / "large.edits"
    path.write_bytes(b"#" * 33)
    with pytest.raises(EditsError, match="exceeds"):
        load_weight_source(path)
    monkeypatch.setattr(weights_module, "MAX_STATEMENTS", 1)
    with pytest.raises(EditsError, match="too many statements"):
        parse_edits("b3[0]=1\nb3[1]=1")


def test_large_positive_slice_steps_and_leading_zero_indexes():
    weights = parse_edits("b3[0:4:9999999999999999999999999]=2\nb3[0000000000000000000000001]=3")
    np.testing.assert_array_equal(weights.tensors["b3"][:4], [2, 3, 0, 0])


def test_weights_are_owned_immutable_and_validated():
    arrays = {name: np.zeros(shape, dtype=np.float64) for name, shape in TENSOR_SHAPES.items()}
    weights = NetworkWeights(arrays)
    arrays["b3"][0] = 10
    assert weights.tensors["b3"][0] == 0
    with pytest.raises(ValueError):
        weights.tensors["b3"].setflags(write=True)
    with pytest.raises(TypeError):
        weights.tensors["b3"] = arrays["b3"]
    arrays["b3"] = arrays["b3"].astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        NetworkWeights(arrays)


def test_artifact_round_trip_hash_and_no_overwrite(tmp_path):
    weights = parse_edits("W1[3,4]=.3\nb3[0]=1000000")
    same = parse_edits("b3[0]=1000000\nW1[3,4]=.3")
    assert weights.content_hash == same.content_hash
    assert weights.content_hash != parse_edits("").content_hash
    path = tmp_path / "weights.npz"
    save_weights(path, weights)
    restored = load_weights(path)
    assert restored.content_hash == weights.content_hash
    for name in TENSOR_SHAPES:
        np.testing.assert_array_equal(restored.tensors[name], weights.tensors[name])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_weights(path, parse_edits(""))
    assert path.read_bytes() == original
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(FileExistsError):
        save_weights(link, weights)
    assert not list(tmp_path.glob(".weights-*"))


def test_failed_save_never_publishes_partial_artifact(tmp_path, monkeypatch):
    def fail(output, **arrays):
        output.write(b"partial artifact")
        raise OSError("disk write failed")
    monkeypatch.setattr(weights_module.np, "savez", fail)
    destination = tmp_path / "missing.npz"
    with pytest.raises(OSError, match="disk write failed"):
        save_weights(destination, parse_edits(""))
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def artifact_arrays(tmp_path):
    path = tmp_path / "valid.npz"
    save_weights(path, parse_edits("b3[0]=3"))
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@pytest.mark.parametrize("case", [
    "hash", "artifact_version", "network_version", "observation_version", "extra_metadata",
    "manifest_type", "manifest_shape", "extra_tensor", "missing_tensor", "shape", "dtype",
    "object", "nan", "inf", "bound",
])
def test_reject_invalid_artifacts(tmp_path, case):
    arrays = artifact_arrays(tmp_path)
    manifest = json.loads(str(arrays["manifest"]))
    if case == "hash":
        arrays["b3"][0] += 1
    elif case.endswith("version"):
        manifest[case] = "99.0"
    elif case == "extra_metadata":
        manifest["unexpected"] = 1
    elif case == "manifest_type":
        arrays["manifest"] = np.array(1)
    elif case == "manifest_shape":
        arrays["manifest"] = np.array(["text"])
    elif case == "extra_tensor":
        arrays["surprise"] = np.array(1)
    elif case == "missing_tensor":
        del arrays["W1"]
    elif case == "shape":
        arrays["W1"] = np.zeros((128, 405))
    elif case == "dtype":
        arrays["W1"] = arrays["W1"].astype(np.float32)
    elif case == "object":
        arrays["W1"] = np.empty((405, 128), dtype=object)
    else:
        arrays["b3"][0] = {"nan": np.nan, "inf": np.inf, "bound": 1000001}[case]
    if case not in ("manifest_type", "manifest_shape"):
        arrays["manifest"] = np.array(json.dumps(manifest))
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_weights(path)


def test_archive_limits_corruption_and_header_claims(tmp_path):
    path = tmp_path / "bad.npz"
    path.write_text("b3[0]=1")
    with pytest.raises(ValueError):
        load_weight_source(path)  # no fallback to valid edits
    path.write_bytes(b"0" * (weights_module.MAX_ARCHIVE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        load_weights(path)
    # Small compressed archive, oversized expanded payload.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (*TENSOR_SHAPES, "manifest"):
            archive.writestr(name + ".npy", b"0" * 310000)
    with pytest.raises(ValueError, match="Expanded"):
        load_weights(path)
    # A header claiming a huge allocation must fail before np.load allocates it.
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        "descr": "<f8", "fortran_order": False, "shape": (10**12,),
    })
    with pytest.raises(ValueError, match="shape"):
        weights_module._read_npy(header.getvalue())
    with pytest.raises(ValueError, match="payload"):
        weights_module._read_npy(header.getvalue().replace(b"1000000000000", b"100          "))


def test_duplicate_members_and_truncated_payload(tmp_path):
    good = tmp_path / "good.npz"
    save_weights(good, parse_edits(""))
    bad = tmp_path / "bad.npz"
    with zipfile.ZipFile(good) as source, zipfile.ZipFile(bad, "w") as target:
        for item in source.infolist():
            data = source.read(item)
            target.writestr(item.filename, data[:-1] if item.filename == "b3.npy" else data)
    with pytest.raises(ValueError, match="payload"):
        load_weights(bad)
    with zipfile.ZipFile(good) as source, zipfile.ZipFile(bad, "w") as target:
        for item in source.infolist():
            target.writestr(item.filename, source.read(item))
        with pytest.warns(UserWarning, match="Duplicate"):
            target.writestr("b3.npy", source.read("b3.npy"))
    with pytest.raises(ValueError, match="duplicate"):
        load_weights(bad)


def test_encoder_all_cells_channels_and_fleet_flags(game):
    observation = game.observe(0)
    states = ("unknown", "miss", "hit", "sunk")
    for r in range(10):
        for c in range(10):
            observation.target_board[r][c] = states[(r * 10 + c) % 4]
    observation.remaining_enemy_ships[:] = ["cruiser", "destroyer"]
    encoded = encode_observation(observation)
    assert encoded.shape == (405,) and encoded.dtype == np.float64
    for index in range(100):
        np.testing.assert_array_equal(encoded[index * 4:index * 4 + 4], np.eye(4)[index % 4])
    np.testing.assert_array_equal(encoded[400:], [0, 0, 1, 0, 1])
    changed = observation.model_copy(update={"own_fleet": [], "incoming_shots": [], "shot_history": [],
                                              "match_id": "other", "revision": 999})
    np.testing.assert_array_equal(encoded, encode_observation(changed))
    with pytest.raises(ValueError, match="version"):
        encode_observation(observation.model_copy(update={"observation_version": "99"}))


def test_encoder_tracks_sinking_disclosures(game):
    from battleship_protocol import Coordinate
    game.fire(0, Coordinate(row=4, column=0))
    hit = encode_observation(game.observe(0))
    assert hit[(4 * 10) * 4 + 2] == hit[404] == 1
    game.fire(1, Coordinate(row=9, column=9))
    game.fire(0, Coordinate(row=4, column=1))
    sunk = encode_observation(game.observe(0))
    for cell in (40, 41):
        np.testing.assert_array_equal(sunk[cell * 4:cell * 4 + 4], [0, 0, 0, 1])
    np.testing.assert_array_equal(sunk[400:], [1, 1, 1, 1, 0])


def test_synthetic_network_orientation_and_public_view_only(game):
    observation = game.observe(0)
    weights = parse_edits("""W1[0, 3]=.5
W1[400, 3]=.25
b1[3]=-.1
W2[3, 7]=2
b2[7]=.3
W3[7, 42]=-3
b3[42]=.2
""")
    player = WeightPlayer(weights, 1)
    expected = np.zeros(100)
    expected[42] = -3 * np.tanh(2 * np.tanh(.5 + .25 - .1) + .3) + .2
    np.testing.assert_allclose(player.logits(observation), expected, atol=1e-14)
    altered = observation.model_copy(update={"own_fleet": [], "incoming_shots": [], "shot_history": []})
    np.testing.assert_array_equal(player.logits(observation), player.logits(altered))


def test_probabilities_mask_stability_and_repeatability(game):
    observation = game.observe(0)
    observation.legal_shot_mask[0][0] = False
    zero = parse_edits("")
    player = WeightPlayer(zero, 11)
    probabilities = player.shot_probabilities(observation)
    assert probabilities[0] == 0
    np.testing.assert_allclose(probabilities[1:], 1 / 99)
    first, second = WeightPlayer(zero, 91), WeightPlayer(zero, 91)
    for _ in range(40):
        shot = first.choose_shot(observation)
        assert shot == second.choose_shot(observation)
        assert observation.legal_shot_mask[shot.row][shot.column]
    extreme = WeightPlayer(parse_edits("b3[:]= -1000000\nb3[0]=1000000\nb3[42]=1000000"), 1)
    assert extreme.shot_probabilities(observation)[42] == 1
    assert extreme.choose_shot(observation).row == 4


def test_inactive_empty_mask_and_nonfinite_failures(game, monkeypatch):
    player = WeightPlayer(parse_edits(""), 1)
    observation = game.observe(0)
    for phase in ("waiting", "placement", "finished"):
        with pytest.raises(ValueError, match="active"):
            player.choose_shot(observation.model_copy(update={"phase": phase}))
        with pytest.raises(ValueError, match="active"):
            player.logits(observation.model_copy(update={"phase": phase}))
    with pytest.raises(ValueError, match="active"):
        player.choose_shot(game.observe(1))
    with pytest.raises(ValueError, match="legal"):
        player.choose_shot(observation.model_copy(update={"legal_shot_mask": [[False] * 10 for _ in range(10)]}))
    monkeypatch.setattr(player, "logits", lambda _: np.full(100, np.nan))
    with pytest.raises(ValueError, match="logits"):
        player.choose_shot(observation)


def test_text_source_and_import_without_session(tmp_path):
    source = tmp_path / "synthetic.edits"
    source.write_text("b3[2]=.5\n")
    target = tmp_path / "model.npz"
    result = subprocess.run([sys.executable, "-m", "battleship_client", "--session", str(tmp_path / "absent"),
                             "weights", "import", str(source), "--output", str(target)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert load_weights(target).content_hash == load_weight_source(source).content_hash
    assert "SHA-256" in result.stdout
