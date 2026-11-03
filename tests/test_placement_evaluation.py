"""Defensive measurement, aligned trials, failures, and real HTTP integration."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from battleship_client.weights import parse_edits
from battleship_protocol import Coordinate
from battleship_server.engine import validate_fleet

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("placement_evaluation", ROOT / "scripts/evaluate_placement.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_rollouts_exact_counts_and_hundredth_shot(monkeypatch):
    fleet = evaluation.fixture_fleet()
    occupied = {cell for cells in validate_fleet(fleet).values() for cell in cells}
    water = [Coordinate(row=r, column=c) for r in range(10) for c in range(10) if (r, c) not in occupied]
    hits = [Coordinate(row=r, column=c) for r, c in sorted(occupied)]
    class Script:
        def __init__(self, sequence):
            self.sequence = iter(sequence)
        def choose_shot(self, observation):
            return next(self.sequence)
    for shots, count in ((hits + water, 17), (water + hits, 100)):
        monkeypatch.setattr(evaluation, "attacker_policy", lambda *args: Script(shots))
        result = evaluation.play_local(fleet, "random", 1)
        assert result["attacker_shots"] == count
        assert len(result["history"]) == count * 2 - 1
        assert sum(e["attacker"] == 0 and e["outcome"] != "miss" for e in result["history"]) == 17
        assert result["history"][-1]["outcome"] == "sunk"


def test_fleet_generation_shared_trials_and_attacker_seed_alignment():
    model = parse_edits("b3[0]=.1", role="placement")
    seeds = evaluation.trial_seeds(42, "benchmark", 0, ("random", "hunt", "weights"))
    assert seeds == evaluation.trial_seeds(42, "benchmark", 0, ("random", "hunt", "weights"))
    assert len(set([seeds["fleet"], *seeds["attackers"].values()])) == 4
    assert seeds != evaluation.trial_seeds(42, "smoke", 0, ("random", "hunt", "weights"))
    for candidate in evaluation.CANDIDATES:
        a = evaluation.generate_fleet(candidate, model, seeds["fleet"])
        assert a == evaluation.generate_fleet(candidate, model, seeds["fleet"])
        assert len(validate_fleet(a)) == 5
        for attacker in ("random", "hunt"):
            first = evaluation.play_local(a, attacker, seeds["attackers"][attacker])
            assert 17 <= first["attacker_shots"] <= 100
            assert evaluation.play_local(a, attacker, seeds["attackers"][attacker]) == first


def rows(trial, offset=0, phase="benchmark"):
    return [{"phase": phase, "trial": trial, "candidate": candidate, "attacker": attacker,
             "attacker_shots": 60 + (offset if candidate == "model" else 0)}
            for candidate in evaluation.CANDIDATES for attacker in ("random", "hunt")]


def test_statistics_pair_whole_trials_and_exclude_incomplete_blocks():
    records = rows(0, 10) + rows(1, 10) + rows(2, -50, "smoke") + rows(3, 0)[:-1]
    report = evaluation.summarize(records, 42, ("random", "hunt"), False, "passed")
    assert report["complete_trials"] == 2 and report["incomplete_trials_excluded"] == 1
    assert report["matchups"]["model_vs_random"]["mean_shots"] == 70
    assert report["matchups"]["model_vs_random"]["median_ci95"] == [70, 70]
    for difference in report["differences"].values():
        assert difference["aligned_trials"] == 2
        assert difference["mean_shot_difference"] == 10
        assert difference["ci95"] == [10, 10]
        assert difference["percentage_improvement"] == pytest.approx(100 / 6)
        assert difference["percentage_ci95"] == pytest.approx([100 / 6, 100 / 6])
    assert report == evaluation.summarize(records, 42, ("random", "hunt"), False, "passed")
    empty = evaluation.summarize([], 42, ("random", "hunt"), False, "not run")
    assert empty["matchups"]["automatic_vs_hunt"]["mean_shots"] is None


def test_placement_real_http_parity_and_cleanup():
    model = parse_edits("", role="placement")
    shooting = parse_edits("")
    with evaluation.match_eval.http_server() as server:
        for candidate in evaluation.CANDIDATES:
            ships = evaluation.generate_fleet(candidate, model, 42)
            for attacker in ("random", "hunt", "weights"):
                assert evaluation.play_http(server, ships, attacker, 7, shooting) == evaluation.play_local(
                    ships, attacker, 7, shooting)
                assert not server.store.rooms and not server.store.creates
        class Broken:
            def choose_shot(self, observation):
                raise RuntimeError("broken")
        from unittest.mock import patch
        with patch.object(evaluation, "attacker_policy", return_value=Broken()):
            with pytest.raises(RuntimeError, match="broken"):
                evaluation.play_http(server, ships, "random", 7)
        assert not server.store.rooms and not server.store.creates


def test_combined_evaluator_seat_pair_fleets_and_parity(monkeypatch):
    match = evaluation.match_eval
    shooting = parse_edits("b3[0]=.1")
    placement = parse_edits("b3[199]=.5", role="placement")
    seeds = match.trial_seeds(42, "smoke", 0, "hunt", schedule=match.TWO_MODEL_SCHEDULE)
    assert seeds != match.trial_seeds(42, "smoke", 0, "hunt")
    games = []
    original = match.Game
    class Tracked(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            games.append(self)
    monkeypatch.setattr(match, "Game", Tracked)
    first = match.play_local(shooting, "hunt", seeds, 0, placement)
    match.play_local(parse_edits(""), "hunt", seeds, 1, placement)
    assert games[0].fleets[0] == games[1].fleets[1]
    assert games[0].fleets[1] == games[1].fleets[0]
    with match.http_server() as server:
        assert server.play(shooting, "hunt", seeds, 0, placement) == first


@pytest.mark.parametrize("error", [RuntimeError("broken"), KeyboardInterrupt()])
def test_partial_report_on_rollout_failure(tmp_path, monkeypatch, error):
    calls = 0
    def failing(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        return {"attacker_shots": 60, "history": []}
    monkeypatch.setattr(evaluation, "SMOKE_TRIALS", 0)
    monkeypatch.setattr(evaluation, "play_local", failing)
    args = SimpleNamespace(output=tmp_path / "run", placement_weights=tmp_path / "source.edits", trials=1, seed=42)
    assert evaluation.evaluate(args, parse_edits("", role="placement")) == 1
    report = json.loads((args.output / "summary.json").read_text())
    assert report["status"] == "incomplete"
    assert report["failure"]["type"] == type(error).__name__
    assert report["complete_trials"] == 0 and report["incomplete_trials_excluded"] == 1
    assert json.loads((args.output / "manifest.json").read_text())["status"] == "incomplete"
    assert len((args.output / "games.jsonl").read_text().splitlines()) == 1


def test_small_complete_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "SMOKE_TRIALS", 1)
    placement, shooting = tmp_path / "placement.edits", tmp_path / "shooting.edits"
    placement.write_text("b3[199]=.5")
    shooting.write_text("")
    output = tmp_path / "report"
    assert evaluation.main(["--placement-weights", str(placement), "--attacker-weights", str(shooting),
                            "--trials", "1", "--strategies", "zero", "automatic",
                            "--attackers", "random", "hunt", "weights", "--output", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "complete" and report["http_verification"] == "passed"
    assert report["complete_trials"] == 1
    assert report["rollouts_by_phase"] == {"smoke": 9, "http_smoke": 9, "benchmark": 9}
    assert len(report["matchups"]) == 9
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["content_hashes"]["model"] == parse_edits(placement.read_text(), role="placement").content_hash
    assert manifest["attacker_content_hash"] == parse_edits("").content_hash
    assert manifest["source_sha256"]["scripts/evaluate_placement.py"]
    records = [json.loads(line) for line in (output / "games.jsonl").read_text().splitlines()]
    benchmark = [r for r in records if r["phase"] == "benchmark"]
    for candidate in manifest["configuration"]["candidates"]:
        group = [r for r in benchmark if r["candidate"] == candidate]
        assert len({r["fleet_hash"] for r in group}) == 1
    assert len({r["seeds"]["attackers"]["hunt"] for r in benchmark}) == 1
    assert "95% CI" in (output / "summary.md").read_text()
    assert "token" not in (output / "games.jsonl").read_text()


def test_preflight_and_make_paths(tmp_path):
    output = tmp_path / "report"
    assert evaluation.main(["--placement-weights", str(tmp_path / "absent.npz"), "--output", str(output)]) == 1
    assert not output.exists()
    with pytest.raises(SystemExit):
        evaluation.main(["--placement-weights", "unused", "--trials", "0"])
    output.mkdir()
    with pytest.raises(SystemExit):
        evaluation.main(["--placement-weights", "unused", "--output", str(output)])
    import subprocess
    result = subprocess.run(["make", "-n", "evaluate-placement", "PLACEMENT_WEIGHTS=placement folder/model.npz",
                             "ATTACKER_WEIGHTS=shooting folder/model.npz", "OUTPUT=report folder"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert '--placement-weights "placement folder/model.npz"' in result.stdout
    assert '--attacker-weights "shooting folder/model.npz"' in result.stdout
    assert '--output "report folder"' in result.stdout
    assert '--trials "1000"' in result.stdout


@pytest.mark.parametrize("selection,has_weights,expected", [
    (None, False, ("hunt",)), (None, True, ("hunt", "weights")),
    (("weights",), True, ("weights",)), (("random", "hunt", "random"), False, ("hunt", "random")),
])
def test_attacker_selection(selection, has_weights, expected):
    assert evaluation.selected_attackers(selection, has_weights=has_weights) == expected


@pytest.mark.parametrize("arguments", [
    ["--attackers", "weights"],
    ["--attackers", "hunt", "--attacker-weights", "missing.npz"],
])
def test_invalid_attacker_selection_precedes_weights_and_output(tmp_path, capsys, arguments):
    output = tmp_path / "report"
    assert evaluation.main(["--placement-weights", "missing-placement.npz", "--output", str(output),
                            *arguments]) == 1
    assert "attacker-weights" in capsys.readouterr().err
    assert not output.exists()


def test_unknown_and_empty_selections_fail():
    for strategies in ((), ("unknown",)):
        with pytest.raises(ValueError):
            evaluation.selected_candidates(strategies)
    for attackers in ((), ("unknown",)):
        with pytest.raises(ValueError):
            evaluation.selected_attackers(attackers)
    with pytest.raises(ValueError, match="Unknown placement"):
        evaluation.generate_fleet("unknown", None, 42)


def test_shared_generators_geometry_and_exhaustion(monkeypatch):
    import random
    import placement_strategies as strategies
    from battleship_server.engine import placement_cells
    for seed in range(30):
        for name in ("automatic", "edge", "center", "touching"):
            fleet = evaluation.generate_fleet(name, None, seed)
            cells = {cell for ship in fleet for cell in placement_cells(ship)}
            assert len(cells) == 17
            if name == "edge":
                assert all(r < 2 or r >= 8 or c < 2 or c >= 8 for r, c in cells)
            if name == "center":
                assert all(2 <= r < 8 and 2 <= c < 8 for r, c in cells)
            if name == "touching":
                occupied = set()
                for ship in fleet:
                    current = placement_cells(ship)
                    if occupied:
                        assert any((r + dr, c + dc) in occupied for r, c in current
                                   for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)))
                    occupied.update(current)
    with pytest.raises(ValueError, match="exhausted retries"):
        strategies.draw_fleet("edge", random.Random(42), max_attempts=0)
    with pytest.raises(ValueError, match="Unknown placement"):
        strategies.draw_fleet("unknown", random.Random(42))


def test_primary_comparison_and_percentiles():
    candidates = ("model", "automatic")
    def data(offset):
        return [{"phase": "benchmark", "trial": t, "candidate": c, "attacker": "hunt",
                 "attacker_shots": 40 + 10 * t + (offset if c == "model" else 0)}
                for t in range(3) for c in candidates]
    for offset, status in ((10, "improved"), (-10, "worse"), (0, "inconclusive")):
        report = evaluation.summarize(data(offset), 42, ("hunt",), True, "passed", candidates=candidates)
        assert report["primary_comparison"]["status"] == status
        model = report["matchups"]["model_vs_hunt"]
        assert model["p10_shots"] == pytest.approx(42 + offset)
        assert model["p90_shots"] == pytest.approx(58 + offset)
        assert model["median_shots"] == 50 + offset
        diff = report["differences"]["model_minus_automatic_vs_hunt"]
        assert diff["percentage_improvement"] == pytest.approx(100 * offset / 50)
    partial = evaluation.summarize(data(10), 42, ("hunt",), False, "passed", candidates=candidates)
    assert partial["primary_comparison"]["status"] == "incomplete"
    missing = evaluation.summarize([], 42, ("random",), True, "passed", candidates=candidates)
    assert missing["primary_comparison"]["status"] == "not evaluated"


def test_strategy_subset_and_added_attacker_do_not_change_existing_sequences(tmp_path, monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(evaluation, "SMOKE_TRIALS", 0)
    monkeypatch.setattr(evaluation.match_eval, "http_server", lambda: nullcontext(object()))
    model = parse_edits("", role="placement")
    runs = []
    for index, (strategies, attackers) in enumerate([
            (evaluation.DEFAULT_STRATEGIES, ("hunt",)),
            (("edge", "automatic", "zero"), ("weights", "hunt"))]):
        args = SimpleNamespace(output=tmp_path / str(index), placement_weights=tmp_path / "source.edits",
                               strategies=strategies, attackers=attackers, trials=2, seed=42)
        assert evaluation.evaluate(args, model, parse_edits("") if "weights" in attackers else None) == 0
        runs.append([json.loads(line) for line in (args.output / "games.jsonl").read_text().splitlines()])
    common = ("model", "automatic", "edge")
    original = [r for r in runs[0] if r["candidate"] in common]
    subset = [r for r in runs[1] if r["candidate"] in common and r["attacker"] == "hunt"]
    for first, second in zip(original, subset):
        assert first["fleet_hash"] == second["fleet_hash"]
        assert first["history"] == second["history"]
        assert first["seeds"]["attackers"]["hunt"] == second["seeds"]["attackers"]["hunt"]
    # Equivalent model and zero weights with a shared placement stream must match.
    for trial in (0, 1):
        group = {r["candidate"]: r for r in runs[1] if r["trial"] == trial and r["attacker"] == "hunt"}
        assert group["model"]["fleet_hash"] == group["zero"]["fleet_hash"]
        assert group["model"]["history"] == group["zero"]["history"]


def test_repeated_layouts_are_kept(tmp_path, monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(evaluation, "SMOKE_TRIALS", 0)
    monkeypatch.setattr(evaluation.match_eval, "http_server", lambda: nullcontext(object()))
    monkeypatch.setattr(evaluation, "generate_fleet", lambda *args: evaluation.fixture_fleet())
    monkeypatch.setattr(evaluation, "play_local", lambda *args: {"attacker_shots": 60, "history": []})
    args = SimpleNamespace(output=tmp_path / "report", placement_weights=tmp_path / "source.edits",
                           strategies=("automatic",), trials=3, seed=42)
    assert evaluation.evaluate(args, parse_edits("", role="placement")) == 0
    report = json.loads((args.output / "summary.json").read_text())
    assert report["complete_trials"] == 3
    records = [json.loads(line) for line in (args.output / "games.jsonl").read_text().splitlines()]
    assert len(records) == 6
    assert len({r["fleet_hash"] for r in records}) == 1


def test_make_strategy_and_attacker_selection():
    import subprocess
    result = subprocess.run(["make", "-n", "evaluate-placement", "PLACEMENT_WEIGHTS=placement path.dat",
                             "STRATEGIES=edge center", "ATTACKERS=hunt weights", "ATTACKER_WEIGHTS=shooting path.dat"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert '--strategies "edge" "center"' in result.stdout
    assert '--attackers "hunt" "weights"' in result.stdout
    assert '--placement-weights "placement path.dat"' in result.stdout
    assert '--attacker-weights "shooting path.dat"' in result.stdout
