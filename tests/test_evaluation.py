"""Evaluation protocol checks; full strength benchmarks are deliberately separate."""

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from battleship_client.weights import load_weight_source, parse_edits
from battleship_protocol import Coordinate
from battleship_server.engine import Game

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


@pytest.fixture(scope="module")
def model():
    return load_weight_source(ROOT / "weights/shooting_attempt_v1_q1.dat")


def test_schedule_streams_and_shared_trials():
    first = evaluation.trial_seeds(42, "benchmark", 0, "random")
    hunt = evaluation.trial_seeds(42, "benchmark", 0, "hunt")
    assert first == evaluation.trial_seeds(42, "benchmark", 0, "random")
    assert first["placement"] == hunt["placement"]
    assert first["candidate"] == hunt["candidate"]
    assert first["opponent"] != hunt["opponent"]
    assert len(set(first.values())) == 3
    assert first != evaluation.trial_seeds(42, "smoke", 0, "random")
    assert first != evaluation.trial_seeds(42, "benchmark", 1, "random")
    # Lock schedule v1.0 against accidental changes.
    assert evaluation.seed_for(42, "benchmark", 0, "placement") == int.from_bytes(
        hashlib.sha256(
            b'["battleship-evaluation-1.0",42,"benchmark",0,"placement"]'
        ).digest()[:8], "big")


def test_pair_preserves_fleets_and_recreates_policies(model, monkeypatch):
    games, instances = [], []
    original_players = evaluation.players

    class TrackedGame(Game):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            games.append(self)

    def tracked_players(*args):
        pair = original_players(*args)
        instances.extend(pair)
        return pair

    monkeypatch.setattr(evaluation, "Game", TrackedGame)
    monkeypatch.setattr(evaluation, "players", tracked_players)
    seeds = evaluation.trial_seeds(42, "benchmark", 3, "hunt")
    a = evaluation.play_local(model, "hunt", seeds, 0)
    b = evaluation.play_local(model, "hunt", seeds, 1)
    assert games[0].fleets[0] == games[1].fleets[1]
    assert games[0].fleets[1] == games[1].fleets[0]
    assert len({id(p) for p in instances}) == 4
    assert instances[0].weights is model and instances[2].weights is model
    assert a["finish_reason"] == b["finish_reason"] == "fleet_destroyed"
    assert evaluation.play_local(model, "hunt", seeds, 0) == a
    assert len(a["history"]) <= 199


def test_policy_boundary_and_illegal_shots(game):
    class Policy:
        calls = 0

        def choose_shot(self, observation):
            self.calls += 1
            assert not hasattr(observation, "fleets")
            return Coordinate(row=9, column=9)

    policy = Policy()
    with pytest.raises(ValueError, match="active turn"):
        evaluation.checked_shot(policy, game.observe(1))
    assert policy.calls == 0
    shot = evaluation.checked_shot(policy, game.observe(0))
    game.fire(0, shot)
    game.fire(1, Coordinate(row=9, column=8))
    with pytest.raises(ValueError, match="illegal"):
        evaluation.checked_shot(policy, game.observe(0))
    game.resign(0)
    with pytest.raises(ValueError, match="active turn"):
        evaluation.checked_shot(policy, game.observe(0))


def test_real_http_parity_and_cleanup(model):
    with evaluation.http_server() as server:
        url = server.url
        for opponent in evaluation.OPPONENTS:
            seeds = evaluation.trial_seeds(42, "smoke", 0, opponent)
            for seat in (0, 1):
                assert server.play(model, opponent, seeds, seat) == evaluation.play_local(
                    model, opponent, seeds, seat)
                assert not server.store.rooms and not server.store.creates
        class Broken:
            def choose_shot(self, observation):
                raise RuntimeError("policy failure")
        from unittest.mock import patch
        with patch.object(evaluation, "players", return_value=(Broken(), Broken())):
            with pytest.raises(RuntimeError, match="policy failure"):
                server.play(model, "random", seeds, 0)
        assert not server.store.rooms and not server.store.creates
    import httpx
    with pytest.raises(httpx.ConnectError):
        httpx.get(url + "/openapi.json", timeout=0.5, trust_env=False)


def test_server_startup_failure_releases_socket(monkeypatch):
    monkeypatch.setattr(evaluation.uvicorn.Server, "run", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="did not start"):
        with evaluation.http_server():
            pytest.fail("Should not yield")


def row(candidate, opponent, trial, seat, won):
    return {"phase": "benchmark", "candidate": candidate, "opponent": opponent,
            "trial": trial, "candidate_seat": seat, "candidate_won": won,
            "candidate_shots": 50, "opponent_shots": 49,
            "candidate_hits": 17, "opponent_hits": 15}


def test_statistics_cluster_pairs_and_align_differences():
    # Every pair has one win: paired CI must collapse to .5, unlike game bootstrap.
    rows = [row(c, o, t, s, s == 0) for c in evaluation.CANDIDATES
            for o in evaluation.OPPONENTS for t in range(4) for s in (0, 1)]
    report = evaluation.summarize(rows, 42, True, "passed")
    assert report["strength"] == "not demonstrated"
    assert report["matchups"]["model_vs_random"]["ci95"] == [0.5, 0.5]
    assert report["model_minus_zero"]["random"]["ci95"] == [0.0, 0.0]
    assert report == evaluation.summarize(rows, 42, True, "passed")
    rows.append(row("model", "random", 4, 0, True))
    report = evaluation.summarize(rows, 42, False, "passed")
    assert report["strength"] == "incomplete"
    assert report["matchups"]["model_vs_random"]["unpaired_games_excluded"] == 1
    assert report["model_minus_zero"]["random"]["aligned_pairs"] == 4
    all_wins = [row("model", "random", t, s, True) for t in range(4) for s in (0, 1)]
    assert evaluation.summarize(all_wins, 42, True, "passed")["strength"] == "passed"
    assert evaluation.bootstrap_ci([], 42) is None
    assert evaluation.bootstrap_ci([0, 0, 0], 42) == [0, 0]


@pytest.mark.parametrize("error", [RuntimeError("broken"), KeyboardInterrupt()])
def test_partial_report_on_failure(tmp_path, model, monkeypatch, error):
    calls = 0
    def failing(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        return {"candidate_won": True, "candidate_seat": 0, "history": [],
                "candidate_shots": 17, "opponent_shots": 16,
                "candidate_hits": 17, "opponent_hits": 0}
    monkeypatch.setattr(evaluation, "play_local", failing)
    output = tmp_path / "run"
    args = SimpleNamespace(output=output, weights=ROOT / "weights/shooting_attempt_v1_q1.dat",
                           pairs=1, seed=42)
    assert evaluation.evaluate(args, model) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["strength"] == summary["status"] == "incomplete"
    assert summary["failure"]["type"] == type(error).__name__
    assert len((output / "games.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_preflight_no_output_for_invalid_weights(tmp_path):
    bad = tmp_path / "bad.edits"
    bad.write_text("invalid edits")
    output = tmp_path / "output"
    assert evaluation.main(["--weights", str(bad), "--output", str(output)]) == 1
    assert not output.exists()
    output.mkdir()
    with pytest.raises(SystemExit):
        evaluation.main(["--output", str(output)])
    with pytest.raises(SystemExit):
        evaluation.main(["--output", str(tmp_path / "new"), "--pairs", "0"])


def test_complete_small_run_and_report_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "SMOKE_PAIRS", 1)
    output = tmp_path / "run"
    assert evaluation.main(["--pairs", "1", "--output", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "complete" and report["http_verification"] == "passed"
    assert report["games_by_phase"] == {"smoke": 8, "http_smoke": 8, "parity": 8,
                                        "benchmark": 8}
    assert all(r["complete_pairs"] == 1 for r in report["matchups"].values())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_sha256"]["scripts/evaluate.py"]
    assert len(manifest["model_content_hash"]) == 64
    assert manifest["zero_content_hash"] == parse_edits("").content_hash
    assert "token" not in (output / "games.jsonl").read_text()
    assert "95% CI" in (output / "summary.md").read_text()


def test_smoke_is_excluded_and_difference_uses_matched_trials():
    records = [row("model", "random", t, s, True) for t in (0, 1) for s in (0, 1)]
    records += [row("zero", "random", 0, s, False) for s in (0, 1)]
    smoke = row("model", "random", 2, 0, False)
    smoke["phase"] = "smoke"
    records.append(smoke)
    result = evaluation.summarize(records, 42, False, "passed")
    assert result["matchups"]["model_vs_random"]["wins"] == 4
    assert result["model_minus_zero"]["random"] == {
        "aligned_pairs": 1, "win_rate_difference": 1.0, "ci95": [1.0, 1.0]}


def test_make_evaluation_quotes_paths():
    import subprocess
    result = subprocess.run(
        ["make", "-n", "evaluate", "OUTPUT=report folder", "EVAL_WEIGHTS=weight file.dat",
         "PAIRS=10", "SEED=7"], cwd=ROOT, capture_output=True, text=True, check=True)
    assert '--weights "weight file.dat"' in result.stdout
    assert '--output "report folder"' in result.stdout
    assert '--pairs "10" --seed "7"' in result.stdout


def test_hunt_only_summary_has_no_random_claim_or_control():
    rows = [row('model', 'hunt', t, s, s == 0) for t in range(3) for s in (0, 1)]
    report = evaluation.summarize(rows, 42, True, 'passed', ('hunt',), ('model',))
    assert set(report['matchups']) == {'model_vs_hunt'}
    assert report['strength'] == 'not evaluated'
    assert report['model_minus_zero'] == {}
    assert 'Model minus zero' not in evaluation.markdown(report)
    partial = evaluation.summarize(rows, 42, False, 'failed', ('hunt',), ('model',))
    assert partial['strength'] == 'incomplete'


def test_hunt_only_run_and_unique_automatic_outputs(tmp_path, monkeypatch, model):
    monkeypatch.setattr(evaluation, 'SMOKE_PAIRS', 1)
    monkeypatch.setattr(evaluation, 'ROOT', tmp_path)
    monkeypatch.setattr(evaluation, 'source_fingerprints', lambda: {})
    monkeypatch.setattr(evaluation, 'load_weight_source', lambda path: model)
    command = ['--weights', 'weight path.dat', '--opponents', 'hunt', '--model-only', '--pairs', '1']
    assert evaluation.main(command) == 0
    first = list((tmp_path / '.battleship/evaluations').iterdir())
    assert len(first) == 1
    output = first[0]
    report = json.loads((output / 'summary.json').read_text())
    assert report['status'] == 'complete' and report['strength'] == 'not evaluated'
    assert report['games_by_phase'] == {'smoke': 2, 'http_smoke': 2, 'parity': 2, 'benchmark': 2}
    assert set(report['matchups']) == {'model_vs_hunt'}
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['configuration']['opponents'] == ['hunt']
    assert manifest['configuration']['candidates'] == ['model']
    assert manifest['zero_content_hash'] is None
    # A second invocation must never overwrite the first report.
    assert evaluation.main(command) == 0
    assert len(list((tmp_path / '.battleship/evaluations').iterdir())) == 2


def test_make_hunt_accepts_weight_only_and_quotes_paths():
    import subprocess
    result = subprocess.run(
        ['make', '-n', 'evaluate-hunt', 'WEIGHTS=weight folder/model.dat'],
        cwd=ROOT, capture_output=True, text=True, check=True)
    command = result.stdout.splitlines()[-1]
    assert '--weights "weight folder/model.dat"' in command
    assert '--opponents hunt --model-only' in command
    assert '--pairs "1000" --seed "42"' in command
    assert '--output' not in command
    override = subprocess.run(
        ['make', '-n', 'evaluate-hunt', 'WEIGHTS=weight.dat', 'OUTPUT=report folder', 'PAIRS=10'],
        cwd=ROOT, capture_output=True, text=True, check=True)
    assert '--output "report folder"' in override.stdout
    assert '--pairs "10"' in override.stdout
