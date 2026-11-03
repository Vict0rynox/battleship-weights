"""Target-practice invariants and benchmark accounting, without network access."""

import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from battleship_client.policies import RandomBot
from battleship_client.weights import WeightPlayer, load_weight_source
from battleship_protocol import Coordinate, ShipPlacement
from battleship_server.engine import Game, GameError, placement_cells, random_fleet, validate_fleet

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shooting_bench', ROOT / 'scripts/shooting_bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.fixture(scope='module')
def weights():
    return load_weight_source(ROOT / 'weights/shooting_attempt_v1_q2.dat')


class SequencePolicy:
    def __init__(self, cells):
        self.cells = iter(cells)

    def choose_shot(self, observation):
        assert observation.phase == 'playing' and observation.turn == observation.player == 0
        assert observation.incoming_shots == []
        assert not hasattr(observation, 'fleets')
        r, c = next(self.cells)
        return Coordinate(row=r, column=c)


def ships_from_layout(layout):
    return [ShipPlacement.model_validate(ship) for ship in layout['ships']]


def test_seed_and_allocation_contract():
    import hashlib
    expected = int.from_bytes(hashlib.sha256(
        b'["battleship-shooting-1.0",42,"edge",0,"placement"]'
    ).digest()[:8], 'big')
    assert bench.seed_for(42, 'edge', 0, 'placement') == expected
    assert expected != bench.seed_for(42, 'edge', 0, 'policy', 'model')
    assert bench.seed_for(42, 'edge', 0, 'policy', 'model') != bench.seed_for(
        42, 'edge', 0, 'policy', 'hunt')
    assert list(bench.board_specs(6, bench.FAMILIES)) == [
        ('random', 0), ('random', 1), ('edge', 0), ('edge', 1), ('center', 0), ('touching', 0)]


def test_placement_geometry_diversity_and_replay():
    seen = set()
    saved = []
    for family in bench.FAMILIES:
        for index in range(25):
            layout = bench.generate_layout(42, family, index, seen)
            saved.append(layout)
            ships = ships_from_layout(layout)
            fleet = validate_fleet(ships)
            cells = {cell for ship_cells in fleet.values() for cell in ship_cells}
            assert len(cells) == 17
            if family == 'edge':
                assert all(r < 2 or r >= 8 or c < 2 or c >= 8 for r, c in cells)
            if family == 'center':
                assert all(2 <= r < 8 and 2 <= c < 8 for r, c in cells)
            if family == 'touching':
                occupied = set()
                for ship in ships:
                    current = placement_cells(ship)
                    if occupied:
                        assert any((r+dr, c+dc) in occupied for r, c in current
                                   for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)))
                    occupied.update(current)
    assert len(seen) == 100
    replay_seen = set()
    assert [bench.generate_layout(42, family, index, replay_seen)
            for family in bench.FAMILIES for index in range(25)] == saved
    expected = random_fleet(random.Random(bench.seed_for(42, 'random', 0, 'placement')))
    assert validate_fleet(ships_from_layout(saved[0])) == expected


def test_generation_retry_limits(monkeypatch):
    monkeypatch.setattr(bench, 'MAX_FLEET_ATTEMPTS', 0)
    with pytest.raises(ValueError, match='exhausted retries'):
        bench.draw_fleet('edge', random.Random(42))
    monkeypatch.setattr(bench, 'MAX_UNIQUE_ATTEMPTS', 2)
    fixed = bench.fixed_own_fleet()
    monkeypatch.setattr(bench, 'draw_fleet', lambda *a: fixed)
    with pytest.raises(ValueError, match='Unique layout'):
        bench.generate_layout(42, 'random', 0, {bench.canonical_fleet(fixed)})


def test_shooting_engine_matches_normal_feedback(fleet):
    practice = Game('practice', shooting_only=True)
    normal = Game('normal')
    for game in (practice, normal):
        game.join()
        game.place(0, fleet)
        game.place(1, fleet)
    # Include a miss, then sink ships, exercising target relabeling and afloat flags.
    targets = [(9, 9)] + [cell for cells in validate_fleet(fleet).values() for cell in cells]
    reserved = placement_cells(fleet[0])[0]
    harmless = iter((r, c) for r in range(10) for c in range(10) if (r,c) != reserved)
    for r, c in targets:
        for game in (practice, normal):
            game.fire(0, Coordinate(row=r, column=c))
        a, b = practice.observe(0), normal.observe(0)
        assert a.target_board == b.target_board
        assert a.remaining_enemy_ships == b.remaining_enemy_ships
        assert practice.history[-1].model_dump(exclude={'number'}) == normal.history[-1].model_dump(exclude={'number'})
        if normal.result is None:
            assert normal.turn == 1 and practice.turn == 0
            with pytest.raises(GameError, match='Wait for your turn'):
                practice.fire(1, Coordinate(row=9, column=8))
            rr, cc = next(harmless)
            normal.fire(1, Coordinate(row=rr, column=cc))
            assert practice.observe(0).legal_shot_mask == normal.observe(0).legal_shot_mask
    assert practice.result == normal.result
    assert practice.observe(0).incoming_shots == []
    assert len(practice.history) == len(targets)
    with pytest.raises(GameError, match='finished'):
        practice.fire(0, Coordinate(row=9, column=8))


def test_perfect_and_last_cell_finish(fleet):
    target_cells = [cell for cells in validate_fleet(fleet).values() for cell in cells]
    perfect = bench.run_trial(SequencePolicy(target_cells), fleet)
    assert perfect['shots_to_clear'] == 17 and perfect['misses'] == 0
    assert perfect['first_hit'] == 1 and perfect['first_sink'] == 5
    assert perfect['hits_at'] == {'10': 10, '25': 17, '50': 17}
    reserve = target_cells[-1]
    water = [(r,c) for r in range(10) for c in range(10) if (r,c) not in target_cells]
    last = bench.run_trial(SequencePolicy(water + target_cells[:-1] + [reserve]), fleet)
    assert last['shots_to_clear'] == 100 and last['misses'] == 83
    assert last['first_hit'] == 84 and last['first_sink'] == 88
    assert last['hits_at'] == {'10': 0, '25': 0, '50': 0}
    assert last['history'][-1]['outcome'] == 'sunk'


def test_own_fleet_cannot_affect_model_and_instances_are_fresh(weights):
    layout = bench.generate_layout(42, 'touching', 0, set())
    ships = ships_from_layout(layout)
    seed = bench.seed_for(42, 'touching', 0, 'policy', 'model')
    first = bench.fresh_policy('model', weights, seed)
    second = bench.fresh_policy('model', weights, seed)
    assert first is not second and first.weights is second.weights is weights
    own_other = [ShipPlacement(name=ship.name, start=Coordinate(row=ship.start.row, column=5),
                               orientation='horizontal') for ship in bench.fixed_own_fleet()]
    assert bench.run_trial(first, ships) == bench.run_trial(second, ships, own_other)
    assert isinstance(bench.fresh_policy('random', weights, seed), RandomBot)


def test_invalid_policy_and_no_post_terminal_call(fleet):
    class Repeat:
        def choose_shot(self, observation):
            return Coordinate(row=9, column=9)
    with pytest.raises(ValueError, match='illegal shot'):
        bench.run_trial(Repeat(), fleet)
    class Invalid:
        def choose_shot(self, observation):
            return (0,0)
    with pytest.raises(ValueError, match='illegal shot'):
        bench.run_trial(Invalid(), fleet)
    class Broken:
        def choose_shot(self, observation):
            raise RuntimeError('inference failed')
    with pytest.raises(RuntimeError, match='inference failed'):
        bench.run_trial(Broken(), fleet)


def result(policy, family, index, shots):
    return {'policy': policy, 'family': family, 'board_id': f'{family}:{index}',
            'shots_to_clear': shots, 'misses': shots-17, 'first_hit': 1, 'first_sink': 5,
            'hits_at': {'10': 5, '25': 10, '50': 17}}


def test_metrics_stratification_and_paired_differences():
    records = []
    for family, index, shots in [('edge', 0, 20), ('edge', 1, 20), ('center', 0, 80)]:
        records += [result('model', family, index, shots),
                    result('hunt', family, index, shots+10),
                    result('random', family, index, shots+15)]
    summary = bench.summarize(records, ('edge','center'), 42, True)
    model = summary['policies']['model']['overall']
    assert model['mean_shots'] == 40
    assert model['median_shots'] == 20 and model['p90_shots'] == 68
    assert model['mean_shots_ci95'] == [40,40]  # Family counts stay 2:1 in every resample.
    assert model['aggregate_hit_rate'] == 51/120
    assert model['mean_misses'] == 23
    assert model['mean_hits_at'] == {'10': 5, '25': 10, '50': 17}
    assert summary['comparisons']['model_minus_hunt']['overall'] == {
        'matched_layouts': 3, 'mean_shot_difference': -10.0, 'ci95': [-10,-10]}
    assert summary == bench.summarize(records, ('edge','center'), 42, True)
    # An unmatched baseline result must not enter the comparison.
    records.append(result('hunt','edge',99,100))
    comparison = bench.comparison_metrics(records, ('edge','center'), 'hunt', 42)
    assert comparison['matched_layouts'] == 3 and comparison['mean_shot_difference'] == -10
    empty = bench.summarize([], ('edge',), 42, False)
    assert empty['policies']['model']['overall']['mean_shots'] is None
    assert empty['comparisons']['model_minus_hunt']['overall']['ci95'] is None
    assert 'win_rate' not in json.dumps(summary) and 'strength' not in summary


@pytest.mark.parametrize('error', [RuntimeError('broken'), KeyboardInterrupt()])
def test_partial_reports_preserve_complete_trials(tmp_path, monkeypatch, weights, error):
    original = bench.run_trial
    calls = 0
    def fail_on_baseline(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        return original(*args, **kwargs)
    monkeypatch.setattr(bench, 'run_trial', fail_on_baseline)
    args = SimpleNamespace(output=tmp_path/'run', weights=ROOT/'weights/shooting_attempt_v1_q2.dat',
                           boards=1, placements=('random',), seed=42)
    assert bench.benchmark(args, weights) == 1
    summary = json.loads((args.output/'summary.json').read_text())
    assert summary['status'] == 'incomplete' and summary['completed_trials'] == 1
    assert summary['failure']['type'] == type(error).__name__
    assert summary['failure']['context']['policy'] == 'random'
    assert summary['comparisons']['model_minus_random']['overall']['matched_layouts'] == 0
    assert len((args.output/'trials.jsonl').read_text().splitlines()) == 1
    assert json.loads((args.output/'manifest.json').read_text())['status'] == 'incomplete'


def test_generation_failure_is_reported(tmp_path, monkeypatch, weights):
    monkeypatch.setattr(bench, 'MAX_UNIQUE_ATTEMPTS', 0)
    args = SimpleNamespace(output=tmp_path/'run', weights=ROOT/'weights/shooting_attempt_v1_q2.dat',
                           boards=1, placements=('center',), seed=42)
    assert bench.benchmark(args, weights) == 1
    summary = json.loads((args.output/'summary.json').read_text())
    assert summary['completed_trials'] == summary['generated_layouts'] == 0
    assert summary['failure']['context'] == {'stage':'generation', 'family':'center', 'index':0}


def test_preflight_and_make_quotes(tmp_path):
    import subprocess
    bad = tmp_path/'bad weights.dat'
    bad.write_text('invalid')
    output = tmp_path/'output'
    assert bench.main(['--weights',str(bad),'--output',str(output)]) == 1
    assert not output.exists()
    output.mkdir()
    with pytest.raises(SystemExit):
        bench.main(['--weights',str(bad),'--output',str(output)])
    with pytest.raises(SystemExit):
        bench.main(['--weights',str(bad),'--boards','0'])
    make = subprocess.run(['make','-n','shooting-bench','WEIGHTS=weight folder/model.dat',
                           'OUTPUT=report folder','BOARDS=20','PLACEMENTS=edge center'],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    command = make.stdout.splitlines()[-1]
    assert '--weights "weight folder/model.dat"' in command
    assert '--output "report folder"' in command
    assert '--boards "20"' in command and '--placements "edge" "center"' in command


def test_small_runs_model_switch_auto_outputs_and_no_sockets(tmp_path, monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('Shooting benchmark must not use sockets')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(bench, 'ROOT', tmp_path)
    monkeypatch.setattr(bench, 'source_fingerprints', lambda: {})
    outputs = []
    for model in ('q1', 'q2'):
        assert bench.main(['--weights',str(ROOT/f'weights/shooting_attempt_v1_{model}.dat'),
                           '--boards','4']) == 0
        paths = list((tmp_path/'.battleship/shooting').iterdir())
        output = next(path for path in paths if path not in outputs)
        outputs.append(output)
        summary = json.loads((output/'summary.json').read_text())
        assert summary['status'] == 'complete' and summary['completed_trials'] == 12
        assert summary['expected_trials'] == 12 and summary['generated_layouts'] == 4
        for row in summary['policies'].values():
            assert row['overall']['completed_trials'] == 4
            assert all(r['completed_trials'] == 1 for r in row['by_family'].values())
        records = [json.loads(line) for line in (output/'trials.jsonl').read_text().splitlines()]
        assert all(len(r['history']) == r['shots_to_clear'] for r in records)
        for board_id in {r['board_id'] for r in records}:
            assert {r['policy'] for r in records if r['board_id'] == board_id} == set(bench.POLICIES)
    assert outputs[0] != outputs[1]
    assert (outputs[0]/'layouts.jsonl').read_bytes() == (outputs[1]/'layouts.jsonl').read_bytes()
    manifests = [json.loads((p/'manifest.json').read_text()) for p in outputs]
    assert manifests[0]['model_content_hash'] != manifests[1]['model_content_hash']


@pytest.mark.parametrize('family,expected', [
    ('random', 'c327da80f227333478ebec1fd2384674a4788970d50298482170884cdc549b7c'),
    ('edge', '87f66812f88ff3c8bd6f54c85f6eec20d7ec9579882d00a3637ff26d261b25b2'),
    ('center', 'cc21221673e395b8acca22d4e38a2823323ac94a5c0e4a53dd39ba65ab3ff6d9'),
    ('touching', '0a297d434285813d85fa4f8a910354a1dc0b4c533b48ef47bb50664dec8265ab'),
])
def test_shared_generator_preserves_original_samples(family, expected):
    import hashlib
    payload = [[s.model_dump(mode='json') for s in bench.draw_fleet(family, random.Random(seed))]
               for seed in (0, 7, 42, 1234)]
    assert hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == expected
