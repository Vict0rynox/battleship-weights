"""Research UI: provenance, adapters, control lifecycle and observation boundaries."""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from battleship_client.weights import NetworkWeights, load_weight_source, parse_edits, save_weights
from battleship_server.ui import create_ui_app
from battleship_server.ui_common import publish, wire
from battleship_server.ui_experiments import Experiments, export_edits
from battleship_server.ui_games import ActiveConflict, Games
from battleship_server.ui_reports import Reports

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / 'weights').mkdir()
    (tmp_path / 'docs').mkdir()
    for name in ('weight-generation-prompt.md', 'placement-weight-generation-prompt.md'):
        shutil.copyfile(ROOT / 'docs' / name, tmp_path / 'docs' / name)
    (tmp_path / 'weights' / 'shooting_q1.dat').write_text('b3[0] = 2')
    (tmp_path / 'weights' / 'placement_q1.dat').write_text('b3[0] = 3')
    return tmp_path


def services(root):
    reports = Reports(root)
    experiments = Experiments(root, reports)
    return reports, experiments, Games(root, experiments)


def add_report(root, label='match', kind='match', *, incomplete=False):
    path = root / '.battleship' / ('shooting' if kind == 'shooting' else 'evaluations') / label
    path.mkdir(parents=True)
    digest = load_weight_source(root / 'weights' / 'shooting_q1.dat').content_hash
    placement = load_weight_source(root / 'weights' / 'placement_q1.dat', role='placement').content_hash
    manifest = {'report_version': 'placement-evaluation-1.1' if kind == 'placement' else '1.1',
                'schedule_version': 'battleship-shooting-benchmark-1.0' if kind == 'shooting' else 'battleship-evaluation-1.0',
                'model_content_hash': digest, 'content_hashes': {'model': placement},
                'weights': '/old/renamed/file.dat', 'created_at': '2026-11-03T00:00:00Z',
                'configuration': {'seed': 42, 'pairs': 1}, 'status': 'incomplete' if incomplete else 'complete'}
    publish(path / 'manifest.json', manifest)
    publish(path / 'summary.json', {'status': manifest['status'], 'matchups': {}, 'failure': {'type': 'PermissionError'} if incomplete else None})
    history = [{'number': 1, 'attacker': 0, 'coordinate': {'row': 0, 'column': 0}, 'outcome': 'hit', 'ship': None, 'sunk_cells': []},
               {'number': 2, 'attacker': 1, 'coordinate': {'row': 1, 'column': 0}, 'outcome': 'miss', 'ship': None, 'sunk_cells': []},
               {'number': 3, 'attacker': 0, 'coordinate': {'row': 0, 'column': 1}, 'outcome': 'sunk', 'ship': 'destroyer', 'sunk_cells': [{'row': 0, 'column': 0}, {'row': 0, 'column': 1}]}]
    record = {'phase': 'benchmark', 'candidate': 'model', 'opponent': 'hunt', 'candidate_seat': 1,
              'candidate_won': True, 'winner': 1, 'candidate_shots': 23, 'attacker_shots': 44,
              'seeds': {'placement': 18446744073709551615}, 'history': history, 'policy': 'model',
              'family': 'edge', 'board_id': 'edge:0', 'shots_to_clear': 50, 'attacker': 'hunt'}
    filename = 'trials.jsonl' if kind == 'shooting' else 'games.jsonl'
    (path / filename).write_text(json.dumps(record) + '\n')
    if kind == 'shooting':
        (path / 'layouts.jsonl').write_text(json.dumps({'board_id': 'edge:0', 'ships': [{'name': 'destroyer', 'start': {'row': 0, 'column': 0}, 'orientation': 'horizontal'}]})+'\n')
    return path


def config(experiments, *, seed='42', model=False, placement=False, starter=0):
    models = experiments.models()
    shooting = next(m['id'] for m in models if m['role'] == 'shooting')
    fleet = next(m['id'] for m in models if m['role'] == 'placement')
    return {'request_id': str(time.monotonic_ns()), 'seed': seed, 'speed': 4, 'starting_player': starter,
            'players': [{'kind': 'weights' if model else 'hunt', 'model_id': shooting if model else None,
                         'placement_model_id': fleet if placement else None}, {'kind': 'random', 'model_id': None, 'placement_model_id': None}]}


@pytest.mark.parametrize('role', ['shooting', 'placement'])
def test_binary_edits_roundtrip(role, tmp_path):
    source = parse_edits('W1[0,0] = .000000000000000012345678901234567\nb3[0] = -0\nb3[1] = -999999.9999999999', role=role)
    tensors = {k: v.copy() for k,v in source.tensors.items()}
    tensors['W2'][2,3] = np.nextafter(0., 1.)
    weights = NetworkWeights(tensors, role)
    path = tmp_path / 'binary.npz'
    save_weights(path, weights)
    imported = load_weight_source(path, role=role)
    text = export_edits(imported)
    assert parse_edits(text, role=role).content_hash == imported.content_hash
    assert 'e-' not in text


def test_historical_metadata_and_hash_linking(workspace):
    add_report(workspace)
    _, experiments, _ = services(workspace)
    attempt = next(a for a in experiments.attempts() if a['role'] == 'shooting')
    assert attempt['type'] == 'unknown'
    assert attempt['parent'] is None and attempt['generation_model'] is None
    assert attempt['state'] == 'evaluated' and len(attempt['evaluations']) == 1
    assert 'shooting_q1.dat' in attempt['commands'][0]


@pytest.mark.parametrize('kind', ['match', 'placement', 'shooting'])
def test_adapters_and_replay(workspace, kind):
    add_report(workspace, kind=kind)
    reports, _, _ = services(workspace)
    run = reports.list()[0]
    assert run['kind'] == kind
    data = reports.trials(run['id'])
    assert data['total'] == 1
    assert reports.trials(run['id'], query='does-not-exist')['total'] == 0
    replay = reports.replay(run['id'], '1')
    if kind == 'match':
        assert replay['players'] == ['hunt', 'file.dat']  # swapped candidate seat
        assert replay['fleets'] == [None, None]
    if kind == 'placement':
        assert all(e['attacker'] == 0 for e in replay['history'])
        assert replay['fleets'][1][0]['name'] == 'destroyer'
    if kind == 'shooting':
        assert replay['fleets'][1][0]['start'] == {'row': 0, 'column': 0}
    assert reports.distributions(run['id'])


def test_live_report_refresh_and_malformed_tail(workspace):
    path = add_report(workspace, incomplete=True)
    reports = Reports(workspace)
    run = reports.list()[0]
    assert run['summary']['failure']['type'] == 'PermissionError'
    assert reports.trials(run['id'])['total'] == 1
    with (path / 'games.jsonl').open('a') as stream:
        stream.write('{"unfinished":')
    assert len(reports.trials(run['id'])['warnings']) == 1
    with (path / 'games.jsonl').open('a') as stream:
        stream.write('1}\n')
    assert reports.trials(run['id'], phase='all')['total'] == 2


def test_compare_conservative_configuration(workspace):
    add_report(workspace, 'a')
    path = add_report(workspace, 'b')
    reports = Reports(workspace)
    ids = [r['id'] for r in reports.list()]
    assert reports.compare(ids)['compatible']
    manifest = json.loads((path/'manifest.json').read_text())
    manifest['configuration']['seed'] = 99
    publish(path/'manifest.json', manifest)
    assert not reports.compare(ids)['compatible']


def test_prompt_snapshots_and_response_validation(workspace):
    add_report(workspace)
    reports, experiments, _ = services(workspace)
    model = next(m for m in experiments.models() if m['role'] == 'shooting')
    run = reports.list()[0]
    prepared = experiments.prepare({'label': 'Improve aiming', 'role': 'shooting', 'type': 'improvement',
                                   'parent_model_id': model['id'], 'report_ids': [run['id']],
                                   'objective': 'Reduce shots against hunt', 'notes': 'Fresh chat'})
    snapshot = prepared['snapshot']
    assert '405 binary' in snapshot['prompt']
    assert 'COMPLETE replacement' in snapshot['prompt']
    assert 'Reduce shots against hunt' in snapshot['prompt']
    assert 'Previous weights' in snapshot['prompt'] and 'Evaluation feedback' in snapshot['prompt']
    assert model['hash'] == snapshot['parent_hash']
    aid = prepared['id']
    with pytest.raises(ValueError):
        experiments.record(aid, {'response': 'b3[999] = 2'})
    assert not (workspace/'weights'/'generated').exists()
    response = 'Design\n```edits\nb3[1] = 4\n```'
    imported = experiments.record(aid, {'response': response, 'generation_model': 'Example', 'notes': 'Recorded'})
    assert imported['state'] == 'imported'
    assert imported['recorded_response']['response'] == response
    assert experiments.detail(aid)['snapshot'] == snapshot
    with pytest.raises(ValueError):
        experiments.record(aid, {'response': 'b3[1]=5'})
    assert any(m['id'] == imported['model_id'] for m in experiments.models())


def test_role_and_feedback_rejection(workspace):
    add_report(workspace)
    reports, experiments, _ = services(workspace)
    models = experiments.models()
    placement = next(m for m in models if m['role'] == 'placement')
    with pytest.raises(ValueError):
        experiments.resolve(placement['id'], 'shooting')
    with pytest.raises(ValueError):
        experiments.prepare({'label':'bad','role':'shooting','type':'improvement','parent_model_id':placement['id'],'objective':'Improve','report_ids':[reports.list()[0]['id']]})
    shooting = next(m for m in models if m['role'] == 'shooting')
    with pytest.raises(ValueError):
        experiments.prepare({'label':'bad','role':'shooting','type':'improvement','parent_model_id':shooting['id'],'objective':'Improve','report_ids':[]})


@pytest.mark.parametrize('model,placement,starter', [(False,False,0),(True,False,1),(True,True,0),(False,True,1)])
def test_games_reproducible_controls_and_complete_fleets(workspace, model, placement, starter):
    _, experiments, games = services(workspace)
    settings = config(experiments, model=model, placement=placement, starter=starter)
    game = games.create(settings)
    assert game['turn'] == starter and len(game['fleets'][0]) == 5
    assert games.create(settings)['id'] == game['id']
    other = dict(settings, request_id='different')
    with pytest.raises(ActiveConflict):
        games.create(other)
    gid = game['id']
    assert games.control(gid, 'pause')['status'] == 'paused'
    assert len(games.control(gid, 'step')['history']) == 1
    games.control(gid, 'speed', 16)
    games.control(gid, 'finish')
    while games.active['status'] == 'running':
        with games.lock:
            games.advance()
    first = games.snapshot(gid)
    assert first['result']['reason'] == 'fleet_destroyed'
    assert first['history'][0]['attacker'] == starter
    second = games.create(other)
    while games.active['status'] == 'running':
        with games.lock:
            games.advance()
    assert games.snapshot(second['id'])['history'] == first['history']
    assert games.snapshot(gid)['history'] == first['history']


def test_both_models_and_independent_observations(workspace):
    _, experiments, games = services(workspace)
    settings = config(experiments, model=True, placement=True)
    settings['players'][1] = dict(settings['players'][0])
    game = games.create(settings)
    for seat, policy in enumerate(games.policies):
        original = policy.choose_shot
        def checked(observation, seat=seat, original=original):
            assert observation.player == seat
            assert len(observation.own_fleet) == 5
            assert not hasattr(observation, 'enemy_fleet')
            return original(observation)
        policy.choose_shot = checked
    for _ in range(10):
        games.control(game['id'], 'step')
    assert len(games.snapshot()['history']) == 10


def test_invalid_preflight_and_seed(workspace):
    _, experiments, games = services(workspace)
    settings = config(experiments, model=True)
    settings['players'][0]['model_id'] = 'missing'
    with pytest.raises(ValueError):
        games.create(settings)
    assert not games.folder.exists()
    settings = config(experiments, seed='18446744073709551616')
    with pytest.raises(ValueError):
        games.create(settings)
    assert wire({'configuration': {'seed': 42}, 'seeds': {'placement': 2**64-1}})['seeds']['placement'] == str(2**64-1)


def test_restart_marks_interrupted_and_worker_pacing(workspace):
    _, experiments, games = services(workspace)
    gid = games.create(config(experiments))['id']
    restarted = Games(workspace, experiments)
    assert restarted.snapshot(gid)['status'] == 'interrupted'
    # Paused worker never advances; instant finish is independent of browser connection.
    games.control(gid, 'pause')
    games.start_worker()
    time.sleep(.03)
    assert not games.snapshot(gid)['history']
    games.control(gid, 'finish')
    deadline = time.monotonic()+5
    while games.snapshot(gid)['status'] == 'running' and time.monotonic()<deadline:
        time.sleep(.02)
    assert games.snapshot(gid)['status'] == 'finished'
    games.close()


def test_recording_failure_stops_runner(workspace, monkeypatch):
    _, experiments, games = services(workspace)
    gid = games.create(config(experiments))['id']
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr('battleship_server.ui_games.publish', fail)
    with pytest.raises(OSError):
        games.control(gid, 'step')
    assert games.snapshot(gid)['status'] == 'error'
    with pytest.raises(ValueError):
        games.control(gid, 'play')


def test_api_security_lifecycle_and_sse(workspace):
    add_report(workspace)
    app = create_ui_app(workspace)
    with TestClient(app, base_url='http://127.0.0.1') as client:
        assert client.get('/').status_code == 200
        assert "script-src 'self'" in client.get('/').headers['content-security-policy']
        assert client.get('/assets/app.js').status_code == 200
        assert client.post('/ui-api/games', json={}, headers={'Origin': 'https://other.example'}).status_code == 403
        assert client.get('/ui-api/models', headers={'Host': 'evil.example'}).status_code == 400
        assert client.post('/ui-api/attempts', content='{}').status_code == 415
        assert client.get('/ui-api/games/../../README.md').status_code == 404
        settings = config(app.state.experiments)
        start = client.post('/ui-api/games', json=settings)
        assert start.status_code == 200
        gid = start.json()['id']
        duplicate = client.post('/ui-api/games', json=settings)
        assert duplicate.json()['id'] == gid
        conflict = client.post('/ui-api/games', json=dict(settings, request_id='second'))
        assert conflict.status_code == 409 and conflict.json()['game_id'] == gid
        assert client.post(f'/ui-api/games/{gid}/control', json={'action':'finish'}).status_code == 200
        deadline = time.monotonic()+5
        while app.state.games.snapshot(gid)['status'] == 'running' and time.monotonic()<deadline:
            time.sleep(.02)
        record = client.get(f'/ui-api/games/{gid}').json()
        assert record['status'] == 'finished'
        assert record['configuration']['seed'] == '42'
        events = client.get(f'/ui-api/games/{gid}/events')
        assert f"id: {record['revision']}" in events.text and 'data: ' in events.text
        assert client.get(f'/ui-api/games/{gid}/events', headers={'Last-Event-ID': str(record['revision'])}).text == ''
        assert client.get('/ui-api/games/'+'a'*32+'/events').status_code == 404


def test_symlinks_outside_roots_rejected(workspace, tmp_path):
    secret = tmp_path/'outside.dat'
    secret.write_text('b3[0]=9')
    (workspace/'weights'/'escape.dat').symlink_to(secret)
    # tmp_path equals workspace here, but secret is outside the allowed weights root.
    _, experiments, _ = services(workspace)
    invalid = next(m for m in experiments.models() if m['label'] == 'escape.dat')
    assert invalid.get('error')
