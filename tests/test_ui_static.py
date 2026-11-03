"""Public export boundaries and builds independent of local/private runtime data."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from battleship_client.weights import load_weight_source

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_ui', ROOT / 'scripts/export_ui.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.fixture
def public_workspace(tmp_path):
    root = tmp_path / 'source'
    (root / 'weights').mkdir(parents=True)
    (root / 'weights/shooting-example.dat').write_text('b3[0] = 2')
    digest = load_weight_source(root / 'weights/shooting-example.dat').content_hash
    report = root / '.battleship/evaluations/example'
    report.mkdir(parents=True)
    manifest = {'report_version': '1.1', 'schedule_version': 'battleship-evaluation-1.0',
                'created_at': '2026-11-03T00:00:00Z', 'status': 'complete',
                'model_content_hash': digest, 'weights': '/synthetic/private/SECRET_SOURCE_PATH/weights.dat',
                'runtime': {'thread_environment': {'PRIVATE': 'SECRET_ENVIRONMENT'}},
                'source_sha256': {'secret_file': 'SECRET_SOURCE'},
                'configuration': {'pairs': 50, 'seed': 42, 'opponents': ['hunt'],
                                  'weights_path': 'SECRET_CONFIG_PATH'}}
    (report / 'manifest.json').write_text(json.dumps(manifest))
    summary = {'status': 'complete', 'matchups': {'model_vs_hunt': {'win_rate': .7, 'completed_games': 100, 'ci95': [.6,.8]}},
               'failure': {'type': 'ExampleError', 'context': {'path': 'SECRET_FAILURE_CONTEXT'}}}
    (report / 'summary.json').write_text(json.dumps(summary))
    with (report / 'games.jsonl').open('w') as stream:
        for trial in range(100):
            row = {'phase': 'benchmark', 'trial': trial, 'candidate': 'model', 'opponent': 'hunt',
                   'candidate_seat': trial % 2, 'candidate_won': trial % 3 != 0,
                   'winner': trial % 2 if trial % 3 != 0 else 1-trial % 2,
                   'candidate_shots': 50, 'notes': 'SECRET_RECORD_NOTES',
                   'seeds': {'placement': 2**64-1},
                   'history': [{'number': 1, 'attacker': 0, 'coordinate': {'row': 0, 'column': 0},
                                'outcome': 'miss', 'ship': None, 'sunk_cells': []}]}
            stream.write(json.dumps(row)+'\n')
    (root / 'host.session.json').write_text('SECRET_TOKEN')
    draft = root / '.battleship/experiments/draft'
    draft.mkdir(parents=True)
    (draft/'attempt.json').write_text(json.dumps({'id':'draft','label':'Private draft', 'role':'shooting','hash':None,'prompt':'SECRET_PROMPT','notes':'SECRET_NOTES'}))
    return root


def test_snapshot_full_statistics_sampled_replays_and_no_private_data(public_workspace, tmp_path):
    snapshot = exporter.export_snapshot(public_workspace, tmp_path/'public', 2)
    catalog = json.loads((snapshot/'index.json').read_text())
    assert len(catalog['models']) == 1 and len(catalog['attempts']) == 1
    assert all(a['id'] != 'draft' for a in catalog['attempts'])
    run = catalog['runs'][0]
    assert run['summary']['matchups']['model_vs_hunt']['completed_games'] == 100
    assert run['summary']['matchups']['model_vs_hunt']['win_rate'] == .7
    assert run['replay_sample']['available'] == 8
    detail = json.loads((snapshot/'runs'/f"{run['id']}.json").read_text())
    assert len(detail['trials']) == 8
    assert detail['trials'][0]['seeds']['placement'] == str(2**64-1)
    assert sum(n for values in detail['distributions'].values() for _, n in values) == 100
    text = ''.join(path.read_text() for path in snapshot.rglob('*.json'))
    assert 'SECRET_' not in text and '/synthetic/private' not in text
    assert 'runtime' not in detail['manifest']
    assert list(snapshot.rglob('*.session.json')) == []
    # Exporting the same inputs is reproducible, with no export-time timestamp noise.
    before = {str(p.relative_to(snapshot)): p.read_bytes() for p in snapshot.rglob('*') if p.is_file()}
    exporter.export_snapshot(public_workspace, snapshot, 2)
    after = {str(p.relative_to(snapshot)): p.read_bytes() for p in snapshot.rglob('*') if p.is_file()}
    assert before == after


def test_static_build_stdlib_only_and_project_subpath(public_workspace, tmp_path):
    snapshot = exporter.export_snapshot(public_workspace, tmp_path/'public', 1)
    output = tmp_path/'github-project'
    # -S removes site packages: publishing must not import NumPy/FastAPI or local data.
    subprocess.run([sys.executable, '-S', str(ROOT/'scripts/export_ui.py'), 'build',
                    '--snapshot', str(snapshot), '--output', str(output)], check=True)
    index = (output/'index.html').read_text()
    assert 'href="./assets/style.css"' in index
    assert 'src="./assets/app.js"' in index
    assert 'name="battleship-mode" content="static"' in index
    assert (output/'.nojekyll').exists()
    assert (output/'assets/browser-game.js').exists()
    assert (output/'data/index.json').exists()
    assert not list(output.rglob('*.py'))
    assert not list(output.rglob('*.session.json'))
    assert not list(output.rglob('*.dat'))


def test_export_refuses_overwriting_source_or_unmarked_directories(public_workspace, tmp_path):
    with pytest.raises(ValueError):
        exporter.export_snapshot(public_workspace, public_workspace)
    destination = tmp_path/'important'
    destination.mkdir()
    (destination/'keep.txt').write_text('keep')
    with pytest.raises(ValueError):
        exporter.export_snapshot(public_workspace, destination)
    assert (destination/'keep.txt').read_text() == 'keep'
    snapshot = exporter.export_snapshot(public_workspace, tmp_path/'public')
    with pytest.raises(ValueError):
        exporter.build_site(snapshot, snapshot)
    (snapshot/'link').symlink_to(public_workspace/'host.session.json')
    with pytest.raises(ValueError):
        exporter.build_site(snapshot, tmp_path/'site')


def test_even_sampling_covers_outcomes_seats_and_phases():
    rows = [{'id':str(i+1), 'candidate_seat':i%2, 'candidate_won':i%3==0,
             'phase':'smoke' if i<10 else 'benchmark'} for i in range(100)]
    sample = exporter.select_trials(rows, 'match', 2)
    for key in ('candidate_seat','candidate_won','phase'):
        assert {r[key] for r in sample} == {r[key] for r in rows}
    assert sample == exporter.select_trials(rows, 'match', 2)


def test_browser_logits_agree_with_numpy(tmp_path):
    import shutil
    import numpy as np
    if not shutil.which('node'):
        pytest.skip('Node is only needed for browser inference verification')
    from battleship_client.weights import parse_edits
    for role, count in [('shooting', 405), ('placement', 505)]:
        source = 'W1[0:3,0:2] = [[.3,-.2],[1.2,.7],[-.8,.9]]\nb1[0:2] = [.1,-.1]\nW2[0:2,0:2] = [[.2,.6],[-.4,.8]]\nW3[0:2,0:3] = [[.5,-.3,.9],[.8,.4,-.7]]\nb3[0:3] = [.1,.2,-.2]'
        weights = parse_edits(source, role=role)
        x = np.random.default_rng(42).uniform(-1, 1, count)
        w = weights.tensors
        expected = np.tanh(np.tanh(x @ w['W1'] + w['b1']) @ w['W2'] + w['b2']) @ w['W3'] + w['b3']
        payload = {'model': {'role': role, 'tensors': {k:v.tolist() for k,v in w.items()}}, 'input':x.tolist()}
        script = """
import {readFileSync} from 'node:fs';
const source=readFileSync(process.argv[1],'utf8');
const runtime=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const payload=JSON.parse(readFileSync(0,'utf8'));
console.log(JSON.stringify(runtime.logits(payload.input,payload.model)));
"""
        result = subprocess.run(['node', '--input-type=module', '-e', script,
                                 str(ROOT/'packages/server/src/battleship_server/static/browser-game.js')],
                                input=json.dumps(payload), capture_output=True, text=True, check=True)
        np.testing.assert_allclose(json.loads(result.stdout), expected, rtol=1e-12, atol=1e-12)


def test_default_export_is_summary_only(public_workspace, tmp_path):
    snapshot = exporter.export_snapshot(public_workspace, tmp_path/'public')
    catalog = json.loads((snapshot/'index.json').read_text())
    assert catalog['export']['replays_per_group'] == 0
    assert not (snapshot/'replays').exists()
    run = catalog['runs'][0]
    assert run['replay_sample']['available'] == 0
    assert run['replay_sample']['total'] == 100
    detail = json.loads((snapshot/'runs'/f"{run['id']}.json").read_text())
    assert detail['trials'] == []
    assert detail['summary']['matchups']['model_vs_hunt']['completed_games'] == 100
    assert sum(n for values in detail['distributions'].values() for _, n in values) == 100
