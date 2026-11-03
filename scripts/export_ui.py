"""Export an intentional public snapshot, or build a dependency-free Pages site.

Snapshot export needs the project's Python dependencies; building an existing
snapshot uses only the standard library and never opens .battleship or weights.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'packages/server/src/battleship_server/static'
CONFIG_KEYS = {
    'seed', 'pairs', 'boards', 'trials', 'placements', 'allocation', 'policies',
    'opponents', 'attackers', 'candidates', 'smoke_pairs', 'smoke_trials',
    'bootstrap_samples', 'placement_distribution', 'sampling_temperature',
    'temperature', 'turn_timeout', 'execution', 'http_smoke', 'fixture', 'defender',
    'sampling_with_replacement', 'max_fleet_attempts', 'max_unique_attempts',
}
RECORD_KEYS = {
    'id', 'report_version', 'phase', 'trial', 'candidate', 'opponent', 'candidate_seat',
    'seeds', 'winner', 'candidate_won', 'finish_reason', 'candidate_shots',
    'opponent_shots', 'candidate_hits', 'opponent_hits', 'attacker', 'fleet_hash',
    'attacker_shots', 'stage', 'board_id', 'family', 'index', 'policy', 'placement_seed',
    'policy_seed', 'shots_to_clear', 'misses', 'first_hit', 'first_sink', 'hits_at',
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')


def public_configuration(config):
    return {k: v for k, v in config.items() if k in CONFIG_KEYS}


def public_run(run):
    summary = dict(run['summary'])
    if summary.get('failure'):
        summary['failure'] = {'type': summary['failure'].get('type', 'Unknown')}
    return {**{k: run[k] for k in ('id', 'label', 'kind', 'created_at', 'status', 'hashes')},
            'configuration': public_configuration(run['configuration']), 'summary': summary}


def select_trials(rows, kind, per_group):
    """Evenly spaced samples per phase/matchup/outcome/seat or policy/layout group."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in ('phase', 'candidate', 'opponent', 'attacker',
                                          'policy', 'family', 'candidate_seat', 'candidate_won'))
        groups[key].append(row)
    selected = []
    for group in groups.values():
        count = min(per_group, len(group))
        indexes = {round(i * (len(group)-1) / max(1, count-1)) for i in range(count)}
        selected.extend(group[i] for i in sorted(indexes))
    return sorted(selected, key=lambda row: int(row['id']))


def export_snapshot(root, destination, per_group=0):
    for package in ('protocol', 'client', 'server'):
        sys.path.insert(0, str(ROOT / 'packages' / package / 'src'))
    from battleship_client.weights import load_weight_source
    from battleship_server.ui_common import wire
    from battleship_server.ui_experiments import Experiments
    from battleship_server.ui_reports import Reports

    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root.is_relative_to(destination) or destination.is_relative_to(root / '.battleship') or destination.is_relative_to(root / 'weights'):
        raise ValueError('Choose a separate public snapshot directory, such as site-data')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.snapshot-', dir=destination.parent))
    try:
        reports = Reports(root)
        experiments = Experiments(root, reports)
        models = [m for m in experiments.models() if not m.get('error')]
        for model in models:
            file, _ = experiments.resolve(model['id'], model['role'])
            weights = load_weight_source(file, role=model['role'])
            write(stage / 'models' / f"{model['id']}.json", {
                **model, 'tensors': {name: array.tolist() for name, array in weights.tensors.items()}})
        runs = []
        for source in reports.list():
            run, _, rows, warnings = reports.index(source['id'])
            public = public_run(run)
            selected = select_trials(rows, run['kind'], per_group) if per_group else []
            public['replay_sample'] = {'available': len(selected), 'total': len(rows),
                                       'per_group': per_group,
                                       'method': 'summary only' if not per_group else 'evenly spaced within phase, matchup, outcome, seat, policy and layout groups'}
            manifest = {k: run['manifest'][k] for k in ('report_version', 'schedule_version', 'created_at', 'status', 'versions') if k in run['manifest']}
            manifest['configuration'] = public['configuration']
            detail = {**public, 'manifest': manifest,
                      'distributions': reports.distributions(source['id']),
                      'trials': [{k: v for k, v in row.items() if k in RECORD_KEYS} for row in selected],
                      'warnings': warnings}
            write(stage / 'runs' / f"{source['id']}.json", wire(detail))
            for row in selected:
                replay = reports.replay(source['id'], row['id'])
                replay['configuration'] = public['configuration']
                replay['record'] = {k: v for k, v in replay['record'].items() if k in RECORD_KEYS}
                write(stage / 'replays' / source['id'] / f"{row['id']}.json", wire(replay))
            runs.append(public)
        attempts = []
        for source in experiments.attempts():
            if not source.get('hash'):
                continue  # Unsubmitted/private drafts are never published.
            attempt = {k: source.get(k) for k in ('id', 'label', 'role', 'model_id', 'hash', 'type',
                                                  'parent', 'parent_hash', 'generation_model', 'created_at', 'state', 'commands')}
            attempt.update(notes='', evaluations=[r for r in runs if r['hashes'].get(attempt['role']) == attempt['hash']])
            attempts.append(attempt)
        write(stage / 'index.json', wire({'format': 'battleship-public-1.0', 'models': models,
              'runs': runs, 'attempts': attempts, 'games': [], 'export': {'replays_per_group': per_group,
              'note': 'Full-run summary statistics and distributions; no raw trial histories.' if not per_group else 'Statistics cover full runs. Replay lists are representative samples, not the full dataset.'}}))
        publish_directory(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination


def publish_directory(stage, destination):
    """Only replace directories explicitly marked as our generated output."""
    marker = '.battleship-generated'
    (stage / marker).write_text('Generated by scripts/export_ui.py\n')
    if destination.exists():
        if destination.is_symlink() or not (destination / marker).is_file():
            raise ValueError('Refusing to replace an unmarked directory')
        backup = destination.with_name(destination.name + '.previous')
        if backup.exists():
            raise ValueError(f'Remove the previous export backup first: {backup}')
        destination.rename(backup)
        try:
            stage.rename(destination)
        except Exception:
            backup.rename(destination)
            raise
        shutil.rmtree(backup)
    else:
        stage.rename(destination)


def build_site(snapshot, output):
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if (snapshot == output or snapshot.is_relative_to(output) or output.is_relative_to(snapshot)
            or output == ROOT or ROOT.is_relative_to(output)):
        raise ValueError('Build output must be separate from source and snapshot')
    catalog = json.loads((snapshot / 'index.json').read_text())
    if catalog.get('format') != 'battleship-public-1.0':
        raise ValueError('Unsupported public snapshot version')
    # Refuse links even in the versioned snapshot; Pages artifacts must contain only files.
    if any(p.is_symlink() for p in snapshot.rglob('*')):
        raise ValueError('Snapshot must not contain symbolic links')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.pages-', dir=output.parent))
    try:
        shutil.copytree(STATIC, stage / 'assets')
        index = (STATIC / 'index.html').read_text().replace('href="/assets/', 'href="./assets/').replace('src="/assets/', 'src="./assets/')
        index = index.replace('<title>', '<meta name="battleship-mode" content="static" />\n    <title>')
        index = index.replace('Local research workspace', 'Public experiment archive').replace('LOCAL WORKSPACE', 'PUBLIC ARCHIVE')
        (stage / 'index.html').write_text(index)
        shutil.copytree(snapshot, stage / 'data', ignore=shutil.ignore_patterns('.battleship-generated'))
        (stage / '.nojekyll').touch()
        publish_directory(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    export = subs.add_parser('snapshot', help='Explicitly export local public results and weights')
    export.add_argument('--root', type=Path, default=ROOT)
    export.add_argument('--output', type=Path, default=ROOT / 'site-data')
    export.add_argument('--replays-per-group', type=int, default=0, help='0 exports summaries only (default); positive values explicitly opt into replay samples')
    build = subs.add_parser('build', help='Build Pages files from a committed public snapshot')
    build.add_argument('--snapshot', type=Path, default=ROOT / 'site-data')
    build.add_argument('--output', type=Path, default=ROOT / 'dist')
    args = parser.parse_args(argv)
    try:
        if args.action == 'snapshot':
            if not 0 <= args.replays_per_group <= 100:
                parser.error('--replays-per-group must be between 0 and 100')
            destination = export_snapshot(args.root, args.output, args.replays_per_group)
        else:
            destination = build_site(args.snapshot, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Export failed: {error}\n')
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
