"""Read-only adapters and byte-offset indexes for existing benchmark reports."""
import json
import threading
from pathlib import Path
from collections import Counter, defaultdict

from .ui_common import confined, identity, read_json


class Reports:
    def __init__(self, root):
        self.root = root
        self.paths = {}
        self.indexes = {}
        self.lock = threading.RLock()

    def list(self):
        runs = []
        with self.lock:
            self.paths = {}
            for folder in ("evaluations", "shooting"):
                base = self.root / '.battleship' / folder
                for manifest in sorted(base.glob('*/manifest.json'), reverse=True):
                    try:
                        path = confined(manifest.parent, base)
                        m = read_json(path / 'manifest.json')
                        version = str(m.get('report_version', ''))
                        schedule = str(m.get('schedule_version', ''))
                        kind = ('placement' if version.startswith('placement-evaluation-') else
                                'shooting' if 'shooting' in schedule else 'match')
                        if kind == 'match' and not schedule.startswith('battleship-evaluation'):
                            continue
                        rid = identity(path.relative_to(self.root))
                        self.paths[rid] = path
                        summary = read_json(path / 'summary.json') if (path / 'summary.json').exists() else {}
                        hashes = ({'placement': m.get('content_hashes', {}).get('model'),
                                   'shooting': m.get('attacker_content_hash')} if kind == 'placement' else
                                  {'shooting': m.get('model_content_hash'), 'placement': m.get('placement_content_hash')})
                        runs.append({'id': rid, 'label': path.name, 'kind': kind,
                                     'created_at': m.get('created_at'), 'status': m.get('status', 'incomplete'),
                                     'hashes': hashes, 'configuration': m.get('configuration', {}),
                                     'summary': summary})
                    except (ValueError, OSError, TypeError):
                        # One broken report must not make the remaining library unusable.
                        continue
        return sorted(runs, key=lambda r: r['created_at'] or '', reverse=True)

    def detail(self, rid):
        runs = self.list()
        run = next((r for r in runs if r['id'] == rid), None)
        if run is None:
            raise FileNotFoundError('Report not found')
        path = self.paths[rid]
        return {**run, 'manifest': read_json(path / 'manifest.json')}

    def index(self, rid):
        with self.lock:
            run = self.detail(rid)
            path = self.paths[rid] / ('trials.jsonl' if run['kind'] == 'shooting' else 'games.jsonl')
            if not path.exists():
                return run, path, [], ['Records are not available yet.']
            confined(path, self.paths[rid])
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            cached = self.indexes.get(rid)
            if cached and cached[0] == stamp:
                return run, path, cached[1], cached[2]
            rows, warnings = [], []
            with path.open('rb') as stream:
                line = 0
                while True:
                    offset = stream.tell()
                    raw = stream.readline()
                    if not raw:
                        break
                    line += 1
                    try:
                        value = json.loads(raw)
                        metadata = {k: v for k, v in value.items() if k != 'history'}
                        metadata.update(id=str(line), offset=offset)
                        rows.append(metadata)
                    except (ValueError, TypeError, AttributeError):
                        warnings.append(f'Record {line} is incomplete or malformed and was skipped.')
            self.indexes[rid] = (stamp, rows, warnings)
            return run, path, rows, warnings

    def trials(self, rid, *, offset=0, limit=30, phase='benchmark', query='', candidate='', opponent='', family='', outcome=''):
        run, path, rows, warnings = self.index(rid)
        def matches(row):
            if phase != 'all' and run['kind'] != 'shooting' and row.get('phase') != phase:
                return False
            if candidate and row.get('candidate', row.get('policy')) != candidate:
                return False
            if opponent and row.get('opponent', row.get('attacker')) != opponent:
                return False
            if family and row.get('family') != family:
                return False
            if outcome and row.get('candidate_won') != (outcome == 'win'):
                return False
            return not query or query.lower() in json.dumps(row).lower()
        selected = [r for r in rows if matches(r)]
        facets = {key: sorted({str(r.get(key, r.get(alias, ''))) for r in rows} - {''})
                  for key, alias in [('candidate', 'policy'), ('opponent', 'attacker'), ('family', 'family')]}
        return {'total': len(selected), 'items': [{k: v for k, v in r.items() if k != 'offset'}
                                                for r in selected[offset:offset + limit]],
                'facets': facets, 'warnings': warnings}

    def distributions(self, rid):
        run, _, rows, _ = self.index(rid)
        groups = defaultdict(Counter)
        for row in rows:
            if run['kind'] != 'shooting' and row.get('phase') != 'benchmark':
                continue
            if run['kind'] == 'shooting':
                group, value = f"{row.get('policy')} · {row.get('family')}", row.get('shots_to_clear')
            elif run['kind'] == 'placement':
                group, value = f"{row.get('candidate')} vs {row.get('attacker')}", row.get('attacker_shots')
            else:
                group, value = f"{row.get('candidate')} vs {row.get('opponent')}", row.get('candidate_shots')
            if value is not None:
                groups[group][value] += 1
        return {key: sorted(value.items()) for key, value in groups.items()}

    def replay(self, rid, tid):
        run, path, rows, _ = self.index(rid)
        row = next((r for r in rows if r['id'] == tid), None)
        if row is None:
            raise FileNotFoundError('Trial not found')
        with path.open('rb') as stream:
            stream.seek(row['offset'])
            record = json.loads(stream.readline())
        history = record.get('history', [])
        fleets = [None, None]
        if run['kind'] == 'shooting':
            layouts = path.with_name('layouts.jsonl')
            if layouts.exists():
                confined(layouts, path.parent)
                with layouts.open() as stream:
                    for raw in stream:
                        layout = json.loads(raw)
                        if layout.get('board_id') == record.get('board_id'):
                            fleets[1] = layout.get('ships')
                            break
        elif run['kind'] == 'placement':
            # Completed attacks reveal the whole target fleet through sink events.
            fleets[1] = [{'name': e['ship'], 'cells': e['sunk_cells']} for e in history
                         if e['attacker'] == 0 and e['outcome'] == 'sunk'] or None
            history = [e for e in history if e['attacker'] == 0]
        model_label = Path(run['manifest'].get('weights', 'model')).name
        if run['kind'] == 'match':
            names = ['', '']
            seat = record['candidate_seat']
            names[seat] = model_label if record['candidate'] == 'model' else record['candidate']
            names[1-seat] = record['opponent']
            result = {'winner': record['winner'], 'reason': record.get('finish_reason')}
        else:
            shooter = record.get('policy', record.get('attacker', 'attacker'))
            names = [model_label if shooter == 'model' else shooter,
                     record.get('candidate', record.get('family', 'target'))]
            result = {'winner': 0, 'reason': 'fleet_destroyed'}
        return {'id': f'{rid}/{tid}', 'label': f"{run['label']} · trial {tid}", 'kind': run['kind'],
                'players': names, 'history': history, 'fleets': fleets, 'result': result,
                'status': 'finished', 'record': record, 'configuration': run['configuration']}

    def compare(self, ids):
        runs = [self.detail(rid) for rid in ids]
        if len(runs) < 2:
            raise ValueError('Select at least two reports')
        def signature(run):
            m = run['manifest']
            # Conservative: all benchmark configuration must agree, including sample size.
            return {'kind': run['kind'], 'configuration': run['configuration'],
                    'versions': m.get('versions'), 'schedule': m.get('schedule_version'),
                    'placement': run['hashes'].get('placement') if run['kind'] != 'placement' else None,
                    'attacker': run['hashes'].get('shooting') if run['kind'] == 'placement' else None}
        compatible = all(signature(r) == signature(runs[0]) for r in runs[1:])
        return {'compatible': compatible,
                'message': ('Configurations match; metrics can be compared descriptively. No cross-run significance is inferred.'
                            if compatible else 'Incompatible configurations: compare settings below; no improvement score is calculated.'),
                'runs': runs}
