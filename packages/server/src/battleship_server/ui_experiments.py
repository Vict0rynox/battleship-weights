"""Model catalog and immutable, user-recorded generation provenance."""
import json
import shlex
import threading
import uuid
from decimal import Decimal

import numpy as np

from battleship_client.weights import load_weight_source, parse_edits, save_weights
from .ui_common import confined, identity, now, publish, read_json


def export_edits(weights):
    """Export decimal literals, preserving binary64 values including signed zero."""
    def number(value):
        return format(Decimal(repr(float(value))), 'f')
    lines = []
    for name, array in weights.tensors.items():
        if array.ndim == 1:
            lines.append(f"{name}[:] = [" + ','.join(number(v) for v in array) + ']')
        else:
            for index, row in enumerate(array):
                if any(row != 0) or any(np.signbit(row)):
                    lines.append(f"{name}[{index},:] = [" + ','.join(number(v) for v in row) + ']')
    text = '```edits\n' + '\n'.join(lines) + '\n```'
    if parse_edits(text, role=weights.role).content_hash != weights.content_hash:
        raise ValueError('Artifact cannot be represented losslessly in the edits contract')
    return text


class Experiments:
    def __init__(self, root, reports):
        self.root, self.reports = root, reports
        self.folder = root / '.battleship' / 'experiments'
        self.models_by_id = {}
        self.cache = {}
        self.lock = threading.RLock()

    def models(self):
        items, models = [], {}
        base = self.root / 'weights'
        for file in sorted(base.rglob('*')):
            if not file.is_file() or file.suffix.lower() not in {'.dat', '.edits', '.npz'}:
                continue
            try:
                file = confined(file, base)
                stat = file.stat()
                key = (str(file), stat.st_mtime_ns, stat.st_size)
                cached = self.cache.get(key)
                if cached is None:
                    # Text sources carry no role tag. The established filenames declare
                    # their role; custom ambiguous text defaults to shooting.
                    roles = ('placement', 'shooting') if 'placement' in file.name else ('shooting', 'placement')
                    errors = []
                    for role in roles:
                        try:
                            weights = load_weight_source(file, role=role)
                            cached = (role, weights.content_hash)
                            break
                        except (ValueError, OSError) as error:
                            errors.append(str(error))
                    if cached is None:
                        raise ValueError('; '.join(errors))
                    self.cache[key] = cached
                role, digest = cached
                mid = identity(file.relative_to(self.root))
                models[mid] = file
                items.append({'id': mid, 'label': str(file.relative_to(base)), 'role': role, 'hash': digest})
            except (ValueError, OSError) as error:
                items.append({'id': identity(str(file)), 'label': file.name, 'error': str(error)})
        self.models_by_id = models
        return items

    def resolve(self, mid, role=None):
        models = self.models()
        item = next((m for m in models if m['id'] == mid and not m.get('error')), None)
        if item is None or (role and item['role'] != role):
            raise ValueError('Select a valid model of the required role')
        return self.models_by_id[mid], item

    def attempts(self):
        runs = self.reports.list()
        items = []
        for model in self.models():
            if model.get('error') or model['label'].startswith('generated/'):
                continue
            items.append({'id': model['id'], 'label': model['label'], 'role': model['role'],
                          'model_id': model['id'], 'hash': model['hash'], 'type': 'unknown',
                          'parent': None, 'generation_model': None, 'notes': '', 'created_at': None})
        for path in sorted(self.folder.glob('*/attempt.json')):
            try:
                confined(path, self.folder)
                item = read_json(path)
                response = path.with_name('response.json')
                if response.exists():
                    confined(response, self.folder)
                    item.update({k: v for k, v in read_json(response).items() if k != 'response'})
                items.append(item)
            except (ValueError, OSError):
                continue
        for item in items:
            item['evaluations'] = [r for r in runs if r['hashes'].get(item['role']) == item.get('hash') and item.get('hash')]
            item['state'] = 'evaluated' if item['evaluations'] else 'imported' if item.get('hash') else 'prepared'
            if item.get('model_id'):
                path = self.models_by_id.get(item['model_id'])
                if path:
                    relative = shlex.quote(str(path.relative_to(self.root)))
                    item['commands'] = ([f'make shooting-bench WEIGHTS={relative}', f'make evaluate-hunt WEIGHTS={relative}']
                                        if item['role'] == 'shooting' else [f'make evaluate-placement PLACEMENT_WEIGHTS={relative}'])
        return items

    def detail(self, aid):
        item = next((a for a in self.attempts() if a['id'] == aid), None)
        if item is None:
            raise FileNotFoundError('Attempt not found')
        path = self.folder / aid
        for filename, field in [('attempt.json', 'snapshot'), ('response.json', 'recorded_response')]:
            if (path / filename).exists():
                confined(path / filename, self.folder)
                item[field] = read_json(path / filename)
        return item

    def prepare(self, body):
        role, kind = body['role'], body['type']
        if role not in ('shooting', 'placement') or kind not in ('one-shot', 'improvement'):
            raise ValueError('Choose a supported role and attempt type')
        label = body['label'].strip()
        if not label:
            raise ValueError('An attempt label is required')
        template = 'weight-generation-prompt.md' if role == 'shooting' else 'placement-weight-generation-prompt.md'
        contract = (self.root / 'docs' / template).read_text().split('---', 1)[-1].strip()
        if kind == 'improvement':
            contract = contract.replace('Choose, from scratch, the numerical weights', 'Choose the numerical weights')
        feedback, parent, parent_hash = [], None, None
        prompt = ('# Fresh-chat weight generation\n\nUse only the explicit package below. '
                  'Do not use earlier conversation, tools, code execution, training, or additional feedback rounds.\n\n' + contract)
        objective = body.get('objective', '').strip()
        if kind == 'improvement':
            path, model = self.resolve(body.get('parent_model_id'), role)
            if not body.get('report_ids') or not objective:
                raise ValueError('Improvement requires a parent, evaluation feedback, and objective')
            for rid in body['report_ids']:
                run = self.reports.detail(rid)
                if run['hashes'].get(role) != model['hash']:
                    raise ValueError('Feedback must evaluate the selected parent artifact')
                feedback.append(run)
            parent = model['id']
            parent_hash = model['hash']
            prompt += ('\n\n# Fresh-context improvement\n\nThe contract above defines the runtime. '
                       'Improve the supplied prior design using the explicit feedback below. '
                       'Return a COMPLETE replacement edits block applied from ZERO. Do not return a patch.\n\n'
                       '# Objective\n' + objective + '\n\n# Previous weights\n' +
                       export_edits(load_weight_source(path, role=role)) +
                       '\n\n# Evaluation feedback\n```json\n' + json.dumps(feedback, indent=2) + '\n```')
        elif objective:
            prompt += '\n\n# Objective\n' + objective
        prompt += ('\n\nReturn a brief design explanation and one complete edits block. '
                   'All omitted parameters start at zero. This is a single response, without iteration.\n')
        aid = uuid.uuid4().hex
        item = {'id': aid, 'label': label, 'role': role, 'type': kind,
                'parent': parent, 'parent_hash': parent_hash, 'hash': None, 'model_id': None,
                'generation_model': body.get('generation_model') or None, 'notes': body.get('notes', ''),
                'created_at': now(), 'objective': objective, 'prompt': prompt, 'feedback': feedback,
                'context_condition': 'user-recorded; external chat isolation is not verified'}
        with self.lock:
            publish(self.folder / aid / 'attempt.json', item, exclusive=True)
        return self.detail(aid)

    def record(self, aid, body):
        with self.lock:
            item = self.detail(aid)
            folder = self.folder / aid
            if not (folder / 'attempt.json').exists() or (folder / 'response.json').exists():
                raise ValueError('Prepare a new attempt; existing artifacts and responses cannot be overwritten')
            weights = parse_edits(body['response'], role=item['role'])
            destination = confined(self.root / 'weights' / 'generated' / f"{item['role']}-{aid}.npz", self.root / 'weights')
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_weights(destination, weights)
            try:
                publish(folder / 'response.json', {'response': body['response'], 'hash': weights.content_hash,
                        'model_id': identity(destination.relative_to(self.root)), 'recorded_at': now(),
                        'generation_model': body.get('generation_model') or item.get('generation_model'),
                        'notes': body.get('notes', item.get('notes', ''))}, exclusive=True)
            except Exception:
                destination.unlink()
                raise
            return self.detail(aid)
