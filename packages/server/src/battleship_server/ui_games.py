"""Single-match local runner. The worker owns pacing; the browser owns replay position."""
import hashlib
import json
import random
import threading
import time
import uuid

from battleship_client.policies import create_policy
from battleship_protocol import Coordinate
from battleship_server.engine import Game
from .ui_common import confined, now, publish, read_json

SPEEDS = (1, 2, 4, 8, 16)


def stream_seed(master, tag):
    return int.from_bytes(hashlib.sha256(json.dumps(['ui-game-1.0', master, tag]).encode()).digest()[:8], 'big')


class ActiveConflict(ValueError):
    def __init__(self, game_id):
        self.game_id = game_id
        super().__init__('An active game already exists')


class Games:
    def __init__(self, root, experiments):
        self.root, self.experiments = root, experiments
        self.folder = root / '.battleship' / 'ui-games'
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.active = None
        self.game = None
        self.policies = None
        self.thread = None
        for path in self.folder.glob('*.json'):
            confined(path, self.folder)
            record = read_json(path)
            if record['status'] in ('running', 'paused'):
                record.update(status='interrupted', error='Service stopped before the match finished.', revision=record['revision'] + 1)
                publish(path, record)

    def start_worker(self):
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        with self.changed:
            self.changed.notify_all()
        if self.thread:
            self.thread.join(timeout=10)
        with self.lock:
            if self.active and self.active['status'] in ('running', 'paused'):
                self.active.update(status='interrupted', error='Service stopped.', revision=self.active['revision'] + 1)
                self.persist()

    def list(self):
        with self.lock:
            records = [read_json(confined(p, self.folder)) for p in self.folder.glob('*.json')]
            if self.active:
                records = [r for r in records if r['id'] != self.active['id']] + [self.snapshot()]
            return [{k: v for k, v in r.items() if k not in ('history', 'fleets')} for r in
                    sorted(records, key=lambda r: r['created_at'], reverse=True)]

    def snapshot(self, gid=None):
        with self.lock:
            if self.active and (gid is None or gid == self.active['id']):
                return json.loads(json.dumps(self.active))
            if not gid or len(gid) != 32 or not all(c in '0123456789abcdef' for c in gid):
                raise FileNotFoundError('Game not found')
            return read_json(confined(self.folder / f'{gid}.json', self.folder))

    def persist(self):
        try:
            publish(self.folder / f"{self.active['id']}.json", self.active)
        except OSError:
            self.active.update(status='error', error='Recording failed. Simulation has stopped.',
                               revision=self.active['revision'] + 1)
            self.changed.notify_all()
            raise
        self.changed.notify_all()

    def create(self, body):
        with self.lock:
            # Idempotent start, including a browser retry after a lost response.
            for existing in self.list():
                if existing.get('request_id') == body['request_id']:
                    return self.snapshot(existing['id'])
            if self.active and self.active['status'] in ('running', 'paused'):
                raise ActiveConflict(self.active['id'])
            seed_text = body['seed']
            if not isinstance(seed_text, str) or not seed_text.isdecimal() or len(seed_text) > 20 or int(seed_text) > 2**64-1:
                raise ValueError('Seed must be a decimal uint64 string')
            master = int(seed_text)
            speed = body['speed']
            if speed not in SPEEDS:
                raise ValueError('Unsupported speed')
            starter = body['starting_player']
            policies, names, configs = [], [], []
            for seat, spec in enumerate(body['players']):
                kind = spec['kind']
                shooting = placement = None
                config = dict(spec)
                if kind == 'weights':
                    shooting, info = self.experiments.resolve(spec.get('model_id'), 'shooting')
                    config['shooting_hash'] = info['hash']
                    names.append(info['label'])
                else:
                    names.append(kind)
                if spec.get('placement_model_id'):
                    placement, info = self.experiments.resolve(spec['placement_model_id'], 'placement')
                    config['placement_hash'] = info['hash']
                policies.append(create_policy(kind, stream_seed(master, f'player-{seat}'), shooting, placement))
                configs.append(config)
            gid = uuid.uuid4().hex
            game = Game(gid, rng=random.Random(stream_seed(master, 'fleet')))
            game.join()
            for seat, policy in enumerate(policies):
                choose = getattr(policy, 'choose_placement', None)
                game.place(seat, choose(game.observe(seat)) if choose else None)
            # Game's standard default is seat zero; this local runner explicitly chooses the starter.
            game.turn = starter
            fleets = [[{'name': name, 'cells': [{'row': r, 'column': c} for r, c in cells]}
                       for name, cells in fleet.items()] for fleet in game.fleets]
            self.active = {'id': gid, 'label': f'{names[0]} vs {names[1]}', 'players': names,
                           'kind': 'match', 'created_at': now(), 'configuration': {**body, 'players': configs},
                           'request_id': body['request_id'], 'status': 'running', 'speed': speed,
                           'revision': 0, 'turn': starter, 'history': [], 'fleets': fleets, 'result': None}
            self.game, self.policies = game, policies
            self.next_turn = time.monotonic() + 1/speed
            self.persist()
            return self.snapshot()

    def advance(self):
        seat = self.game.turn
        observation = self.game.observe(seat)
        shot = self.policies[seat].choose_shot(observation)
        if not isinstance(shot, Coordinate) or not observation.legal_shot_mask[shot.row][shot.column]:
            raise ValueError('Policy returned an illegal shot')
        self.game.fire(seat, shot)
        self.active.update(history=[e.model_dump(mode='json') for e in self.game.history],
                           turn=self.game.turn, result=self.game.result.model_dump(mode='json') if self.game.result else None,
                           revision=self.active['revision'] + 1)
        if self.game.result:
            self.active['status'] = 'finished'
        self.persist()

    def control(self, gid, action, speed=None):
        with self.lock:
            if not self.active or gid != self.active['id'] or self.active['status'] not in ('running', 'paused'):
                raise ValueError('This game is no longer active')
            if action == 'pause':
                self.active['status'] = 'paused'
            elif action == 'play':
                self.active['status'] = 'running'
            elif action == 'step':
                self.active['status'] = 'paused'
                try:
                    self.advance()
                except Exception:
                    self.fail()
                    raise
                return self.snapshot()
            elif action == 'speed':
                if speed not in SPEEDS:
                    raise ValueError('Unsupported speed')
                self.active['speed'] = speed
            elif action == 'finish':
                self.active.update(status='running', instant=True)
            else:
                raise ValueError('Unknown game control')
            self.active['revision'] += 1
            self.next_turn = time.monotonic() + 1/self.active['speed']
            self.persist()
            return self.snapshot()

    def fail(self):
        self.active.update(status='error', error='Simulation or recording failed; no further shots will run.',
                           revision=self.active['revision'] + 1)
        try:
            self.persist()
        except OSError:
            pass

    def work(self):
        while not self.stop_event.is_set():
            with self.changed:
                if self.active and self.active['status'] == 'running':
                    delay = self.next_turn - time.monotonic()
                    if self.active.get('instant') or delay <= 0:
                        try:
                            self.advance()
                            self.next_turn = time.monotonic() + 1/self.active['speed']
                        except Exception:
                            self.fail()
                        continue
                    self.changed.wait(timeout=min(delay, .5))
                else:
                    self.changed.wait(timeout=.5)
