"""Loopback-only research dashboard, separate from the credentialed player API."""
import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .ui_common import wire
from .ui_experiments import Experiments
from .ui_games import ActiveConflict, Games
from .ui_reports import Reports


class Body(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Prepare(Body):
    label: str = Field(min_length=1, max_length=200)
    role: Literal['shooting', 'placement']
    type: Literal['one-shot', 'improvement']
    parent_model_id: str | None = None
    report_ids: list[str] = Field(default_factory=list, max_length=20)
    objective: str = Field(default='', max_length=20_000)
    generation_model: str = Field(default='', max_length=200)
    notes: str = Field(default='', max_length=20_000)


class Response(Body):
    response: str = Field(max_length=8*1024*1024)
    generation_model: str = Field(default='', max_length=200)
    notes: str = Field(default='', max_length=20_000)


class Player(Body):
    kind: Literal['random', 'hunt', 'weights']
    model_id: str | None = None
    placement_model_id: str | None = None


class NewGame(Body):
    request_id: str = Field(min_length=1, max_length=128)
    seed: str = '42'
    players: list[Player] = Field(min_length=2, max_length=2)
    speed: Literal[1, 2, 4, 8, 16] = 4
    starting_player: Literal[0, 1] = 0


class Control(Body):
    action: Literal['play', 'pause', 'step', 'speed', 'finish']
    speed: Literal[1, 2, 4, 8, 16] | None = None


class Comparison(Body):
    report_ids: list[str] = Field(min_length=2, max_length=10)


def create_ui_app(root: Path | None = None):
    root = Path(root or Path.cwd()).resolve()
    reports = Reports(root)
    experiments = Experiments(root, reports)
    games = Games(root, experiments)

    @asynccontextmanager
    async def lifespan(app):
        games.start_worker()
        try:
            yield
        finally:
            games.close()

    app = FastAPI(title='Battleship research dashboard', lifespan=lifespan)
    app.state.games, app.state.experiments, app.state.reports = games, experiments, reports
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]'])

    @app.middleware('http')
    async def local_boundary(request: Request, call_next):
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            expected = f'{request.url.scheme}://{request.headers.get("host", "")}'
            origin = request.headers.get('origin')
            if (origin and origin != expected) or request.headers.get('sec-fetch-site') == 'cross-site':
                return JSONResponse({'error': 'Cross-origin changes are not allowed.'}, status_code=403)
            if 'application/json' not in request.headers.get('content-type', ''):
                return JSONResponse({'error': 'Use application/json.'}, status_code=415)
            # Bound chunked requests as well as Content-Length before parsing JSON.
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 9*1024*1024:
                    return JSONResponse({'error': 'Request exceeds 9 MiB.'}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        if isinstance(error, ActiveConflict):
            return JSONResponse({'error': str(error), 'game_id': error.game_id}, status_code=409)
        return JSONResponse({'error': str(error)}, status_code=422)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, error):
        return JSONResponse({'error': 'Requested record or required file was not found.'}, status_code=404)

    @app.exception_handler(OSError)
    async def storage_error(request, error):
        return JSONResponse({'error': 'Local storage could not be read or written. Check available disk space and permissions.'}, status_code=500)

    @app.get('/ui-api/models')
    def models():
        return experiments.models()

    @app.get('/ui-api/attempts')
    def attempts():
        return wire(experiments.attempts())

    @app.get('/ui-api/attempts/{aid}')
    def attempt(aid: str):
        return wire(experiments.detail(aid))

    @app.post('/ui-api/attempts')
    def prepare(body: Prepare):
        return wire(experiments.prepare(body.model_dump()))

    @app.post('/ui-api/attempts/{aid}/response')
    def record(aid: str, body: Response):
        return wire(experiments.record(aid, body.model_dump()))

    @app.get('/ui-api/attempts/{aid}/prompt')
    def prompt(aid: str):
        item = experiments.detail(aid)
        if not item.get('snapshot'):
            raise FileNotFoundError('Prompt is not recorded for this historical artifact')
        return PlainTextResponse(item['snapshot']['prompt'], headers={'Content-Disposition': f'attachment; filename="attempt-{aid}.txt"'})

    @app.get('/ui-api/runs')
    def runs():
        return wire(reports.list())

    @app.post('/ui-api/compare')
    def compare(body: Comparison):
        return wire(reports.compare(body.report_ids))

    @app.get('/ui-api/runs/{rid}')
    def run(rid: str):
        return wire({**reports.detail(rid), 'distributions': reports.distributions(rid)})

    @app.get('/ui-api/runs/{rid}/trials')
    def trials(rid: str, offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=100),
               phase: str = 'benchmark', query: str = '', candidate: str = '', opponent: str = '',
               family: str = '', outcome: Literal['', 'win', 'loss'] = ''):
        return wire(reports.trials(rid, offset=offset, limit=limit, phase=phase, query=query,
                                   candidate=candidate, opponent=opponent, family=family, outcome=outcome))

    @app.get('/ui-api/runs/{rid}/trials/{tid}')
    def replay(rid: str, tid: str):
        return wire(reports.replay(rid, tid))

    @app.get('/ui-api/games')
    def saved_games():
        return wire(games.list())

    @app.post('/ui-api/games')
    def start(body: NewGame):
        return wire(games.create(body.model_dump()))

    @app.get('/ui-api/games/{gid}')
    def snapshot(gid: str):
        return wire(games.snapshot(gid))

    @app.post('/ui-api/games/{gid}/control')
    def control(gid: str, body: Control):
        return wire(games.control(gid, body.action, body.speed))

    @app.get('/ui-api/games/{gid}/events')
    async def events(gid: str, request: Request, after: int = -1):
        games.snapshot(gid)  # Resolve errors before starting streaming response.
        try:
            revision = int(request.headers.get('last-event-id', after))
        except ValueError:
            raise ValueError('Invalid event revision')
        async def stream():
            nonlocal revision
            heartbeat = 0
            while not await request.is_disconnected():
                state = games.snapshot(gid)
                if state['revision'] > revision:
                    revision = state['revision']
                    yield f'id: {revision}\ndata: {json.dumps(wire(state))}\n\n'
                if state['status'] not in ('running', 'paused'):
                    break
                heartbeat += 1
                if heartbeat % 50 == 0:
                    yield ': keep-alive\n\n'
                await asyncio.sleep(.1)
        return StreamingResponse(stream(), media_type='text/event-stream', headers={'X-Accel-Buffering': 'no'})

    static = Path(__file__).with_name('static')
    app.mount('/assets', StaticFiles(directory=static), name='assets')

    @app.get('/')
    def index():
        return FileResponse(static / 'index.html')

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(f'Battleship dashboard: http://127.0.0.1:{args.port}', flush=True)
    uvicorn.run(create_ui_app(args.root), host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
