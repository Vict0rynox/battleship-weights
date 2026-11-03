# Battleship game foundation

A local, authoritative Battleship server with a separately installable player SDK,
terminal play, and random and hunt/target bots. Humans and bots use the same HTTP API.

The optional weights extra adds safe edits import and NumPy CPU float64 inference
for fixed network 1.0. A local paired benchmark evaluates saved weights against
random and hunt/target, with HTTP parity checks. An independent sequential placement
network can construct fleets before play. Weight generation happens in a separate
chat; the local dashboard prepares prompts and records responses. Tournaments
remain separate work.

## Experiment dashboard and live viewer

```sh
make ui
```

Open [the local dashboard](http://127.0.0.1:8002) to track generation attempts,
prepare complete one-shot or fresh-context improvement prompts, import model
responses, compare existing results, replay recorded trials, and start live
model-vs-bot or model-vs-model games. Games default to four shots per second with
pause, step, speed, and fleet-reveal controls. Use `UI_PORT=8003` for another port.

Generation happens in a separate chat; evaluation commands remain in the CLI.
No Node build or external assets are required. See the [dashboard guide](docs/ui.md)
for provenance, persistence, report compatibility, and playback behavior.

## Public static site / GitHub Pages

```sh
make ui-static
make ui-static-preview
```

The static site includes current experiment summaries, aggregate charts,
and browser-only model/bot games. Raw trial histories stay local by default. It builds from the included `site-data/`
snapshot with Python's standard library; no server or Node build is required.
Run `make ui-snapshot` to refresh public data after new local evaluations.

A GitHub Pages workflow is included. After pushing the project, set the
repository's **Settings → Pages → Source** to **GitHub Actions**, then run
**Publish experiment dashboard**. See the [Pages guide](docs/github-pages.md)
for setup, public-data boundaries, and differences from the local dashboard.

## Evaluate generated weights

To measure shooting efficiency without incoming fire or own-fleet losses:

```sh
make shooting-bench WEIGHTS=weights/shooting_attempt_v1_q2.dat
```

The model, random bot, and hunt/target bot each clear the same 2,000 target layouts
across random, edge, center, and touching-ship families. Reports show shots-to-clear,
aiming metrics, and matched comparisons. Change `WEIGHTS` to switch models; use
`BOARDS=40` for a quick run. Reports get a fresh directory automatically. See the
[shooting-only benchmark](docs/shooting-benchmark.md) for options and interpretation.

For 2,000 local games against hunt/target using any saved weight file:

```sh
make evaluate-hunt WEIGHTS=weights/shooting_attempt_v1_q2.dat
```

This creates a unique report directory under `.battleship/evaluations/` and prints
its path. It includes local/HTTP smoke and parity checks, and skips the zero-weight
control. Defaults are `PAIRS=1000` (two games per pair) and `SEED=42`; optionally
provide `OUTPUT=NEW_DIRECTORY`. The seed schedule matches the full benchmark.

For the full model/control benchmark against both opponents:

```sh
make evaluate OUTPUT=.battleship/evaluations/attempt-v1-q1
```

This evaluates `weights/shooting_attempt_v1_q1.dat` in 8,000 local benchmark games,
plus local and real HTTP smoke checks. It saves reproducible game records and
JSON/Markdown reports. Use a fresh output directory; override `EVAL_WEIGHTS`,
`PAIRS`, or `SEED` as needed. See the [evaluation protocol](docs/evaluation.md)
for pairing, uncertainty estimates, and the success criterion.

## Launch from source with Make (no additional installation)

The existing project-local `.venv` supplies the dependencies. Make selects that
Python and sets `PYTHONPATH` automatically; shell activation and installing the
Battleship packages are unnecessary. No Make target downloads or installs packages.
`make venv` only creates an empty virtual environment if one is missing; missing
third-party dependencies produce an explanatory error.

For a human-vs-bot game, run:

```sh
make play
```

This starts a local server if needed, creates a fresh room, joins the hunt bot,
and opens your terminal player. Choose random/manual placement and enter shots.
Use `make play BOT=random` for the random opponent, or `make bots` for a single
automated hunt-vs-random game with final-board output. `SEED=42` controls bot actions.

Fresh private sessions and background logs are saved under `.battleship/runs/`;
your existing session files are never overwritten. On exit/Ctrl-C, the launcher
stops only its own children. If it started the server, that server and its in-memory
matches stop too. If it reused an existing server, that server remains running.

For separately managed components, use different terminals:

```sh
make server
make create SESSION=me.session.json
make client SESSION=me.session.json
```

Give the printed match ID/invitation to the other human, who runs:

```sh
make join MATCH=MATCH_ID SESSION=guest.session.json
make client SESSION=guest.session.json
```

Or attach a bot to that room directly (paste the invitation at the prompt):

```sh
make attach-bot MATCH=MATCH_ID BOT=hunt
```

To run a bot in a seat that is already joined:

```sh
make bot SESSION=guest.session.json BOT=random
```

Use fresh session paths when creating/joining a new room. `make help` lists the
targets. Set `PORT=8001` for another local port, `URL=http://HOST:PORT` to connect
to an existing server elsewhere, or `VENV=/path/to/venv` to select another venv.
`make server TURN_TIMEOUT=30` enables a shooting timer. `make client` resumes
`host.session.json` by default; override `SESSION` for other saved players.

## Install

Python 3.10 or newer is required. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-tested.txt -r requirements-dev.txt
```

For a runtime-only installation:

```sh
python -m pip install ./packages/protocol ./packages/client ./packages/server
```

The three distributions are `battleship-protocol`, `battleship-client`, and
`battleship-server`. A remote/local player only needs the first two:

```sh
python -m pip install ./packages/protocol ./packages/client
```

The client does not depend on FastAPI, Uvicorn, or the engine. Distributable wheels
can be built with `python -m pip wheel --wheel-dir dist ./packages/protocol
./packages/client ./packages/server` (one command, on one line). Only share the
client/protocol wheels and their dependencies with players who should not receive
engine source.

## Play

Start the server in one terminal:

```sh
battleship-server
```

The server listens on `http://127.0.0.1:8000`. Interactive API documentation is at
`http://127.0.0.1:8000/docs`, with the machine-readable contract at
`http://127.0.0.1:8000/openapi.json`.

Create a room in another terminal:

```sh
battleship --session host.session.json create
```

Give the printed match ID and invitation to player two. They run:

```sh
battleship --session guest.session.json join MATCH_ID
```

Paste the invitation at the private prompt. Each player can now run:

```sh
battleship --session host.session.json play
```

Player two uses `guest.session.json` instead. Choose random or manual placement,
then shoot with coordinates `A1`–`J10` (columns A–J, rows 1–10). Manual placement
accepts a starting cell and direction, such as `A1 H` or `J5 V`.

Session files contain private bearer credentials, are created with mode `0600`,
and are never overwritten. Use a different session path for each player and room.
Share the invitation, not a session file. Ctrl-C leaves the match running; use
`play` again with the saved session to resume. Run `resign` to concede explicitly.
Stopping the server loses all matches.

## Bots and individual commands

After joining the room, replace either player's `play` command with:

```sh
battleship --session guest.session.json bot hunt --seed 42
```

For bot–bot play, run both commands in separate terminals after creating/joining:

```sh
battleship --session host.session.json bot random --seed 7
battleship --session guest.session.json bot hunt --seed 42
```

Bots use server automatic placement unless `--placement-weights` is supplied.
`--seed` controls shooting randomness and, when selected, a separate placement-model
random generator; it does not control or reveal server automatic placement. The
hunt/target bot follows hits and plausible line extensions, otherwise using a checkerboard search. It does not
assume ships are separated by water.

Individual commands are also available:

```sh
battleship --session host.session.json show
battleship --session host.session.json show --json
battleship --session host.session.json place
battleship --session host.session.json place --manual
battleship --session host.session.json fire D5
battleship --session host.session.json resign
```

`place` commits a fleet only once; the examples above are alternatives. Global
options (`--session`, `--url`) go before the command. `--url` defaults to localhost
when creating/joining and to the saved server URL for later commands.

## Fixed-weight players

Install the optional dependency into your chosen environment (Make never installs
it automatically):

```sh
.venv/bin/python -m pip install -c requirements-tested.txt './packages/client[weights]'
```

The development requirements already include this extra. Human, random, and hunt
modes remain usable with the base client installation and no NumPy.

Import raw `.edits` text or a full model response containing an `edits` block:

```sh
.venv/bin/python -m battleship_client weights import examples/synthetic.edits --output model.npz
```

No session or server is needed for import. An invalid statement rejects the entire
import, and an existing output file is never overwritten. The command prints the
artifact's deterministic parameter-content hash. The included synthetic example
exercises the network; it makes no claim about playing strength.

Use the artifact, or pass an edits/response file directly:

```sh
make play BOT=weights WEIGHTS=model.npz
make bots BOT=weights WEIGHTS=examples/synthetic.edits
make bots BOT=weights WEIGHTS=model.npz OPPONENT=weights OPPONENT_WEIGHTS=other.npz
make bot SESSION=guest.session.json BOT=weights WEIGHTS=model.npz
make attach-bot MATCH=MATCH_ID BOT=weights WEIGHTS=model.npz
```

The equivalent installed CLI for an existing seat is:

```sh
battleship --session guest.session.json bot weights --weights model.npz --seed 42
```

Launchers validate the selected players before starting a server or creating/joining
a match. Weight shooters use only public target knowledge. Their seed controls a private
shot-sampling generator; server automatic placement randomness is separate. See
the [weight contract](docs/weight-contract.md) for the grammar,
Python interfaces, artifact format, and numerical behavior.

## Independent placement weights

A sequential placement model constructs all five legal ships before submitting the
fleet. Use a separate file from the shooting model:

```sh
.venv/bin/python -m battleship_client weights import examples/placement-synthetic.edits \
  --role placement --output placement.npz
make bots BOT=weights WEIGHTS=shooting.npz PLACEMENT_WEIGHTS=placement.npz OPPONENT=hunt
make bots BOT=hunt PLACEMENT_WEIGHTS=placement.npz \
  OPPONENT=random OPPONENT_PLACEMENT_WEIGHTS=other-placement.npz
```

Placement weights work with random, hunt, and weight shooters. Existing commands
without placement weights retain automatic server placement. The CLI option is
`bot KIND --placement-weights PATH`; Make also forwards placement weights through
`play`, `bot`, and `attach-bot`. Both model files are validated before match activity.
Placement and shooting use independent random generators.

Measure placement defense separately from shooting:

```sh
make evaluate-placement PLACEMENT_WEIGHTS=weights/placement_attempt_v1_q1.dat
make evaluate-placement PLACEMENT_WEIGHTS=weights/placement_attempt_v1_q1.dat \
  ATTACKER_WEIGHTS=weights/shooting_attempt_v1_q3.dat TRIALS=10
make evaluate-hunt WEIGHTS=shooting.npz PLACEMENT_WEIGHTS=placement.npz PAIRS=10
```

The placement benchmark compares model, automatic random, edge, center, and
touching-ship fleets against hunt/target: 1,000 fleets each and 5,000 measured
attacks by default. It measures shots to sink the entire fleet; higher is better.
Use `STRATEGIES="automatic edge"` to choose controls (including optional `zero`)
and `ATTACKERS="hunt random"` to select attackers. Supplying `ATTACKER_WEIGHTS`
adds a shooting-weight attacker when `ATTACKERS` is omitted. Reports include
paired differences, percentage improvements, and a primary comparison against
automatic placement. `TRIALS=1000` and `SEED=42` are defaults.
Reports go to a fresh directory; optionally set `OUTPUT`. Synthetic weights test
integration and make no strength claim. See the [placement contract](docs/placement-weight-contract.md),
[defensive evaluation](docs/evaluation.md#placement-defense), and
[placement-generation prompt](docs/placement-weight-generation-prompt.md).

## Rules and deployment

- 10×10 grid; carrier 5, battleship 4, cruiser 3, submarine 3, destroyer 2.
- Horizontal/vertical straight ships; no overlap; touching is allowed.
- Player one starts after both seats are joined and both fleets are committed.
- Alternate one shot per turn, even after hits. No salvo or sticky actions.
- Sinking reveals that ship's identity and all its cells; other enemy cells stay hidden.
- Destroy the entire enemy fleet to win. Invalid actions never consume turns.

The server is a **single-process, in-memory** local service. Do not run multiple
workers: they would have independent match stores. There are no accounts,
persistence, spectator endpoints, or administrative hidden-state endpoints.

Optional configuration:

```sh
battleship-server --port 8001 --turn-timeout 30
```

Turn timeouts default to disabled. When enabled, each shooting turn has the
configured deadline; expiry forfeits the match. Placement has no timer, and invalid
actions do not reset a shooting deadline. `--host` is available, but remote TLS,
accounts, and deployment hardening are outside this stage.

Automatic placement chooses uniformly among current legal placements, in descending
ship length order, restarting if necessary. This is not a uniform distribution over
all complete fleets.

## Develop and extend

```sh
python -m pytest -q
```

Tests include real loopback HTTP connections, separate bot processes, terminal
flows, authentication, concurrency, timeouts, seeded reproducibility, and hidden
layout noninterference. Sandboxed environments must permit loopback sockets for
the HTTP integration tests.

- [Player API and SDK](docs/player-api.md)
- [Weight/network contract and NumPy player](docs/weight-contract.md)
- [Generated-weight evaluation](docs/evaluation.md)
- [Shooting-only benchmark](docs/shooting-benchmark.md)
- [Reusable future weight-generation prompt](docs/weight-generation-prompt.md)
- [Custom SDK-only player](examples/custom_player.py)

Protocol, observation, shooting network, and shooting artifact versions remain **1.0**.
Placement uses network **placement-1.0** and artifact **1.1**.
See `requirements-tested.txt` for the dependencies used during validation.

Players receive only authorized views, and the SDK installs without engine source.
This is an API/process boundary, not a sandbox for hostile code with access to the
server's operating-system account, files, or memory.
