.DEFAULT_GOAL := help

VENV ?= .venv
PYTHON := $(VENV)/bin/python
HOST ?= 127.0.0.1
PORT ?= 8000
URL ?= http://$(HOST):$(PORT)
SESSION ?= host.session.json
BOT ?= hunt
OPPONENT ?= random
WEIGHTS ?=
EVAL_WEIGHTS ?= weights/shooting_attempt_v1_q1.dat
PAIRS ?= 1000
BOARDS ?= 2000
PLACEMENTS ?= random edge center touching
OUTPUT ?=
PLACEMENT_WEIGHTS ?=
OPPONENT_PLACEMENT_WEIGHTS ?=
ATTACKER_WEIGHTS ?=
STRATEGIES ?= automatic edge center touching
ATTACKERS ?=
TRIALS ?= 1000
OPPONENT_WEIGHTS ?=
SEED ?= 42
MATCH ?=
TURN_TIMEOUT ?=

# Run repository source directly; installing Battleship packages is unnecessary.
export PYTHONPATH := $(CURDIR)/packages/protocol/src:$(CURDIR)/packages/client/src:$(CURDIR)/packages/server/src

.PHONY: help venv check-client check-server server client create join bot attach-bot play bots evaluate evaluate-hunt shooting-bench evaluate-placement

help:
	@printf '%s\n' \
	  'make ui-static                    Build the public GitHub Pages site into dist/' \
	  'make ui-snapshot                  Refresh the public results/weights snapshot' \
	  'make ui                           Open the local research dashboard service (port 8002)' \
	  'make play                         Start server if needed; play against hunt bot' \
	  'make play BOT=random              Play against the random bot' \
	  'make play BOT=weights WEIGHTS=model.npz   Play against a weight player' \
	  'make bots                         Watch a hunt-vs-random game (final boards)' \
	  'make evaluate OUTPUT=.battleship/evaluations/first   Benchmark saved weights locally' \
	  'make evaluate-hunt WEIGHTS=weights/model.dat   2,000 games against hunt; saves a fresh report' \
	  'make shooting-bench WEIGHTS=weights/model.dat  Shooting-only benchmark on 2,000 target boards' \
	  'make bots BOT=weights WEIGHTS=model.edits OPPONENT=weights OPPONENT_WEIGHTS=other.npz' \
	  'make evaluate-placement PLACEMENT_WEIGHTS=weights/placement_attempt_v1_q1.dat   Benchmark fleet defense' \
	  'make server                       Run just the server in this terminal' \
	  'make create SESSION=me.session.json   Create a room and save a human seat' \
	  'make join MATCH=ID SESSION=you.session.json   Join using the invitation prompt' \
	  'make client SESSION=me.session.json   Play/resume an existing seat' \
	  'make bot SESSION=you.session.json BOT=hunt    Run a bot in an existing seat' \
	  'make attach-bot MATCH=ID BOT=hunt  Join an open seat and run a bot' \
	  'make venv                         Create .venv only if missing; installs nothing' \
	  '' \
	  'Options: HOST=127.0.0.1 PORT=8000 URL=http://127.0.0.1:8000 VENV=.venv SEED=42' \
	  'Server timer: make server TURN_TIMEOUT=30. Ctrl-C stops launcher-owned processes.'

venv:
	@if [ ! -x "$(PYTHON)" ]; then python3 -m venv "$(VENV)"; fi

check-client: venv
	@"$(PYTHON)" -c 'import battleship_client, battleship_protocol' || \
	  { printf '%s\n' 'Client dependencies are missing from this venv. See README.md; nothing was installed automatically.' >&2; exit 1; }

check-server: venv
	@"$(PYTHON)" -c 'import battleship_server.app, uvicorn' || \
	  { printf '%s\n' 'Server dependencies are missing from this venv. See README.md; nothing was installed automatically.' >&2; exit 1; }

server: check-server
	@"$(PYTHON)" -m battleship_server --host "$(HOST)" --port "$(PORT)" $(if $(TURN_TIMEOUT),--turn-timeout "$(TURN_TIMEOUT)")

client: check-client
	@"$(PYTHON)" -m battleship_client --url "$(URL)" --session "$(SESSION)" play

create: check-client
	@"$(PYTHON)" -m battleship_client --url "$(URL)" --session "$(SESSION)" create

join: check-client
	@test -n "$(MATCH)" || { printf '%s\n' 'Usage: make join MATCH=ID SESSION=guest.session.json' >&2; exit 1; }
	@"$(PYTHON)" -m battleship_client --url "$(URL)" --session "$(SESSION)" join "$(MATCH)"

bot: check-client
	@"$(PYTHON)" -m battleship_client --url "$(URL)" --session "$(SESSION)" bot "$(BOT)" --seed "$(SEED)" $(if $(filter weights,$(BOT)),--weights "$(WEIGHTS)") $(if $(PLACEMENT_WEIGHTS),--placement-weights "$(PLACEMENT_WEIGHTS)")

attach-bot: check-client
	@test -n "$(MATCH)" || { printf '%s\n' 'Usage: make attach-bot MATCH=ID BOT=hunt' >&2; exit 1; }
	@"$(PYTHON)" scripts/launch.py attach-bot --url "$(URL)" --match "$(MATCH)" --bot "$(BOT)" --seed "$(SEED)" $(if $(filter weights,$(BOT)),--weights "$(WEIGHTS)") $(if $(PLACEMENT_WEIGHTS),--placement-weights "$(PLACEMENT_WEIGHTS)")

play: check-server check-client
	@"$(PYTHON)" scripts/launch.py play --url "$(URL)" --bot "$(BOT)" --seed "$(SEED)" $(if $(filter weights,$(BOT)),--weights "$(WEIGHTS)") $(if $(PLACEMENT_WEIGHTS),--placement-weights "$(PLACEMENT_WEIGHTS)")

bots: check-server check-client
	@"$(PYTHON)" scripts/launch.py bots --url "$(URL)" --bot "$(BOT)" --opponent "$(OPPONENT)" --seed "$(SEED)" $(if $(filter weights,$(BOT)),--weights "$(WEIGHTS)") $(if $(filter weights,$(OPPONENT)),--opponent-weights "$(OPPONENT_WEIGHTS)") $(if $(PLACEMENT_WEIGHTS),--placement-weights "$(PLACEMENT_WEIGHTS)") $(if $(OPPONENT_PLACEMENT_WEIGHTS),--opponent-placement-weights "$(OPPONENT_PLACEMENT_WEIGHTS)")

evaluate: check-server check-client
	@test -n "$(OUTPUT)" || { printf '%s\n' 'Usage: make evaluate OUTPUT=NEW_DIRECTORY [EVAL_WEIGHTS=PATH PAIRS=1000 SEED=42]' >&2; exit 1; }
	@"$(PYTHON)" scripts/evaluate.py --weights "$(EVAL_WEIGHTS)" --pairs "$(PAIRS)" --seed "$(SEED)" --output "$(OUTPUT)" $(if $(PLACEMENT_WEIGHTS),--placement-weights "$(PLACEMENT_WEIGHTS)")

evaluate-hunt: check-server check-client
	@test -n "$(WEIGHTS)" || { printf '%s\n' 'Usage: make evaluate-hunt WEIGHTS=PATH [PAIRS=1000 SEED=42 OUTPUT=NEW_DIRECTORY]' >&2; exit 1; }
	@"$(PYTHON)" scripts/evaluate.py --weights "$(WEIGHTS)" --opponents hunt --model-only --pairs "$(PAIRS)" --seed "$(SEED)" $(if $(OUTPUT),--output "$(OUTPUT)") $(if $(PLACEMENT_WEIGHTS),--placement-weights "$(PLACEMENT_WEIGHTS)")

shooting-bench: check-client
	@test -n "$(WEIGHTS)" || { printf '%s\n' 'Usage: make shooting-bench WEIGHTS=PATH [BOARDS=2000 SEED=42 PLACEMENTS="random edge center touching" OUTPUT=NEW_DIRECTORY]' >&2; exit 1; }
	@"$(PYTHON)" scripts/shooting_bench.py --weights "$(WEIGHTS)" --boards "$(BOARDS)" --seed "$(SEED)" --placements $(foreach placement,$(PLACEMENTS),"$(placement)") $(if $(OUTPUT),--output "$(OUTPUT)")

evaluate-placement: check-server check-client
	@test -n "$(PLACEMENT_WEIGHTS)" || { printf '%s\n' 'Usage: make evaluate-placement PLACEMENT_WEIGHTS=PATH [TRIALS=1000 SEED=42 STRATEGIES="automatic edge center touching" ATTACKERS=hunt OUTPUT=NEW_DIRECTORY]' >&2; exit 1; }
	@"$(PYTHON)" scripts/evaluate_placement.py --strategies $(foreach strategy,$(STRATEGIES),"$(strategy)") $(if $(ATTACKERS),--attackers $(foreach attacker,$(ATTACKERS),"$(attacker)")) --placement-weights "$(PLACEMENT_WEIGHTS)" --trials "$(TRIALS)" --seed "$(SEED)" $(if $(ATTACKER_WEIGHTS),--attacker-weights "$(ATTACKER_WEIGHTS)") $(if $(OUTPUT),--output "$(OUTPUT)")

UI_PORT ?= 8002
.PHONY: ui
ui: check-server check-client
	$(PYTHON) -m battleship_server.ui --port $(UI_PORT) --root "$(CURDIR)"

STATIC_PYTHON ?= $(if $(wildcard $(PYTHON)),$(PYTHON),python3)
REPLAYS_PER_GROUP ?= 0
.PHONY: ui-static ui-snapshot ui-static-preview
ui-snapshot: check-server check-client
	$(PYTHON) scripts/export_ui.py snapshot --replays-per-group $(REPLAYS_PER_GROUP)

ui-static:
	$(STATIC_PYTHON) scripts/export_ui.py build

ui-static-preview: ui-static
	@printf '%s\n' 'Static preview: http://127.0.0.1:8003'
	$(STATIC_PYTHON) -m http.server 8003 --bind 127.0.0.1 --directory dist
