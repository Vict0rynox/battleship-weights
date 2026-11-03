# Generated-weight evaluation — report 1.1, seed schedule 1.0

The repository runner measures shooting strength locally using the authoritative
`Game` engine and the existing `WeightPlayer`, `RandomBot`, and `HuntTargetBot`.
It also verifies real HTTP integration and local/HTTP parity. It does not change weights or implement training. An optional independent
placement model can construct the candidate fleet; placement defense also has a
separate benchmark below.

For full searches without incoming fire or losses from own-fleet destruction, use
the separate [shooting-only benchmark](shooting-benchmark.md). Head-to-head shot
counts below stop at game termination; shooting-only trials always clear the fleet.

## Run

Use the development environment described in the README (including NumPy and
server dependencies). From the repository root:

For **2,000 games against hunt/target only**, provide just the weight path:

```sh
make evaluate-hunt WEIGHTS=weights/shooting_attempt_v1_q2.dat
```

This runs the model only, with 1,000 swapped-seat pairs, 10 local smoke games,
10 real HTTP smoke games, and two local parity replays. Smoke/parity results are
excluded from the 2,000 benchmark games. It generates a unique output directory
under `.battleship/evaluations/` and prints the report path. Repeated invocations
preserve previous reports. Override `PAIRS`, `SEED`, or `OUTPUT` if needed.

Equivalent direct command:

```sh
.venv/bin/python scripts/evaluate.py --weights weights/shooting_attempt_v1_q2.dat \
  --opponents hunt --model-only
```

`--opponents` selects `random`, `hunt`, or both; `--model-only` omits the zero-network
control. Without these options, the original full benchmark remains the default:

```sh
make evaluate OUTPUT=.battleship/evaluations/attempt-v1-q1
```

Equivalent direct command, with no package installation needed:

```sh
.venv/bin/python scripts/evaluate.py \
  --weights weights/shooting_attempt_v1_q1.dat --pairs 1000 --seed 42 \
  --output .battleship/evaluations/attempt-v1-q1
```

An explicit `OUTPUT` / `--output` must be a fresh directory; existing directories
and symlinks are rejected. The full `make evaluate` target requires `OUTPUT` and
accepts `EVAL_WEIGHTS`, `PAIRS`, and `SEED`. For an end-to-end
check with less statistical evidence, choose `PAIRS=10`. Smoke counts remain fixed.
The runner never downloads dependencies, resumes a run, or changes input files.
Loopback sockets must be permitted for HTTP verification.

## Schedule and fairness

In the full benchmark, the model and an all-zero network each play random and hunt/target:

1. **Local smoke:** five paired trials per matchup, 40 games total.
2. **HTTP smoke:** the same five paired trials per matchup, 40 games total.
   Replay the first pair of each HTTP matchup locally (eight additional games)
   and require identical public shot histories and winners.
3. **Benchmark:** 1,000 paired trials per matchup by default, 8,000 local games.

Smoke and parity games do not enter strength statistics. A pair contains two
games with candidate seats 0 and 1; seat 0 always starts. Each participant keeps
its fleet and policy seed across the pair. The runner repeats the placement RNG
seed and places the candidate first in both games. It uses automatic server
placement: uniform over currently legal placements for each ship, **not** uniform
over all complete fleet layouts. Ships may touch.

Each game creates fresh policy instances. Only validated weights are shared.
Candidate/placement seeds are shared across opponents and with the zero-network
control; opponent seeds depend on opponent identity. All-zero logits sample
uniformly over legal shots, exercising the network path as a random-policy
control. Python's random bot and the NumPy control need not produce the same
shot sequence even with identical numeric seeds.

Seed schedule `battleship-evaluation-1.0` hashes compact JSON arrays using SHA-256
and interprets the first eight digest bytes as an unsigned big-endian integer:

```text
[schedule_version, master_seed, phase, trial, "placement"]
[schedule_version, master_seed, phase, trial, "candidate"]
[schedule_version, master_seed, phase, trial, "opponent", opponent_name]
[schedule_version, master_seed, "statistics", matchup_name]
[schedule_version, master_seed, "statistics", "difference", opponent_name]
```

Benchmark `phase` is `benchmark`; local smoke, HTTP smoke, and parity use `smoke`.
Trial indexes start at zero. Inference remains NumPy float64 with temperature-1
sampling. Determinism is expected in the same numerical environment; cross-platform
bitwise identity is not guaranteed.

## Execution and failure behavior

The trusted runner owns local game objects and placement RNGs. Policies receive
only their own public `Observation` and choose shots on active turns. Actions are
checked against the legal-shot mask. No policy receives engine objects, hidden
enemy fleets, placement seeds, or shared RNGs. This is an observation boundary,
not isolation from hostile code in the same process.

HTTP smoke uses a dedicated ephemeral loopback Uvicorn server and the existing
SDK, with normal retry handling. No running gameplay server is reused. Completed
rooms and request caches are removed through private runner lifecycle code, and
clients/server are closed after completion or failure. No administrative API is
added. Neither execution path uses gameplay turn timeouts.

Errors and interruption stop the run, leave its reports marked `incomplete`, and
return a nonzero exit code. Completed games are retained; unfinished pairs are
excluded from statistics. Failures are never converted into losses or replacement
policies. `summary.json` records exception type and trial context without raw
HTTP exception strings or credentials. Fix the problem and use a new output
folder. A complete run that fails the strength criterion still exits zero.

## Reports and interpretation

- `manifest.json`: model/control content hashes, protocol versions, configuration,
  source SHA-256 fingerprints, seed schedule, runtime/dependency versions, and run
  status. Random match IDs, timestamps, and timing are not replay invariants.
- `games.jsonl`: flushed after each game; phase, matchup, trial, seats, seeds,
  winner, finish reason, shot/hit counts, and public shot history. No credentials
  or hidden fleet layouts are recorded. Seeds allow trusted replay.
- `summary.json` and `summary.md`: results per matchup, starting-seat win rates,
  hit rates, mean shots at termination, paired confidence intervals, aligned
  model-minus-control win-rate differences, HTTP status, and failure context.

Confidence intervals use 10,000 deterministic percentile-bootstrap resamples of
**whole pairs**, keeping both games together. Differences resample aligned
model/control pairs. Only complete pairs enter metrics; the summary counts any
unpaired completed games separately. The pair is the independent sampling unit,
not an individual game. Percentile intervals are descriptive uncertainty estimates;
very small custom runs should not support strong conclusions. If every pair has
the same score, the bootstrap interval collapses even in a large run. An observed
100% win rate and a collapsed interval do not establish a true 100% win probability.

The random strength criterion is **passed** only for a complete run whose model
win-rate interval has lower bound strictly above 50%. Otherwise it is **not
demonstrated**, or **incomplete** if execution failed. Hunt is a separate strength
reference without a pass threshold. There is no combined opponent score.
Hunt-only reports mark the random strength criterion **not evaluated** and show
only the requested matchups; model-only reports omit model-minus-control results.
The unchanged seed schedule allows comparison with earlier runs on the same trials.

Shot counts stop when either side wins, so a loser's count does not measure how
many shots it would need to finish. HTTP verification is reported separately from
shooting strength. Keep the weight hash and schedule fixed during evaluation;
tuning against these results requires fresh trials for subsequent confirmation.

## Two-model match evaluation

Pass `--placement-weights PATH` to `scripts/evaluate.py`, or set
`PLACEMENT_WEIGHTS=PATH` with `make evaluate` / `make evaluate-hunt`. The candidate
uses that placement model; the baseline opponent keeps automatic placement. Both
shooting candidates (model and zero) use the same placement model and placement
seed, and swapped-seat games preserve each participant's fleet. Placement sampling
uses a separate RNG derived from the candidate seed using the placement contract's
SHA-256 tag; it never consumes shooting randomness.

These runs use seed schedule `battleship-evaluation-two-model-1.0` with the original
schedule's stream parts. The statistics streams use that schedule too. The manifest
records the placement hash and distribution. Without placement weights, the
original schedule, shooting artifacts, and gameplay sequences are preserved.
Match results measure the combined player; they do not isolate placement strength.

## Placement defense

The placement testbench measures how long a fleet survives against a fixed
attacker. Report `placement-evaluation-1.1` and seed schedule
`battleship-placement-evaluation-1.1` extend the original evaluator. Existing
reports remain unchanged; results from different schedules are not aligned trials.

```sh
make evaluate-placement PLACEMENT_WEIGHTS=weights/placement_attempt_v1_q1.dat
make evaluate-placement PLACEMENT_WEIGHTS=weights/placement_attempt_v1_q1.dat TRIALS=10
make evaluate-placement PLACEMENT_WEIGHTS=weights/placement_attempt_v1_q1.dat \
  STRATEGIES="automatic edge" ATTACKERS=hunt
```

The default compares the supplied placement model with automatic, edge, center,
and touching placement against **hunt/target**, using 1,000 fleets per strategy:
**5,000 measured attacks**, plus smoke/parity checks. Set `TRIALS`, `SEED` (default
42), and optionally `OUTPUT=NEW_DIRECTORY`. Placement weights are always explicit.
The benchmark never trains, changes weights, or searches for a best fleet.

### Strategies and attackers

The supplied model is always included. `--strategies` / `STRATEGIES` select controls:

| Strategy | Definition |
|---|---|
| `automatic` | Authoritative sequential random placement; not uniform over complete fleets |
| `edge` | Every occupied cell has row or column in `{0,1,8,9}` |
| `center` | Every occupied cell has row and column in `{2,3,4,5,6,7}` |
| `touching` | Each ship after the first orthogonally touches an earlier ship |
| `zero` | Optional placement adapter control with all-zero weights, uniformly sampling currently legal actions |

Heuristic generators are shared with the shooting benchmark. Ship order,
orientation enumeration, sampling, and bounded whole-fleet restarts preserve its
original behavior. A strategy that exhausts 1,000 fleet attempts fails explicitly.
Every generated fleet is checked by the authoritative validator. Keep naturally
repeated layouts: resampling them would alter the strategy distribution.

`--attackers` / `ATTACKERS` accept `hunt`, `random`, and `weights`. With no explicit
selection, hunt is used; supplying `--attacker-weights` / `ATTACKER_WEIGHTS` adds the
fixed-weight attacker alongside hunt. An explicit selection containing `weights`
requires a shooting-weight file. Providing a file without selecting `weights` is
an error. Duplicate selections are removed and execution order is canonical.

```sh
# Compare both attackers on the same generated fleets.
make evaluate-placement PLACEMENT_WEIGHTS=weights/placement_attempt_v1_q1.dat \
  ATTACKER_WEIGHTS=weights/shooting_attempt_v1_q3.dat

# Isolate the shooting-model attacker.
make evaluate-placement PLACEMENT_WEIGHTS=weights/placement_attempt_v1_q1.dat \
  ATTACKERS=weights ATTACKER_WEIGHTS=weights/shooting_attempt_v1_q3.dat

.venv/bin/python scripts/evaluate_placement.py \
  --placement-weights weights/placement_attempt_v1_q1.dat \
  --strategies automatic edge center touching --attackers hunt --trials 1000 --seed 42
```

### Uncensored measurement and seeds

The metric is **attacker shots to destroy all 17 occupied cells**, between 17 and
100 inclusive; higher is better for placement defense. A trusted rollout uses the
normal authoritative engine with the attacker in seat 0. Its own fixture fleet is
horizontal, starting at column 0 on rows 0–4 in fleet order. The defender's script
fires at all water cells of this fixture in row-major order, then all occupied
cells in row-major order. It cannot destroy the attacker until shot 100. Since the
attacker starts, it always completes first, even when it needs 100 shots. These
are defensive measurements, not competitive win rates. Attacking policies receive
only authorized observations; no hidden target geometry is passed to inference.
The shooting network retains its unchanged public-target-only encoder.

Each trial generates one fleet per selected strategy and reuses it across all
attacker types. One attack is run for each strategy/attacker/trial combination.
Candidate identity is excluded from seed derivation so placement seeds and
attacker seeds align between controls. Each strategy initializes a fresh generator;
there is no RNG shared between strategies, attackers, or trials. Adding an attacker
or selecting fewer strategies cannot change existing fleets or attack sequences.
Generation, attack, and statistics streams are separate.

The schedule takes the first eight SHA-256 bytes, interpreted big endian, of
compact JSON arrays:

```text
[schedule, master_seed, phase, trial, "fleet"]
[schedule, master_seed, phase, trial, "attacker", attacker_name]
[schedule, master_seed, "statistics", matchup, "mean" or "median"]
[schedule, master_seed, "statistics", difference_name]
[schedule, master_seed, "statistics", difference_name, "percentage"]
```

`phase` is `benchmark` for measured trials and `smoke` for local and HTTP smoke.
Placement players use the fleet seed directly. Heuristic strategies use fresh
Python RNGs with that seed; attackers receive their own seed. Every attack starts
a fresh policy instance. A shared seed is an alignment mechanism, not a claim that
different strategies generate the same fleet or Python and NumPy share sequences.

Before measurement, run five local smoke trials and five real HTTP smoke trials
for every selected strategy/attacker combination. Replay the first HTTP trial
locally and require exact shot-history parity. HTTP uses complete explicit fleet
submission and a dedicated ephemeral server. Smoke/parity results are excluded
from statistics. Rooms, clients, and the server are cleaned up on success or failure.

### Reports and interpretation

Reports use fresh directories, defaulting to `.battleship/evaluations/placement-*`:

- `manifest.json`: placement/control and optional shooting hashes, network versions,
  strategy definitions, schedule, configuration, fixture, source fingerprints,
  runtime/dependencies, thread environment, and status.
- `games.jsonl`: flushed completed attacks with phase, trial, strategy, attacker,
  seeds, submitted-fleet hash, shot count, and public shot history.
- `summary.json` / `summary.md`: mean and median shots with 95% confidence intervals,
  P10/P90 shot counts, paired model-minus-control mean-shot and percentage
  improvements with 95% confidence intervals, primary comparison, HTTP status,
  attack counts, excluded trial counts, and failure context.

P10/P90 use linear-interpolated sample quantiles and describe fleet survivability,
not confidence bounds. Percentage improvement is
`100 * (model_mean - control_mean) / control_mean`. Positive improvements mean the
model requires more attacker shots to sink.

Intervals use 10,000 deterministic percentile bootstrap resamples of whole trials.
A trial block must contain every requested strategy/attacker combination to enter
statistics. Interrupted blocks are excluded as a whole, preserving alignment and
shared-fleet grouping. Mean-shot differences resample paired within-trial
model-minus-control counts. Percentage intervals resample model and control counts
using the same trial indexes and recompute the ratio of means.

The primary comparison is **model versus automatic against hunt**. On a complete
run, a paired mean-difference interval with lower bound strictly above zero is
`improved`; upper bound strictly below zero is `worse`; an interval containing zero
is `inconclusive`. If that comparison is not selected, its status is `not evaluated`.
An unfinished run cannot demonstrate improvement. Other comparisons are descriptive;
there is no combined attacker score. Small runs or collapsed intervals do not
establish general strength. Confirm any later tuned weights on a separate seed.

Failures and interruption stop the run with nonzero exit status, retain completed
records, and mark reports incomplete. Failure metadata contains exception types
and trial context, not credentials or raw HTTP exception strings. Preflight rejects
invalid selections and model files before creating the output directory. No invalid
model, exhausted generator, or failed inference falls back to another strategy.
