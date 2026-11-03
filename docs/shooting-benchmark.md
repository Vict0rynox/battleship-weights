# Shooting-only benchmark — protocol 1.0

This benchmark measures aiming and fleet-clearing efficiency independently of
whether the player's own ships survive. There are no incoming shots. Every policy
continues until all five target ships and all 17 occupied cells are hit, requiring
between 17 and 100 legal shots. Lower shots-to-clear is better.

## Run and switch models

```sh
make shooting-bench WEIGHTS=weights/shooting_attempt_v1_q2.dat
make shooting-bench WEIGHTS=weights/shooting_attempt_v1_q1.dat
```

Only `WEIGHTS` is required. The default generates **2,000 distinct boards**, 500
per placement family, and runs the model, random bot, and hunt/target bot on each:
**6,000 complete shooting trials**. It creates a unique report directory beneath
`.battleship/shooting/` and prints its path. Each invocation preserves earlier runs.
This is a local benchmark; it needs no running server, HTTP checks, or sockets.
Use the project development environment with NumPy installed.

A smaller diagnostic run and a selected-family run:

```sh
make shooting-bench WEIGHTS=weights/shooting_attempt_v1_q2.dat BOARDS=40
make shooting-bench WEIGHTS=weights/shooting_attempt_v1_q2.dat BOARDS=100 PLACEMENTS="edge center"
```

Defaults: `BOARDS=2000`, `SEED=42`, and
`PLACEMENTS="random edge center touching"`. Optional `OUTPUT=NEW_DIRECTORY`
selects an explicit fresh destination; existing paths and symlinks are rejected.
`BOARDS` is the total layout count, not a per-policy count or a paired-game count.
At least one board per selected family is required. Counts are allocated evenly;
any remainder goes to the first selected families in the order listed below.

Direct invocation, including paths containing spaces:

```sh
.venv/bin/python scripts/shooting_bench.py --weights "weights/my model.dat" \
  --boards 2000 --seed 42 --placements random edge center touching
```

Raw edits, full response files, and `.npz` weight artifacts use the existing
validated loader. Neither the network nor its temperature-1 softmax sampling is
changed. Invalid weights fail before creating output. The existing head-to-head
`evaluate` and `evaluate-hunt` commands retain their separate rules and seed schedule.

## Target placements

| Family | Eligibility |
|---|---|
| `random` | Existing `random_fleet` generator |
| `edge` | Every occupied cell is in the outer two-cell border: row or column 0, 1, 8, or 9 |
| `center` | Every occupied cell is inside rows 2–7 and columns 2–7 |
| `touching` | Each ship after the first touches an earlier ship orthogonally, creating a connected cluster |

All families use the standard five ships in protocol order. Ships must be straight,
inside the board, and nonoverlapping; touching is allowed. Structured generators
sample uniformly among eligible legal placements for the next ship, restarting
fleet construction on a dead end. These distributions are **not uniform over
complete fleets**. Each layout has a private placement RNG.

Duplicate complete layouts are rejected across the entire selected suite, using
ship names and sorted cell coordinates as the canonical identity. Cruiser and
submarine remain distinguishable. Construction is bounded to 1,000 full-fleet
attempts per structured draw and 1,000 draws per unique layout. Exhaustion stops
the benchmark explicitly; it never silently reduces coverage.

Selected families are always generated in table order. Layout sampling happens
without replacement across the selected suite, so a different family selection
can affect duplicate rejection. To compare models, keep family selection, board
count, and seed fixed and verify the saved layout hash.

## Execution and seeds

The trusted runner uses `Game(..., shooting_only=True)`, an engine-only mode that
keeps player 0 on turn after each nonterminal shot. It places a fixed valid own
fleet solely to preserve the normal observation structure. No opponent shots occur.
The usual hit, miss, sinking disclosure, afloat flags, legal-shot masks, and victory
logic are reused. The mode is not exposed through HTTP, and ordinary games still
alternate turns. Turn timeouts are disabled.

Each policy receives only its own public observation, never the hidden target
layout, engine object, or placement seed. Only the trusted runner sees target
placements. The benchmark is not a sandbox for hostile policy code in the same
process. Model tensors are loaded once and shared immutably, while every policy
instance and its sampling RNG are recreated for each layout.

Seed schedule `battleship-shooting-1.0` takes the first eight SHA-256 digest bytes
as an unsigned big-endian integer from compact JSON:

```text
[schedule_version, master_seed, family, index, "placement"]
[schedule_version, master_seed, family, index, "policy", policy_name]
[schedule_version, master_seed, "statistics", policy_name, scope]
[schedule_version, master_seed, "statistics", "difference", baseline_name, scope]
```

Indexes start at zero within each family; policy names are `model`, `random`, and
`hunt`; scope is `overall` or the family name. Neither weight path nor content hash
enters seeds. Switching model files therefore preserves target layouts and model
sampling seeds. Numerical reproducibility is expected in the same environment;
bitwise identity across platforms is not guaranteed.

## Metrics

Reports show each policy overall and separately by family:

- Mean shots-to-clear with a 95% bootstrap interval, median, and 90th percentile.
- Mean shot number of the first hit and first sinking.
- Mean cumulative hits after 10, 25, and 50 shots. Once a fleet is cleared, later
  checkpoints retain 17 hits rather than penalizing the early finish.
- Mean misses, and aggregate hit rate: `17 × completed trials / total shots`.
- Model-minus-random and model-minus-hunt mean shot differences with 95% intervals.
  Negative differences favor the model.

Comparisons match the same saved layout, not independently sampled boards. Each
bootstrap uses 10,000 deterministic whole-layout resamples. Overall resamples
preserve each family's observed sample count, so mixture weights stay fixed.
Medians and P90 use NumPy's default linear quantile interpolation. All three
policies finish their searches; these are not the censored shot counts from
head-to-head games. There are no win rates or win-rate pass criteria.

Small samples, or identical scores on every sampled layout, can produce collapsed
bootstrap intervals. Results describe the selected placement distributions and
provide no measurement of defensive placement quality.

## Files and failure behavior

- `manifest.json`: status, model content hash, placement definitions and allocation,
  seed schedule, versions, runtime and dependency details, source fingerprints,
  and the SHA-256 of the complete `layouts.jsonl` file.
- `layouts.jsonl`: all target layouts, IDs, family indexes, and placement seeds.
- `trials.jsonl`: one flushed record per complete trial, including layout ID,
  policy and sampling seed, metrics, and shot history.
- `summary.json` and `summary.md`: metrics, matched comparisons, generated/expected/
  completed counts, failure context, and timing.

All layouts are generated and saved before shooting starts. Invalid actions,
inference errors, generation failures, and Ctrl-C stop the run with nonzero exit
status. Complete records are retained and reports are marked `incomplete`.
Individual policy metrics include completed trials; each model/baseline comparison
uses only layouts where both trials completed. Never substitute another policy or
convert an execution failure into a performance score. Use a fresh directory for
a retry; resume is not supported.
