# Local experiment dashboard

Run from the repository root:

```sh
make ui
```

Open [the dashboard](http://127.0.0.1:8002). Use `make ui UI_PORT=8003` for a different
port. The existing project `.venv` needs the development dependencies, including
NumPy. The target never installs packages or downloads browser assets. Stop it
with Ctrl-C. The dashboard binds only to `127.0.0.1`, uses one process, and is
separate from the player server on port 8000.

## Experiments

The experiment asks a generation model to choose numerical network weights in a
single response. Generation uses no tools, code execution, dataset, gradients, or
training. The imported fixed network subsequently selects actions with NumPy;
there is no language-model call for each move. Shooting and placement have
independent contracts and objectives.

**One-shot generation** includes a complete role-specific contract, with tensors
initialized to zero. **Fresh-context improvement** includes that contract, a
previous artifact, selected evaluation reports, and an explicit improvement
objective in a new chat. It requests a complete replacement edits block from
zero, never a patch. The app records these conditions but cannot verify the
external chat's isolation or tool use.

1. Select **Prepare attempt**, choose the role and method, and enter a label.
2. For improvement, select a parent artifact, at least one of its linked reports,
   and an objective. The complete weights are exported as decimal literals and
   round-trip checked before preparing the prompt.
3. Preview, copy, or download the saved prompt and use it in a separate fresh chat.
   Nothing calls a model provider or sends this package anywhere automatically.
4. Paste the full response or load a text file. Optionally record the generation
   model and notes. The existing importer validates the final `edits` block.
5. Run the displayed evaluation commands in a terminal. Refresh the dashboard to
   link reports automatically by artifact content hash.

Prepared prompts and feedback snapshots are immutable. A successful response is
also immutable and preserves the exact pasted text. Invalid imports create no
usable artifact; correct the input and retry. To generate another candidate,
prepare another attempt. Very large artifacts that cannot fit the importer's
text limits fail explicitly rather than being truncated.

The initial q1–q3 shooting and q1 placement artifacts appear as historical attempts.
Unknown generation models, prompts, methods, and parentage remain unknown. A
filename does not prove how an artifact was generated. Custom text model files
use `placement` in the filename to declare placement; other text files default
to shooting unless only the placement contract accepts them. Binary artifacts
validate their embedded network role. Put custom models under `weights/`.

## Results and comparisons

The library reads `.battleship/evaluations/*/` and `.battleship/shooting/*/`, using
manifest metadata to distinguish report types. Existing files are never changed.
It displays existing confidence intervals and paired comparisons without
recalculating uncertainty. Charts include shots distributions derived from
recorded benchmark trials.

- Head-to-head: win rate, first/second-player results, hit rate, and shots taken at
  termination. Losing shot counts are incomplete searches, not shots to clear.
- Shooting: shots to clear a target fleet; **lower is better**, grouped by policy
  and layout family.
- Placement: shots required to destroy the defender; **higher is better**, grouped
  by strategy and attacker. Fixture counterfire is omitted from the viewer.

Select reports in the Results library and click **Compare selected**. A
conservative compatibility check requires matching type, schedule, contract
versions, benchmark configuration (including seeds and sample sizes), and the
other network's hash when relevant. Incompatible runs remain inspectable, but
receive no improvement score. Matching configurations permit descriptive
comparison; they do not establish cross-run statistical significance.

Incomplete runs remain visible with their failure metadata. Records default to
the benchmark phase; select **all** to inspect smoke-only reports. Malformed
record lines are skipped with warnings, and indexes refresh when files change.
Model links use content hashes even when manifest paths no longer exist.

## Game viewer

Start a new game with either seat configured as random, hunt/target, or saved
shooting weights, plus optional independent placement weights. The seed defaults
to `42`, player one starts, and playback defaults to four shots per second.
A typical match takes about 20–30 seconds at that rate. Supported rates are
1, 2, 4, 8, and 16 shots per second. No turn timeout is enabled.

The two boards show fleets receiving incoming shots; their counters show each
owner's outgoing shooting performance. The next attacker is highlighted.
Single-attacker tests show one target board. Misses, hits, sinks, and the latest
shot have distinct visual markers.

- **Pause / Play** controls the simulation while following an active live game.
- **Next** advances exactly one shot and leaves live play paused.
- **Finish instantly** removes simulation pacing and saves the completed game.
- Seeking or stepping backward enters historical playback without modifying live
  play. **Return to live** follows the latest recorded turn again.
- Replay state, counters, sunk disclosures, and the winner use only the selected
  history prefix. **Reveal recorded fleets** intentionally reveals saved layouts.
- Historical match records do not contain both full layouts. Unavailable fleet
  cells remain unknown; opening a replay never reruns the original model.
- **Rematch** repeats the configuration and seed; **New seed** creates a fresh
  matchup. Only one running or paused UI game is allowed at a time.

Space toggles play/pause and arrow keys navigate replay when focus is outside a
form control. Controls have keyboard labels and focus outlines; reduced-motion
preferences disable shot animations.

Closing the browser does not stop the runner. Restarting the service marks
unfinished games interrupted, retaining their latest durably saved events.
Recording failures stop further shots and surface an error. The last event whose
write failed may be visible in the current process but is not guaranteed to survive
a restart.

## Files and service boundary

- `.battleship/experiments/<id>/attempt.json`: exact prepared prompt, feedback,
  objective, and user-recorded provenance.
- `.battleship/experiments/<id>/response.json`: exact validated response and
  resulting artifact identity.
- `weights/generated/<role>-<id>.npz`: new, non-overwriting model artifacts.
- `.battleship/ui-games/<id>.json`: configuration, both fleets, event history,
  revision, status, and outcome; atomically persisted after every shot/control.

The `/ui-api` endpoints expose models, attempts, prompt download/response import,
reports, paginated trials, replay details, comparisons, and games. A live game's
`events` endpoint sends complete revisioned snapshots over SSE and supports
`Last-Event-ID`; it reconnects without rerunning turns. Seed values cross the
browser boundary as decimal strings to preserve uint64 precision. Live games
use the `ui-game-1.0` seed schedule with independent fleet and per-seat streams,
plus the policy factory's independent placement stream.

This is a trusted local observer service. It exposes complete fleets of its own
UI games, never player bearer tokens. Policies still receive only authorized
observations. File selection uses opaque IDs, resolved paths are confined to
configured roots, imported responses are rendered as text, and cross-origin
mutations and unrecognized Host headers are rejected. There are no accounts,
remote hosting, provider integrations, benchmark job controls, or human-play
controls in this release. Run one dashboard process per workspace.

## Validation

```sh
.venv/bin/python -m pytest -q
node tests/test_ui_replay.mjs
```

Node is needed only for the standalone replay-state check, not to run the UI.
The full Python suite needs loopback socket access for existing integration tests.

## Public static version

Use `make ui-static` for a GitHub Pages-compatible build with published experiment
data and browser games. See the [static-site guide](github-pages.md) for deployment
and snapshot updates. The static build does not require the local UI service.
