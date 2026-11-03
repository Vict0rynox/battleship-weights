# Fixed-weight contract — network 1.0

The optional `battleship-client[weights]` extra implements this contract with a
restricted edits parser, observation encoder, and NumPy CPU float64 inference.
Weight generation is separate; strength evaluation follows the
[evaluation protocol](evaluation.md).

This network chooses shots. A separate [placement network](placement-weight-contract.md)
can construct the fleet using independent weights. It consumes only public target
knowledge from observation version **1.0**. Without a placement model, the player
uses automatic server placement; evaluation is specified separately from this network contract.

## Inputs: 405 binary numbers

For each target cell, use four one-hot channels:

| Channel | Meaning | API value |
|---|---|---|
| 0 | Unknown / never fired upon | `unknown` |
| 1 | Miss | `miss` |
| 2 | Unresolved hit | `hit` |
| 3 | Sunk ship cell | `sunk` |

For rows and columns 0–9, input index is:

```text
i = (row * 10 + column) * 4 + channel
```

The first 400 numbers are interleaved channels in row-major order. Exactly one
channel is active per cell. Inputs 400–404 are afloat flags, in order: carrier,
battleship, cruiser, submarine, destroyer. A flag is 1 when that name appears in
`remaining_enemy_ships`, otherwise 0. Cruiser and submarine have separate flags
despite sharing a length.

Own fleet geometry, incoming shots, credentials, seeds, and hidden enemy state are
not inputs. All 405 numbers can be derived from the public player observation.
The current target board already incorporates sinking disclosures.

## Network and action selection

```text
h1 = tanh(x @ W1 + b1)       W1: 405 × 128, b1: 128
h2 = tanh(h1 @ W2 + b2)      W2: 128 × 128, b2: 128
logits = h2 @ W3 + b3        W3: 128 × 100, b3: 100
```

There are 81,380 parameters: 51,840 + 128 + 16,384 + 128 + 12,800 + 100.
All parameters start at zero. Matrix row indexes denote source units; columns
denote destination units. The NumPy runner uses float64 arithmetic.

Output `row * 10 + column` is the logit for that shot. On an active shooting turn:

1. Keep only outputs whose corresponding `legal_shot_mask` entries are true.
2. Subtract the largest legal logit, exponentiate, and normalize (temperature 1).
3. Sample one legal shot using player-local randomness, separate from fleet RNG.

There are no sticky actions or no-op outputs. Do not invoke inference during
waiting/placement, on the other seat's turn, or after completion. An all-zero
network samples uniformly over legal shots. An empty legal set on an alleged
active turn is an integration error, not a request to fire an arbitrary cell.

## HOW TO WRITE THE WEIGHTS

Write a code block whose opening line is three backticks followed by `edits`.
Use one statement per line. A `#` starts a comment; lists may span multiple lines.
Only the **last** `edits` block of a response is read. The six available tensors are
`W1`, `b1`, `W2`, `b2`, `W3`, and `b3`.

```edits
W1[12, 0] = 2.5
W1[0:4, 3] = [1, 0, -1, 0]
b1[0:4] = -1
W1[3:400:40, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
W3[:, 2] *= 2
```

- `a:b` is half-open; `a:b:s` selects every `s`th index; `:` selects an axis.
- All cells in column `c`, channel `ch`: `W1[c*4+ch:400:40, unit]`.
- All cells in row `r`, channel `ch`: `W1[r*40+ch:r*40+40:4, unit]`.
- These formulas explain indexing; actual edit statements must use computed
  numerical indexes, not expressions or variables.
- A scalar fills the selected range/block. Lists must match the selection shape
  exactly; do not rely on general NumPy broadcasting.
- Support `=`, `+=`, `-=`, and `*=` with scalar or shape-matched numerical values.
- Statements apply in order. Assignment overwrites previous values; augmented
  assignment updates the existing values.
- Values are plain signed decimal numbers, including integers, with absolute value
  at most 1,000,000. No exponent notation, NaN, infinity, or expressions.

### Loader validation

Parse a restricted grammar; never execute the text as Python, use `eval`/`exec`,
or load pickle. Permit only the tensor names, numerical indexes/slices, numerical
values/lists, comments, and supported assignment operations described above.

Require nonnegative in-bounds indexes and slice bounds, positive slice steps,
nonempty selections, exact shapes, and finite results within ±1,000,000 after
every operation. Reject the **entire import** if any statement is invalid; do not
silently skip lines or retain partially applied edits. This strict behavior differs
from examples that skip unreadable statements.

The importer accepts a full response containing an `edits` block or raw edits
text. An unclosed final edits block is rejected. Empty raw text or an empty edits
block yields the all-zero network. Limits are 8 MiB of UTF-8 input and 100,000
statements in the selected block. Errors identify source line and column.

## Artifacts — format 1.0

`battleship weights import INPUT --output model.npz` validates the input and saves
an artifact without a session or running server. Existing destinations, including
symlinks, are never overwritten. The complete file is published atomically.

An artifact is an NPZ archive with exactly seven NPY members: `W1`, `b1`, `W2`,
`b2`, `W3`, `b3`, and `manifest` (each with `.npy` suffix). Tensors have the shapes
above and float64 dtype. The manifest is a Unicode scalar containing JSON with
exactly `artifact_version`, `network_version`, `observation_version` (all `"1.0"`),
and the lowercase SHA-256 hex string `content_hash`. Pickled/object arrays are
forbidden. File size and total expanded archive size are limited to 2 MiB; array
headers and payload sizes are validated before numerical loading.

The content hash is independent of ZIP timestamps and compression. Hash the
following byte sequence using SHA-256:

1. The three version fields as sorted-key, compact UTF-8 JSON, then a newline.
2. For each tensor in order `W1`, `b1`, `W2`, `b2`, `W3`, `b3`: compact JSON
   `[name, shape_as_list, "<f8"]`, then a newline, then its C-order little-endian
   float64 bytes. There are no separators between these bytes and the next header.

Loading verifies the manifest, shapes, dtypes, finite parameter bounds, and hash.
Unsupported versions and malformed artifacts fail without a fallback. The hash
checks content integrity; it is not a signature or proof of origin.

## Python and command-line use

```python
from battleship_client.policies import play_policy
from battleship_client.weights import WeightPlayer, load_weights

player = WeightPlayer(load_weights("model.npz"), seed=42)
# After creating/joining with an SDK client:
# result = play_policy(client, player)
```

The weights module also exports `NetworkWeights`, `parse_edits`, `save_weights`,
and `encode_observation`. `NetworkWeights.tensors` contains owned, immutable
arrays; `content_hash` exposes the deterministic parameter hash. The encoder
returns a `(405,)` float64 array. `WeightPlayer.logits` exposes the 100 logits;
`shot_probabilities` exposes a length-100 vector with zero mass on illegal shots.
`choose_shot` uses a private NumPy random generator. A seed repeats decisions for
the same observation sequence and numerical environment; cross-platform bitwise
identity is not guaranteed.

```sh
battleship --session guest.session.json bot weights --weights model.npz --seed 42
```

Paths ending in `.npz` (case-insensitive) are loaded strictly as artifacts. All
other weight source paths are UTF-8 edits or full response text. Missing NumPy,
invalid weights, and inference failures surface as errors, with no random-policy
fallback. All-zero weights are suitable for integration tests, not strength claims.

## Evaluation

The [evaluation protocol](evaluation.md) defines opponent schedules, placements,
paired games, confidence intervals, failure handling, and reports. No score,
strength claim, or benchmark guarantee is part of network contract 1.0 itself.
