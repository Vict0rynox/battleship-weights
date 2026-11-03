# Sequential placement weights — placement-1.0

Placement constructs a complete legal fleet before play. Shooting observes public
target knowledge and selects shots. The two policies have independent weights and
random generators; either can be evaluated without attributing the other's strength
to it. This adapter performs inference only; it does not train or optimize weights.

## Inputs and network

Place ships in this fixed order: carrier (5), battleship (4), cruiser (3), submarine
(3), destroyer (2). Ships are horizontal or vertical, cannot overlap or leave the
10×10 board, and may touch. Each choice sees only the partial fleet constructed by
this invocation and the identity of the next ship.

Inputs are 505 binary float64 numbers. Inputs 0–499 contain five occupancy channels
per cell in the ship order above:

```text
index = (row * 10 + column) * 5 + ship_index
```

Rows and columns are 0–9. An occupied cell has a 1 in its ship's channel. Empty
cells have five zeros. Inputs 500–504 are one-hot flags identifying the current
ship. Only earlier ships appear in the partial fleet. Match identity, player seat,
readiness, own observation fields, opponent information, and seeds are not encoded.
The observation is used only to check that the seat has not committed a fleet and
is in the waiting or placement phase.

```text
h1 = tanh(x @ W1 + b1)       W1: 505 × 128, b1: 128
h2 = tanh(h1 @ W2 + b2)      W2: 128 × 128, b2: 128
logits = h2 @ W3 + b3        W3: 128 × 200, b3: 200
```

There are 107,080 parameters. Matrix rows represent source units; columns represent
destination units. All parameters initially equal zero. Arithmetic is NumPy CPU
float64, as for the existing shooting network.

## Actions and complete-fleet submission

Output `2 * (row * 10 + column) + orientation` represents a starting cell and an
orientation: 0 horizontal, 1 vertical. A ship extends right or down from its start.
For example, output 119 places a vertical ship starting at row 5, column 9 (J6).

For each ship, mask out actions that overlap the partial fleet or extend beyond
the board. Apply stable temperature-1 softmax over the remaining logits and sample
one action. Illegal outputs have exactly zero probability. Update the internal
partial fleet before choosing the next ship. Zero weights sample uniformly among
currently legal actions; this is **not** a uniform distribution over complete fleets.

There is always a legal action for each prefix under this fixed board, fleet, and
order. Before placing ships of lengths 4, 3, 3, and 2, at most 5, 9, 12, and 15 cells
are occupied. Across ten rows there are respectively 20, 30, 30, and 50 disjoint
horizontal segments of the required length, so those occupied cells cannot block
all choices. An empty legal mask indicates an integration error. There are no
restarts, retries with random policies, or full-fleet candidate ranking.

Validate the completed fleet locally and submit all five `ShipPlacement` objects
in one existing SDK `place(ships)` call. The server remains authoritative. No
partial fleet is sent to the server. Committed seats skip placement inference on
resume. Nonfinite inference and invalid model data fail explicitly.

## Weight text and artifacts

Use the same six tensor names and restricted edits grammar described in the
[shooting weight contract](weight-contract.md#how-to-write-the-weights), with the
placement shapes above. The final `edits` block is selected; raw edits are also
accepted. Limits remain 8 MiB UTF-8 input, 100,000 statements, finite values within
±1,000,000 after every operation, exact list shapes, and positive, bounded slices.
Invalid statements reject the entire import. No Python text is executed.

```sh
.venv/bin/python -m battleship_client weights import examples/placement-synthetic.edits \
  --role placement --output placement.npz
```

Placement NPZ artifacts contain exactly `W1.npy`, `b1.npy`, `W2.npy`, `b2.npy`,
`W3.npy`, `b3.npy`, and `manifest.npy`. The manifest is a Unicode JSON scalar with
exactly these fields:

```json
{
  "artifact_version": "1.1",
  "role": "placement",
  "network_version": "placement-1.0",
  "observation_version": "1.0",
  "content_hash": "<lowercase SHA-256>"
}
```

Hash all metadata fields except `content_hash` as sorted-key compact UTF-8 JSON,
then a newline. For each tensor in contract order, hash compact JSON
`[name, shape_as_list, "<f8"]`, then a newline, then C-order little-endian float64
bytes. The archive and total expanded size limits remain 2 MiB. Existing
destinations, including symlinks, are never overwritten; publication is atomic.
The loader checks member counts, headers, payload lengths, types, bounds, versions,
role, and hash. Pickled arrays are forbidden.

Shooting artifacts retain their original format 1.0 and hashes. Both roles default
to shooting in existing Python import/load calls; supply `role="placement"` for
placement. Raw edits have no embedded role: their selected role determines the
shapes. NPZ role mismatches are rejected, with no text fallback.

```python
from battleship_client.weights import PlacementWeightPlayer, load_weight_source

placement = PlacementWeightPlayer(
    load_weight_source("placement.npz", role="placement"), seed=42
)
# ships = placement.choose_placement(client.observe())
# client.place(ships)
```

`encode_placement(ships, current_ship)` returns the 505 features;
`placement_mask(ships, current_ship)` returns 200 booleans. The player exposes
`logits` and `placement_probabilities` for inspecting an ordered partial fleet.

## Composing gameplay

```sh
battleship --session guest.session.json bot weights \
  --weights shooting.npz --placement-weights placement.npz --seed 42

make bots BOT=weights WEIGHTS=shooting.npz PLACEMENT_WEIGHTS=placement.npz \
  OPPONENT=hunt OPPONENT_PLACEMENT_WEIGHTS=other-placement.npz
```

`--placement-weights` also works with random and hunt shooters. Omission retains
automatic server placement. `create_policy(kind, seed, weights_path,
placement_weights_path)` creates the composition; `ComposedPlayer` also accepts
separate shooting and placement policy objects directly.

The composition preserves the existing shooting seed. Its placement seed is the
first eight SHA-256 digest bytes, interpreted big endian, of compact UTF-8 JSON
`["battleship-placement-1.0", seed]`. With no seed, placement and shooting each
initialize an independent generator from entropy. Direct `PlacementWeightPlayer`
construction uses its supplied seed directly. In a fixed numerical environment,
the same weights and seed reproduce choices. Cross-platform bitwise identity is
not guaranteed.

See [defensive evaluation](evaluation.md#placement-defense) and the
[placement generation prompt](placement-weight-generation-prompt.md).
