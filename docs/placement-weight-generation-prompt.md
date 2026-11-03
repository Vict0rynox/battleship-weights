# Placement weight-generation prompt

This template defines a later experiment, not a weight-generation or training run.
The adapter implements [placement-1.0](placement-weight-contract.md). Assess any
result with the defensive evaluator on fresh trials.

---

YOUR TASK: Choose fixed numerical weights for a Battleship placement network,
optimizing how many shots opponents need to destroy the complete fleet. There is
no training, dataset, loss, gradient calculation, or fine-tuning. Choose the numbers
yourself without tools or code execution. Briefly explain the design, then supply
one `edits` block. Shooting is handled by a separate model.

THE GAME: A 10×10 board, rows and columns 0–9. Place carrier (5), battleship (4),
cruiser (3), submarine (3), destroyer (2), in that order. Ships extend horizontally
right or vertically down, cannot overlap or leave the board, and may touch.
The runtime constructs the entire fleet before submitting it. Sinking reveals a
ship's identity and all its cells. The placement policy receives no opponent data.

INPUTS: 505 binary features. Inputs 0–499 contain five occupancy channels per cell,
one per ship in the order above. Index `(row * 10 + column) * 5 + ship_index` is 1
when that ship occupies that cell; empty cells contain five zeros. Inputs 500–504
are one-hot flags identifying the current ship. The partial fleet contains only
ships already selected during this invocation.

NETWORK: CPU float64, two tanh hidden layers. Matrices index source units by row
and destination units by column. All 107,080 parameters begin at zero.

```text
h1 = tanh(x @ W1 + b1)       W1: 505 × 128, b1: 128
h2 = tanh(h1 @ W2 + b2)      W2: 128 × 128, b2: 128
logits = h2 @ W3 + b3        W3: 128 × 200, b3: 200
```

OUTPUTS: Action `2 * (row * 10 + column) + orientation`, with 0 horizontal and 1
vertical. Illegal actions are masked for overlap and bounds. The runtime applies
stable temperature-1 softmax and samples one legal action with private randomness.
It updates the partial fleet and repeats for the next ship. Zero weights uniformly
sample currently legal actions, not complete fleets. There is no whole-fleet
ranking, search, retry, or opponent simulation at inference time.

EDITS: Write a fenced code block labeled `edits`. The only tensor names are W1,
b1, W2, b2, W3, b3. One statement per line; `#` starts a comment; lists may span
lines. Indexes must be nonnegative, in bounds, and literal integers. Slices `a:b`
and `a:b:s` are half-open, require positive steps, and cannot select an empty range.
Use `:` for a whole axis. Use `=`, `+=`, `-=`, or `*=` with a scalar or a list
matching the exact selected shape. Statements apply in order. Only the last edits
block is read. Invalid statements reject the entire import.

Values must be plain signed decimals within ±1,000,000, including intermediate
results. No exponent notation, expressions, variables, imports, calls, loops,
NaN, or infinity. Compute all indexing expressions before emitting literal indexes.
Unspecified entries remain zero.

Syntax examples only:

```edits
W1[500, 0] = 1
b1[0] = -.5
W2[0, 0] = 1
W3[0, 0:2] = [.25, -.25]
b3[199] += .1
```

The examples make no claim about defensive strength.
