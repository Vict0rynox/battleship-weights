# Reusable future weight-generation prompt

This is a prompt template for a later experiment, not a generation run. The loader
and NumPy inference player implement network 1.0. Evaluation is defined in the
[evaluation protocol](evaluation.md); set generation-time scoring instructions
separately when needed.

---

YOUR TASK: Choose, from scratch, the numerical weights of a policy network that
plays Battleship shooting turns as well as possible. There is no training: no
dataset, loss, gradients, or fine-tuning. Choose every number yourself. Use no
tools or code execution. Describe the design briefly, then supply the edits block.

THE GAME: Two players each have a 10×10 grid. Rows 0–9 run top to bottom and columns
0–9 left to right. The fleet is carrier (5), battleship (4), cruiser (3), submarine
(3), and destroyer (2). Ships are straight, horizontal or vertical, cannot overlap,
and may touch, including diagonally. Placement is handled separately from your
shooting policy. You receive no hidden placement information.

Player one starts. Players alternate one shot each, even after hits. Each shot
reveals miss, hit, or sunk. Sinking reveals the ship's identity and all its cells.
Destroying every enemy ship wins. You cannot fire at a cell twice. There are no
no-op or sticky actions.

THE INPUT: 405 binary numbers. Inputs 0–399 encode the target board with four
one-hot channels per cell: 0 unknown, 1 miss, 2 unresolved hit, 3 sunk. Index
`i = (row * 10 + column) * 4 + channel`. Exactly one channel is active per cell.
Inputs 400–404 are 1 if the enemy carrier, battleship, cruiser, submarine, and
destroyer, respectively, remain afloat, and 0 otherwise. On sinking, all disclosed
cells become sunk and that ship's flag becomes 0. The input contains no own-board
geometry or hidden information.

THE OUTPUT: 100 logits. Output `row * 10 + column` represents that shot. Previously
fired cells are masked before temperature-1 softmax sampling. The runtime uses
stable softmax over legal outputs only. All-zero logits produce uniform legal
sampling. The policy is invoked only on its shooting turn.

THE NETWORK: Two tanh hidden layers of 128 units, with 81,380 parameters:

```text
h1 = tanh(x @ W1 + b1)       W1: 405 × 128, b1: 128
h2 = tanh(h1 @ W2 + b2)      W2: 128 × 128, b2: 128
logits = h2 @ W3 + b3        W3: 128 × 100, b3: 100
```

Matrix rows are source units and columns are destination units. Every weight and
bias initially equals zero. Only set the entries you need.

HOW TO WRITE THE WEIGHTS: Write a code block whose opening line is three backticks
followed by `edits`. Use one statement per line. `#` starts a comment, and lists may
continue over multiple lines. Tensors: `W1`, `b1`, `W2`, `b2`, `W3`, `b3`.

```edits
W1[12, 0] = 2.5
W1[0:4, 3] = [1, 0, -1, 0]
b1[0:4] = -1
W1[3:400:40, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
W3[:, 2] *= 2
```

Ranges `a:b` are half-open; `a:b:s` takes every `s`th index; `:` selects an axis.
The whole column `c` in channel `ch` corresponds to input indexes
`c*4+ch : 400 : 40`; row `r` in channel `ch` corresponds to
`r*40+ch : r*40+40 : 4`. Compute these expressions before writing actual edits;
the block permits literal numerical indexes only.

Scalars fill selections. Lists must exactly match the selected shape. Support
`=`, `+=`, `-=`, and `*=`. Apply statements in order. Assignment overwrites earlier
values. Only the final `edits` block is read. Values must be plain decimal numbers
with absolute value at most 1,000,000, including intermediate results. No exponent
notation, expressions, imports, loops, function calls, NaN, or infinity. Indexes
must be nonnegative and in bounds, slice steps positive, and selections nonempty.

The importer rejects the entire block if any statement is invalid;
unreadable statements will not be skipped. Nothing in the block is executed as
Python. These examples illustrate syntax, not a recommended Battleship policy.
