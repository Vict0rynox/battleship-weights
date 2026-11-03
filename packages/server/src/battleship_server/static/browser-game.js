// Browser simulation uses its own reproducible RNG schedule, never benchmark parity.
export const FLEET = {
  carrier: 5,
  battleship: 4,
  cruiser: 3,
  submarine: 3,
  destroyer: 2,
};
const names = Object.keys(FLEET);
export function randomStream(seed, tag) {
  let value = 2166136261;
  for (const ch of `${BigInt(seed)}|browser-game-1.0|${tag}`)
    value = Math.imul(value ^ ch.charCodeAt(0), 16777619) >>> 0;
  if (!value) value = 1;
  return () => {
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    return (value >>> 0) / 4294967296;
  };
}
export function validateModel(model, role) {
  if (model.role !== role) throw new Error(`Expected ${role} weights`);
  const input = role === "placement" ? 505 : 405,
    output = role === "placement" ? 200 : 100;
  const shapes = {
    W1: [input, 128],
    b1: [128],
    W2: [128, 128],
    b2: [128],
    W3: [128, output],
    b3: [output],
  };
  for (const [name, shape] of Object.entries(shapes)) {
    const a = model.tensors?.[name];
    if (
      !Array.isArray(a) ||
      a.length !== shape[0] ||
      (shape.length === 2 &&
        a.some((row) => !Array.isArray(row) || row.length !== shape[1]))
    )
      throw new Error("Invalid model shape");
    const values = shape.length === 2 ? a.flat() : a;
    if (
      values.some(
        (v) =>
          typeof v !== "number" || !Number.isFinite(v) || Math.abs(v) > 1000000,
      )
    )
      throw new Error("Invalid model values");
  }
  return model;
}
export function logits(input, model) {
  const w = model.tensors;
  function layer(x, m, b, activation) {
    const result = b.slice();
    for (let i = 0; i < x.length; i++)
      if (x[i] !== 0)
        for (let j = 0; j < result.length; j++) result[j] += x[i] * m[i][j];
    if (activation)
      for (let j = 0; j < result.length; j++) result[j] = Math.tanh(result[j]);
    if (result.some((v) => !Number.isFinite(v)))
      throw new Error("Nonfinite network inference");
    return result;
  }
  return layer(
    layer(layer(input, w.W1, w.b1, true), w.W2, w.b2, true),
    w.W3,
    w.b3,
    false,
  );
}
export function sample(scores, legal, rng) {
  if (!legal.length) throw new Error("No legal actions");
  const max = Math.max(...legal.map((i) => scores[i]));
  const weights = legal.map((i) => Math.exp(scores[i] - max));
  let threshold = rng() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < legal.length; i++) {
    threshold -= weights[i];
    if (threshold < 0) return legal[i];
  }
  return legal.at(-1);
}
function cellsFor(row, column, length, vertical) {
  return Array.from({ length }, (_, i) => ({
    row: row + (vertical ? i : 0),
    column: column + (vertical ? 0 : i),
  }));
}
export function generateFleet(model, rng) {
  for (let attempt = 0; attempt < 1000; attempt++) {
    const ships = [],
      occupied = new Set();
    for (let n = 0; n < names.length; n++) {
      const name = names[n],
        length = FLEET[name],
        legal = [];
      for (let row = 0; row < 10; row++)
        for (let column = 0; column < 10; column++)
          for (let vertical = 0; vertical < 2; vertical++) {
            const cells = cellsFor(row, column, length, vertical);
            if (
              cells.every(
                (c) =>
                  c.row < 10 &&
                  c.column < 10 &&
                  !occupied.has(c.row * 10 + c.column),
              )
            )
              legal.push(2 * (row * 10 + column) + vertical);
          }
      if (!legal.length) {
        if (model) throw new Error("Placement model left no legal action");
        break;
      }
      let index;
      if (model) {
        const input = Array(505).fill(0);
        input[500 + n] = 1;
        ships.forEach((ship, i) =>
          ship.cells.forEach(
            (c) => (input[(c.row * 10 + c.column) * 5 + i] = 1),
          ),
        );
        index = sample(logits(input, model), legal, rng);
      } else index = legal[Math.floor(rng() * legal.length)];
      const row = Math.floor(index / 20),
        column = Math.floor(index / 2) % 10,
        cells = cellsFor(row, column, length, index % 2);
      ships.push({ name, cells });
      cells.forEach((c) => occupied.add(c.row * 10 + c.column));
    }
    if (ships.length === 5) return ships;
  }
  throw new Error("Could not place a legal fleet");
}
export function observation(history, attacker) {
  const board = Array(100).fill("unknown"),
    remaining = new Set(names);
  for (const e of history)
    if (e.attacker === attacker) {
      board[e.coordinate.row * 10 + e.coordinate.column] =
        e.outcome === "sunk" ? "hit" : e.outcome;
      if (e.outcome === "sunk") {
        remaining.delete(e.ship);
        for (const c of e.sunk_cells) board[c.row * 10 + c.column] = "sunk";
      }
    }
  return { board, remaining: [...remaining] };
}
// This policy interface accepts public target knowledge only; no enemy fleet.
export function chooseShot(kind, model, view, rng) {
  const legal = view.board.flatMap((v, i) => (v === "unknown" ? [i] : []));
  if (!legal.length) throw new Error("No legal shots");
  if (kind === "weights") {
    const input = Array(405).fill(0),
      channels = { unknown: 0, miss: 1, hit: 2, sunk: 3 };
    view.board.forEach((v, i) => (input[i * 4 + channels[v]] = 1));
    names.forEach(
      (name, i) => (input[400 + i] = Number(view.remaining.includes(name))),
    );
    return sample(logits(input, model), legal, rng);
  }
  let choices = legal;
  if (kind === "hunt") {
    const scores = legal.map((i) => {
      const r = Math.floor(i / 10),
        c = i % 10;
      let score = 0;
      for (const [dr, dc] of [
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
      ]) {
        const row = r + dr,
          col = c + dc;
        if (
          row >= 0 &&
          row < 10 &&
          col >= 0 &&
          col < 10 &&
          view.board[row * 10 + col] === "hit"
        ) {
          score++;
          const row2 = r + 2 * dr,
            col2 = c + 2 * dc;
          if (
            row2 >= 0 &&
            row2 < 10 &&
            col2 >= 0 &&
            col2 < 10 &&
            view.board[row2 * 10 + col2] === "hit"
          )
            score++;
        }
      }
      return score;
    });
    const max = Math.max(...scores);
    choices = max
      ? legal.filter((_, i) => scores[i] === max)
      : legal.filter((i) => (Math.floor(i / 10) + (i % 10)) % 2 === 0);
    if (!choices.length) choices = legal;
  } else if (kind !== "random") throw new Error("Unknown policy");
  return choices[Math.floor(rng() * choices.length)];
}
export class BrowserMatch {
  constructor(settings, models, id) {
    if (
      !/^[0-9]{1,20}$/.test(settings.seed) ||
      BigInt(settings.seed) > 18446744073709551615n
    )
      throw new Error("Seed must be a decimal uint64 string");
    if (
      ![1, 2, 4, 8, 16].includes(settings.speed) ||
      ![0, 1].includes(settings.starting_player) ||
      settings.players.length !== 2
    )
      throw new Error("Invalid match settings");
    this.models = models;
    this.random = [0, 1].map((seat) =>
      randomStream(settings.seed, `shooting-${seat}`),
    );
    const fleets = settings.players.map((p, seat) =>
      generateFleet(
        models[seat].placement,
        randomStream(settings.seed, `placement-${seat}`),
      ),
    );
    const labels = settings.players.map((p, seat) =>
      p.kind === "weights" ? models[seat].shooting.label : p.kind,
    );
    this.data = {
      id,
      label: labels.join(" vs "),
      kind: "match",
      players: labels,
      created_at: new Date().toISOString(),
      configuration: {
        ...settings,
        schedule: "browser-game-1.0",
        execution: "browser; not a Python benchmark",
      },
      request_id: settings.request_id,
      status: "running",
      speed: settings.speed,
      revision: 0,
      turn: settings.starting_player,
      history: [],
      fleets,
      result: null,
    };
  }
  advance() {
    const d = this.data;
    if (!["running", "paused"].includes(d.status)) return;
    const seat = d.turn,
      spec = d.configuration.players[seat];
    const index = chooseShot(
      spec.kind,
      this.models[seat].shooting,
      observation(d.history, seat),
      this.random[seat],
    );
    const coordinate = { row: Math.floor(index / 10), column: index % 10 };
    const hit = d.fleets[1 - seat].find((s) =>
      s.cells.some(
        (c) => c.row === coordinate.row && c.column === coordinate.column,
      ),
    );
    const fired = new Set(
      d.history
        .filter((e) => e.attacker === seat)
        .map((e) => e.coordinate.row * 10 + e.coordinate.column),
    );
    fired.add(index);
    const sunk =
      hit && hit.cells.every((c) => fired.has(c.row * 10 + c.column));
    d.history.push({
      number: d.history.length + 1,
      attacker: seat,
      coordinate,
      outcome: sunk ? "sunk" : hit ? "hit" : "miss",
      ship: sunk ? hit.name : null,
      sunk_cells: sunk ? hit.cells : [],
    });
    if (
      d.fleets[1 - seat].every((s) =>
        s.cells.every((c) => fired.has(c.row * 10 + c.column)),
      )
    ) {
      d.result = { winner: seat, reason: "fleet_destroyed" };
      d.status = "finished";
      d.turn = null;
    } else d.turn = 1 - seat;
    d.revision++;
  }
}
