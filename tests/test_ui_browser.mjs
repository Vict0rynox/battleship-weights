// No browser tooling or dependencies: test the pure JS inference / simulation.
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
const root = new URL("../", import.meta.url);
const source = readFileSync(
  new URL("packages/server/src/battleship_server/static/browser-game.js", root),
  "utf8",
);
const runtimeURL = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const {
  BrowserMatch,
  randomStream,
  observation,
  chooseShot,
  generateFleet,
  logits,
  validateModel,
} = await import(runtimeURL);
const catalog = JSON.parse(
  readFileSync(new URL("site-data/index.json", root), "utf8"),
);
const byRole = (role) => catalog.models.filter((m) => m.role === role);
const load = (m) =>
  validateModel(
    JSON.parse(
      readFileSync(new URL(`site-data/models/${m.id}.json`, root), "utf8"),
    ),
    m.role,
  );
const shooting = load(byRole("shooting")[0]),
  placement = load(byRole("placement")[0]);
const settings = {
  request_id: "test",
  seed: "18446744073709551615",
  speed: 4,
  starting_player: 1,
  players: [{ kind: "weights" }, { kind: "hunt" }],
};
function complete(spec, models) {
  const game = new BrowserMatch(spec, models, "test");
  while (game.data.status === "running") {
    game.advance();
    assert.ok(game.data.history.length <= 199);
  }
  return game.data;
}
const models = [
  { shooting, placement },
  { shooting: null, placement: null },
];
const first = complete(settings, models),
  second = complete(settings, models);
assert.deepEqual(
  first.history,
  second.history,
  "same configuration and uint64 seed reproduces browser history",
);
assert.equal(first.history[0].attacker, 1);
assert.equal(
  first.history.filter(
    (e) => e.attacker === first.result.winner && e.outcome !== "miss",
  ).length,
  17,
);
for (const seat of [0, 1]) {
  const shots = first.history
    .filter((e) => e.attacker === seat)
    .map((e) => e.coordinate.row * 10 + e.coordinate.column);
  assert.equal(new Set(shots).size, shots.length, "no repeated shots");
  assert.equal(
    new Set(
      first.fleets[seat].flatMap((s) =>
        s.cells.map((c) => c.row * 10 + c.column),
      ),
    ).size,
    17,
    "no overlap",
  );
}
for (const kinds of [
  ["random", "random"],
  ["hunt", "weights"],
  ["weights", "weights"],
]) {
  const d = complete(
    { ...settings, players: kinds.map((kind) => ({ kind })) },
    [
      { shooting, placement: null },
      { shooting, placement },
    ],
  );
  assert.equal(d.status, "finished");
}
assert.throws(
  () =>
    new BrowserMatch(
      { ...settings, seed: "18446744073709551616" },
      models,
      "bad",
    ),
);
assert.throws(() => validateModel(placement, "shooting"));
const empty = observation([], 0),
  visible = observation(
    [{ attacker: 0, coordinate: { row: 3, column: 3 }, outcome: "hit" }],
    0,
  );
assert.equal(empty.board.filter((x) => x === "unknown").length, 100);
const target = chooseShot("hunt", null, visible, () => 0);
assert.ok([23, 32, 34, 43].includes(target), "hunt follows public hits");
assert.equal(
  chooseShot("weights", shooting, empty, randomStream("42", "test")),
  chooseShot("weights", shooting, empty, randomStream("42", "test")),
  "policy receives only public observations",
);
// Independent direct calculation fixture: h1=tanh(0.5), h2=tanh(h1*2), output=h2*3-0.2.
const zero = Object.fromEntries(
  Object.entries(shooting.tensors).map(([key, a]) => [
    key,
    Array.isArray(a[0]) ? a.map((row) => row.map(() => 0)) : a.map(() => 0),
  ]),
);
zero.W1[0][0] = 0.5;
zero.W2[0][0] = 2;
zero.W3[0][0] = 3;
zero.b3[0] = -0.2;
assert.ok(
  Math.abs(
    logits([1, ...Array(404).fill(0)], { tensors: zero })[0] -
      (Math.tanh(Math.tanh(0.5) * 2) * 3 - 0.2),
  ) < 1e-14,
);
console.log(
  "Browser game: weights, bots, placement, uint64 seeds, public observations, legal shots, reproducibility, inference, and termination passed.",
);
