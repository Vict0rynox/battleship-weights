// Run with: node tests/test_ui_replay.mjs (no packages required).
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
const source = readFileSync(
  new URL(
    "../packages/server/src/battleship_server/static/replay.js",
    import.meta.url,
  ),
  "utf8",
);
const { boardState, playerStats, fleetCells } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`
);
const events = [
  { number: 1, attacker: 0, coordinate: { row: 0, column: 0 }, outcome: "hit" },
  {
    number: 2,
    attacker: 1,
    coordinate: { row: 1, column: 1 },
    outcome: "miss",
  },
  {
    number: 3,
    attacker: 0,
    coordinate: { row: 0, column: 1 },
    outcome: "sunk",
    ship: "destroyer",
    sunk_cells: [
      { row: 0, column: 0 },
      { row: 0, column: 1 },
    ],
  },
];
assert.deepEqual(boardState(events, 0, 1), {});
assert.equal(boardState(events, 1, 1)["0,0"].state, "hit");
assert.equal(
  boardState(events, 1, 1)["0,1"],
  undefined,
  "future shot stays unknown",
);
assert.equal(boardState(events, 3, 1)["0,0"].state, "sunk");
assert.equal(boardState(events, 3, 1)["0,1"].latest, true);
assert.equal(
  boardState(events, 1, 1)["0,0"].state,
  "hit",
  "rewinding removes future sink disclosure",
);
assert.equal(boardState(events, 2, 0)["1,1"].state, "miss");
assert.equal(playerStats(events.slice(0, 1), 0).sunk, 0);
assert.deepEqual(playerStats(events, 0), {
  shots: 2,
  hits: 2,
  accuracy: 1,
  sunk: 1,
});
assert.equal(
  fleetCells([
    {
      name: "destroyer",
      start: { row: 0, column: 0 },
      orientation: "horizontal",
    },
  ]).size,
  2,
);
assert.equal(fleetCells(null).size, 0);
console.log(
  "Replay prefix, sink disclosure, rewind, seat mapping, fleet conversion, and counters passed.",
);
