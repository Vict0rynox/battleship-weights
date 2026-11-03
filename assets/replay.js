// All visible state comes from this prefix. Never apply future sink disclosures.
export function boardState(history, cursor, defender) {
  const cells = {};
  const prefix = history.slice(0, cursor);
  for (const event of prefix) {
    if (event.attacker === defender) continue;
    const key = `${event.coordinate.row},${event.coordinate.column}`;
    cells[key] = { state: event.outcome === "sunk" ? "hit" : event.outcome };
    if (event.outcome === "sunk")
      for (const c of event.sunk_cells || [])
        cells[`${c.row},${c.column}`] = { state: "sunk" };
  }
  const last = prefix.at(-1);
  if (last && last.attacker !== defender) {
    const key = `${last.coordinate.row},${last.coordinate.column}`;
    cells[key] = { ...cells[key], latest: true };
  }
  return cells;
}
export function fleetCells(fleet) {
  const result = new Set();
  for (const ship of fleet || []) {
    if (ship.cells)
      for (const c of ship.cells) result.add(`${c.row},${c.column}`);
    else {
      const length = {
        carrier: 5,
        battleship: 4,
        cruiser: 3,
        submarine: 3,
        destroyer: 2,
      }[ship.name];
      for (let i = 0; i < length; i++)
        result.add(
          `${ship.start.row + (ship.orientation === "vertical" ? i : 0)},${ship.start.column + (ship.orientation === "horizontal" ? i : 0)}`,
        );
    }
  }
  return result;
}
export function playerStats(history, attacker) {
  const shots = history.filter((e) => e.attacker === attacker);
  const hits = shots.filter((e) => e.outcome !== "miss").length;
  return {
    shots: shots.length,
    hits,
    accuracy: shots.length ? hits / shots.length : null,
    sunk: shots.filter((e) => e.outcome === "sunk").length,
  };
}
