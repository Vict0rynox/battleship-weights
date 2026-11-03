import { BrowserMatch, validateModel } from "./browser-game.js";
const clone = (value) => JSON.parse(JSON.stringify(value));
const DATA = new URL("../data/", import.meta.url);
const cache = new Map();
async function json(path) {
  if (!cache.has(path))
    cache.set(
      path,
      fetch(new URL(path, DATA))
        .then(async (r) => {
          if (!r.ok) throw new Error("Published record was not found");
          return r.json();
        })
        .catch((error) => {
          cache.delete(path);
          throw error;
        }),
    );
  return clone(await cache.get(path));
}
const safe = (id) => {
  if (!/^[a-z0-9-]+$/.test(id || "")) throw new Error("Invalid record ID");
  return id;
};
const catalog = () => json("index.json");
let active = null,
  starting = false,
  timer = null,
  records = null;
const listeners = new Set();
const storageKey = `battleship-browser-1.0:${new URL("../", import.meta.url).pathname}`;
let storageWarning = "";
function saved() {
  if (records) return records;
  try {
    records = JSON.parse(localStorage.getItem(storageKey) || "[]");
    if (!Array.isArray(records)) records = [];
  } catch {
    records = [];
    storageWarning =
      "Browser storage is unavailable. Games will be kept only until this page closes.";
  }
  // No promise to resume an RNG state after the tab closes. Keep its replay instead.
  for (const r of records)
    if (["running", "paused"].includes(r.status)) {
      r.status = "interrupted";
      r.error = "Browser page closed before the game finished.";
      r.revision++;
    }
  return records;
}
function store() {
  const data = active?.data;
  if (data)
    records = [clone(data), ...saved().filter((r) => r.id !== data.id)].slice(
      0,
      10,
    );
  try {
    localStorage.setItem(storageKey, JSON.stringify(records || []));
  } catch {
    storageWarning =
      "Browser storage is full or unavailable. This game is kept in memory only; download its replay before closing.";
  }
}
function notify() {
  store();
  for (const listener of listeners) listener(clone(active.data));
}
function schedule() {
  clearTimeout(timer);
  if (!active || active.data.status !== "running") return;
  timer = setTimeout(
    () => {
      try {
        const count = active.data.instant ? 8 : 1;
        for (let i = 0; i < count && active.data.status === "running"; i++)
          active.advance();
        notify();
      } catch (error) {
        active.data.status = "error";
        active.data.error = error.message;
        active.data.revision++;
        notify();
      }
      schedule();
    },
    active.data.instant ? 0 : 1000 / active.data.speed,
  );
}
async function model(id, role) {
  const data = await catalog();
  if (!data.models.some((m) => m.id === id && m.role === role))
    throw new Error(`Choose valid ${role} weights`);
  return validateModel(await json(`models/${safe(id)}.json`), role);
}
function game(id) {
  const result =
    active?.data.id === id ? active.data : saved().find((r) => r.id === id);
  if (!result) throw new Error("Game not found in this browser");
  return { ...clone(result), storage_warning: storageWarning };
}
export function watchGame(id, update) {
  const listener = (data) => {
    if (data.id === id) update({ ...data, storage_warning: storageWarning });
  };
  listeners.add(listener);
  return { close: () => listeners.delete(listener) };
}
export async function staticApi(path, body) {
  const url = new URL(path, "https://static.invalid"),
    parts = url.pathname.split("/").filter(Boolean),
    query = url.searchParams;
  const data = await catalog();
  if (parts[0] === "models" && parts.length === 1) return data.models;
  if (parts[0] === "attempts") {
    if (body !== undefined)
      throw new Error(
        "Prompt preparation and response import use the local dashboard: make ui",
      );
    const result =
      parts.length === 1
        ? data.attempts
        : data.attempts.find((a) => a.id === parts[1]);
    if (!result) throw new Error("Published attempt not found");
    return result;
  }
  if (parts[0] === "runs") {
    if (parts.length === 1) return data.runs;
    const id = safe(parts[1]);
    const run = await json(`runs/${id}.json`);
    if (parts.length === 2) return run;
    if (parts[2] !== "trials") throw new Error("Published endpoint not found");
    if (parts[3]) return json(`replays/${id}/${safe(parts[3])}.json`);
    const rows = run.trials;
    const selected = rows.filter((r) => {
      const phase = query.get("phase") || "benchmark";
      if (run.kind !== "shooting" && phase !== "all" && r.phase !== phase)
        return false;
      for (const [key, alias] of [
        ["candidate", "policy"],
        ["opponent", "attacker"],
        ["family", "family"],
      ])
        if (query.get(key) && (r[key] ?? r[alias]) !== query.get(key))
          return false;
      if (
        query.get("outcome") &&
        r.candidate_won !== (query.get("outcome") === "win")
      )
        return false;
      return (
        !query.get("query") ||
        JSON.stringify(r)
          .toLowerCase()
          .includes(query.get("query").toLowerCase())
      );
    });
    const offset = Number(query.get("offset") || 0),
      limit = Number(query.get("limit") || 30);
    const facets = Object.fromEntries(
      [
        ["candidate", "policy"],
        ["opponent", "attacker"],
        ["family", "family"],
      ].map(([key, alias]) => [
        key,
        [
          ...new Set(
            rows.map((r) => String(r[key] ?? r[alias] ?? "")).filter(Boolean),
          ),
        ].sort(),
      ]),
    );
    return {
      summary_only: run.replay_sample.available === 0,
      items: selected.slice(offset, offset + limit),
      total: selected.length,
      facets,
      warnings: [
        run.replay_sample.available === 0
          ? "Summary-only public snapshot. Historical trials remain local; charts cover the complete original run."
          : `Public replay sample: ${run.replay_sample.available} of ${run.replay_sample.total} records. Charts and summary statistics use the full original run.`,
        ...run.warnings,
      ],
    };
  }
  if (parts[0] === "compare" && body) {
    const runs = await Promise.all(
      body.report_ids.map((id) => json(`runs/${safe(id)}.json`)),
    );
    const sorted = (value) =>
      value && typeof value === "object"
        ? Array.isArray(value)
          ? value.map(sorted)
          : Object.fromEntries(
              Object.keys(value)
                .sort()
                .map((k) => [k, sorted(value[k])]),
            )
        : value;
    const signature = (r) =>
      JSON.stringify(
        sorted({
          kind: r.kind,
          configuration: r.configuration,
          versions: r.manifest.versions,
          schedule: r.manifest.schedule_version,
          placement: r.kind !== "placement" ? r.hashes.placement : null,
          attacker: r.kind === "placement" ? r.hashes.shooting : null,
        }),
      );
    const compatible = runs.every((r) => signature(r) === signature(runs[0]));
    return {
      compatible,
      message: compatible
        ? "Configurations match. Compare metrics descriptively; no cross-run significance is inferred."
        : "Incompatible configurations. No improvement score is calculated.",
      runs,
    };
  }
  if (parts[0] === "games") {
    saved();
    if (parts.length === 1 && !body)
      return saved().map((r) => {
        const { history, fleets, ...meta } = r;
        return meta;
      });
    if (parts.length === 1 && body) {
      const previous = saved().find((r) => r.request_id === body.request_id);
      if (previous) return game(previous.id);
      if (
        starting ||
        (active && ["running", "paused"].includes(active.data.status))
      ) {
        const e = new Error("A browser game is already active");
        e.game_id = active?.data.id;
        throw e;
      }
      starting = true;
      try {
        const models = await Promise.all(
          body.players.map(async (p) => ({
            shooting:
              p.kind === "weights" ? await model(p.model_id, "shooting") : null,
            placement: p.placement_model_id
              ? await model(p.placement_model_id, "placement")
              : null,
          })),
        );
        active = new BrowserMatch(
          body,
          models,
          crypto.randomUUID().replaceAll("-", ""),
        );
        notify();
        schedule();
        return game(active.data.id);
      } finally {
        starting = false;
      }
    }
    const id = safe(parts[1]);
    if (parts.length === 2) return game(id);
    if (parts[2] === "control" && body) {
      if (
        active?.data.id !== id ||
        !["running", "paused"].includes(active.data.status)
      )
        throw new Error("Game is no longer active");
      const d = active.data;
      if (body.action === "pause") d.status = "paused";
      else if (body.action === "play") d.status = "running";
      else if (body.action === "step") {
        d.status = "paused";
        active.advance();
      } else if (body.action === "speed") {
        if (![1, 2, 4, 8, 16].includes(body.speed))
          throw new Error("Unsupported speed");
        d.speed = body.speed;
      } else if (body.action === "finish") {
        d.instant = true;
        d.status = "running";
      } else throw new Error("Unknown game control");
      d.revision++;
      notify();
      schedule();
      return game(id);
    }
  }
  throw new Error("This operation requires the local dashboard: make ui");
}
