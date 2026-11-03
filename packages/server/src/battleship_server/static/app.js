import { boardState, fleetCells, playerStats } from "./replay.js";
const app = document.querySelector("#app");
const STATIC =
  document.querySelector('meta[name="battleship-mode"]')?.content === "static";
const staticBackend = STATIC ? import("./static-api.js") : null;
const state = {
  models: [],
  runs: [],
  attempts: [],
  games: [],
  role: "shooting",
  selected: new Set(),
  viewer: null,
  source: null,
  timer: null,
  route: 0,
};
const fmt = (v, d = 1) =>
  v == null ? "—" : typeof v === "number" ? v.toFixed(d) : String(v);
const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const date = (v) => (v ? new Date(v).toLocaleString() : "Not recorded");
const short = (v) => (v ? `${v.slice(0, 12)}…` : "Unknown");
const pill = (text) => h("span", { class: `pill ${text}` }, text);
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k.startsWith("on"))
      el.addEventListener(k.slice(2).toLowerCase(), (event) => {
        Promise.resolve()
          .then(() => v(event))
          .catch(showError);
      });
    else if (k === "class") el.className = v;
    else if (k === "checked") el.checked = v;
    else if (k === "value") el.value = v;
    else if (k === "disabled") el.disabled = v;
    else el.setAttribute(k, v);
  }
  for (const child of children.flat(Infinity)) {
    if (child != null && child !== false)
      el.append(
        child instanceof Node ? child : document.createTextNode(String(child)),
      );
  }
  return el;
}
function replace(target, ...children) {
  target.replaceChildren(
    ...children
      .flat(Infinity)
      .filter((child) => child != null && child !== false),
  );
}
function showError(error) {
  const el = document.querySelector("#error");
  el.replaceChildren(document.createTextNode(error.message || String(error)));
  if (error.game_id)
    el.append(
      h(
        "a",
        { class: "inline-link", href: `#game/${error.game_id}` },
        " Open active game",
      ),
    );
  el.hidden = false;
  el.onclick = () => (el.hidden = true);
  console.error(error);
}
function toast(text) {
  const el = document.querySelector("#toast");
  el.textContent = text;
  el.hidden = false;
  setTimeout(() => (el.hidden = true), 3500);
}
async function api(path, body) {
  if (STATIC) return (await staticBackend).staticApi(path, body);
  const response = await fetch(
    `/ui-api${path}`,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(
      data.error ||
        (Array.isArray(data.detail)
          ? data.detail.map((e) => `${e.loc.join(".")}: ${e.msg}`).join("; ")
          : data.detail) ||
        "Request failed",
    );
    Object.assign(error, data);
    throw error;
  }
  return data;
}
async function refresh() {
  [state.models, state.runs, state.attempts, state.games] = await Promise.all(
    ["/models", "/runs", "/attempts", "/games"].map((p) => api(p)),
  );
}
const heading = (k, title, description, action) =>
  h(
    "div",
    { class: "page-heading" },
    h(
      "div",
      {},
      h("p", { class: "eyebrow" }, k),
      h("h1", {}, title),
      h("p", {}, description),
    ),
    action,
  );
const panel = (title, ...children) =>
  h(
    "section",
    { class: "panel" },
    title ? h("h2", {}, title) : null,
    ...children,
  );
const button = (label, fn, cls = "") =>
  h(
    "button",
    {
      type: "button",
      class: cls,
      "data-focus-key": ["Ⅱ Pause", "▶ Play"].includes(label)
        ? "playback-toggle"
        : label,
      "aria-label":
        {
          "↤": "Restart replay",
          "←": "Previous shot",
          "→": "Next shot",
          "↦": "Jump to end",
        }[label] || label,
      title:
        {
          "↤": "Restart replay",
          "←": "Previous shot",
          "→": "Next shot",
          "↦": "Jump to end",
        }[label] || label,
      onclick: fn,
    },
    label,
  );
const field = (label, input) => {
  if (!input.hasAttribute("aria-label"))
    input.setAttribute("aria-label", label);
  return h("label", {}, h("span", {}, label), input);
};
function select(options, value, attrs = {}) {
  const el = h(
    "select",
    attrs,
    ...options.map((o) =>
      h(
        "option",
        { value: typeof o === "string" ? o : o[0] },
        typeof o === "string" ? o : o[1],
      ),
    ),
  );
  el.value = String(value ?? "");
  return el;
}
const stat = (label, value, note) =>
  h(
    "div",
    { class: "stat-card" },
    h("span", { class: "caption" }, label),
    h("span", { class: "value" }, value),
    h("span", { class: "hint" }, note),
  );
function table(headers, rows) {
  return h(
    "div",
    { class: "table-scroll" },
    h(
      "table",
      {},
      h("thead", {}, h("tr", {}, ...headers.map((c) => h("th", {}, c)))),
      h("tbody", {}, ...rows),
    ),
  );
}
const cell = (...v) => h("td", {}, ...v);
const details = (label, value) =>
  h(
    "details",
    { class: "details" },
    h("summary", {}, label),
    h(
      "pre",
      {},
      typeof value === "string" ? value : JSON.stringify(value, null, 2),
    ),
  );
function empty(text) {
  return h("div", { class: "empty" }, text);
}
function modelOptions(role) {
  return state.models
    .filter((m) => m.role === role && !m.error)
    .map((m) => [m.id, m.label]);
}
function metric(run) {
  const s = run.summary;
  if (run.kind === "match") {
    const row = s.matchups?.model_vs_hunt || Object.values(s.matchups || {})[0];
    return row ? `${pct(row.win_rate)} win rate` : "No measured games";
  }
  if (run.kind === "shooting")
    return `${fmt(s.policies?.model?.overall?.mean_shots)} shots to clear`;
  const row = s.matchups?.model_vs_hunt || Object.values(s.matchups || {})[0];
  return `${fmt(row?.mean_shots)} shots survived`;
}
function runCard(run, check = false) {
  return h(
    "div",
    { class: "run-card" },
    h("div", { class: "headline" }, metric(run)),
    h(
      "h3",
      {},
      check
        ? h("input", {
            type: "checkbox",
            "aria-label": `Compare ${run.label}`,
            checked: state.selected.has(run.id),
            onchange: (e) => {
              if (e.target.checked) state.selected.add(run.id);
              else state.selected.delete(run.id);
            },
          })
        : null,
      h("a", { class: "inline-link", href: `#results/${run.id}` }, run.label),
    ),
    h(
      "div",
      { class: "run-meta" },
      pill(run.kind),
      pill(run.status),
      h("span", {}, date(run.created_at)),
    ),
    h(
      "div",
      { class: "hint" },
      `Shooting ${short(run.hashes.shooting)} · Placement ${short(run.hashes.placement)}`,
    ),
  );
}
function comparisonSettings(run) {
  const c = run.configuration;
  return h(
    "div",
    { class: "comparison-settings" },
    `Seed ${c.seed ?? "—"} · Opponents / attackers: ${(c.opponents || c.attackers || c.policies || []).join(", ") || "—"} · Layouts: ${(c.placements || []).join(", ") || c.placement_distribution || "—"} · Pairs / boards / trials: ${c.pairs ?? c.boards ?? c.trials ?? "—"} · Placement artifact: ${short(run.hashes.placement)}`,
  );
}
function experiments() {
  const role = state.role;
  const items = state.attempts.filter((a) => a.role === role);
  const evaluated = state.attempts.filter(
    (a) => a.state === "evaluated",
  ).length;
  replace(
    app,
    heading(
      "WEIGHTS → EVALUATION → INSIGHT",
      "Experiments",
      "Explore how a single model response becomes a fixed Battleship policy.",
      h(
        "a",
        { href: "#prepare", class: "button primary" },
        STATIC ? "Local generation workflow" : "＋ Prepare attempt",
      ),
    ),
    h(
      "div",
      { class: "grid-3 stats" },
      stat(
        "Weight artifacts",
        state.models.filter((m) => !m.error).length,
        "Independent shooting and placement networks",
      ),
      stat("Evaluated attempts", evaluated, "Linked by parameter content hash"),
      stat(
        "Evaluation runs",
        state.runs.length,
        "Matches, shooting, and fleet defense",
      ),
    ),
    h(
      "section",
      { class: "intro" },
      h("h2", {}, "Can a model design a policy in one response?"),
      h(
        "p",
        {},
        "The generation model chooses numerical weights directly. No training dataset, gradients, tools, or code execution are used during generation. After import, a fixed network plays the game; no language model is called for individual moves.",
      ),
      h(
        "div",
        { class: "grid-2" },
        h(
          "div",
          { class: "method" },
          h("span", { class: "method-number" }, "01"),
          h("h3", {}, "One-shot generation"),
          h(
            "p",
            {},
            "A self-contained game and network contract. All tensors start at zero. One response supplies the design and complete weights.",
          ),
        ),
        h(
          "div",
          { class: "method" },
          h("span", { class: "method-number" }, "02"),
          h("h3", {}, "Fresh-context improvement"),
          h(
            "p",
            {},
            "A new chat receives the contract, previous weights, selected results, and an objective. No earlier conversation is included.",
          ),
        ),
      ),
    ),
    h(
      "div",
      { class: "panel-heading" },
      h("h2", {}, "Attempt history"),
      h(
        "span",
        { class: "hint" },
        "External chat conditions are user-recorded",
      ),
    ),
    h(
      "div",
      { class: "tabs" },
      ...["shooting", "placement"].map((r) =>
        button(
          r === "shooting" ? "Shooting policy" : "Fleet placement",
          () => {
            state.role = r;
            experiments();
          },
          r === role ? "selected" : "",
        ),
      ),
    ),
    h(
      "section",
      { class: "panel" },
      h(
        "p",
        { class: "hint" },
        role === "shooting"
          ? "Shooting: choose the next legal shot from public target knowledge."
          : "Placement: construct a legal fleet that takes opponents more shots to destroy.",
      ),
      items.length
        ? table(
            [
              "Attempt / artifact",
              "Method",
              "Generation model",
              "Results",
              "State",
              "",
            ],
            items.map((a) =>
              h(
                "tr",
                { class: "row-link" },
                cell(
                  h(
                    "a",
                    { href: `#experiments/${a.id}` },
                    h("strong", {}, a.label),
                  ),
                  h("small", {}, short(a.hash)),
                ),
                cell(
                  a.type === "unknown" ? "Not recorded" : a.type,
                  h(
                    "small",
                    {},
                    a.parent
                      ? `Parent ${short(a.parent_hash)}`
                      : "Parent not recorded",
                  ),
                ),
                cell(a.generation_model || "Unknown"),
                cell(`${a.evaluations.length} linked runs`),
                cell(pill(a.state)),
                cell(
                  h(
                    "a",
                    {
                      href: `#experiments/${a.id}`,
                      class: "button small quiet",
                    },
                    "Inspect →",
                  ),
                ),
              ),
            ),
          )
        : empty(
            "No attempts in this role yet. Prepare a self-contained prompt to begin.",
          ),
    ),
    h(
      "p",
      { class: "hint" },
      "Historical artifacts retain unknown provenance until explicitly recorded. Filenames do not establish generation order or parentage.",
    ),
  );
}
async function attemptDetail(id) {
  const token = state.route;
  const a = await api(`/attempts/${id}`);
  if (token !== state.route) return;
  replace(
    app,
    heading(
      "EXPERIMENT / ATTEMPT",
      a.label,
      `${a.role} · ${a.type} · ${a.generation_model || "Generation model unknown"}`,
      h("a", { href: "#prepare", class: "button" }, "Prepare another"),
    ),
    panel(
      "Artifact and provenance",
      h("div", { class: "toolbar" }, pill(a.state), pill(a.role)),
      h(
        "p",
        {},
        `Parent: ${a.parent || "Not recorded"} · Created: ${date(a.created_at)}`,
      ),
      h(
        "p",
        {},
        "Content hash: ",
        h("code", {}, a.hash || "Awaiting response"),
      ),
      h("p", { class: "hint" }, a.notes || "No notes recorded."),
      h(
        "p",
        { class: "hint" },
        "Fresh-context isolation and no-tool conditions are user-recorded, not verified by this app.",
      ),
      ...(a.commands || []).map((command) =>
        h(
          "div",
          { class: "toolbar" },
          h("code", {}, command),
          button(
            "Copy command",
            () => navigator.clipboard.writeText(command),
            "small",
          ),
        ),
      ),
    ),
    a.snapshot
      ? panel(
          "Prepared prompt",
          h(
            "p",
            { class: "hint" },
            "Immutable package. Copy the complete prompt into a new chat.",
          ),
          h(
            "div",
            { class: "toolbar" },
            button("Copy complete prompt", async () => {
              await navigator.clipboard.writeText(a.snapshot.prompt);
              toast("Complete prompt copied");
            }),
            h(
              "a",
              {
                href: `/ui-api/attempts/${id}/prompt`,
                class: "button",
                download: "",
              },
              "Download prompt",
            ),
          ),
          h("textarea", {
            class: "prompt",
            readonly: "",
            value: a.snapshot.prompt,
            "aria-label": "Complete prepared prompt",
          }),
        )
      : panel(
          "Generation history",
          h(
            "p",
            { class: "hint" },
            STATIC
              ? "Raw prompts, responses, and private notes are omitted from the public snapshot. Use the local dashboard for generation workflows."
              : "Original prompt and response were not recorded for this existing artifact.",
          ),
        ),
    a.snapshot && !a.recorded_response ? responseForm(a) : null,
    a.recorded_response
      ? panel(
          "Recorded response",
          details("View original model response", a.recorded_response.response),
        )
      : null,
    panel(
      "Linked evaluations",
      a.evaluations.length
        ? a.evaluations.map((r) => runCard(r))
        : empty(
            "Run the copied CLI command, then refresh this page to link results automatically.",
          ),
    ),
  );
}
function responseForm(a) {
  const response = h("textarea", {
    rows: 10,
    placeholder: "Paste the full model response or edits text.",
    "aria-label": "Model response",
    required: "",
  });
  const model = h("input", {
    value: a.generation_model || "",
    placeholder: "Optional model name",
  });
  const notes = h("textarea", { rows: 3, value: a.notes || "" });
  const file = h("input", {
    type: "file",
    accept: ".txt,.md,.dat,.edits",
    onchange: async (e) => {
      const f = e.target.files[0];
      if (f) {
        if (f.size > 8 * 1024 * 1024) throw new Error("Response exceeds 8 MiB");
        response.value = await f.text();
      }
    },
  });
  const submit = h(
    "button",
    { type: "submit", class: "primary" },
    "Validate and import response",
  );
  return panel(
    "Record a response",
    h(
      "form",
      {
        onsubmit: async (e) => {
          e.preventDefault();
          submit.disabled = true;
          try {
            await api(`/attempts/${a.id}/response`, {
              response: response.value,
              generation_model: model.value,
              notes: notes.value,
            });
            await refresh();
            await attemptDetail(a.id);
            toast("Response preserved and weights imported");
          } finally {
            submit.disabled = false;
          }
        },
      },
      field("Paste response", response),
      field("Or load a response file", file),
      h(
        "div",
        { class: "grid-2" },
        field("Generation model", model),
        field("Notes", notes),
      ),
      h(
        "p",
        { class: "hint" },
        "The final edits block is applied to zero tensors. A response cannot overwrite an earlier artifact.",
      ),
      submit,
    ),
  );
}
function prepareForm() {
  if (STATIC) {
    replace(
      app,
      heading(
        "GENERATION WORKFLOW",
        "Continue in the local dashboard",
        "The public site publishes experiment evidence and runs browser demonstrations.",
      ),
      panel(
        "Prepare and record new attempts",
        h(
          "p",
          {},
          "Run the project locally to prepare complete one-shot or fresh-context improvement prompts, import responses, and launch Python evaluations.",
        ),
        h("pre", {}, "make ui"),
        h(
          "p",
          { class: "hint" },
          "Open http://127.0.0.1:8002. Public hosting never writes to the project or receives model responses.",
        ),
        h(
          "a",
          { href: "#experiments", class: "button" },
          "Browse published experiments",
        ),
      ),
    );
    return;
  }
  const role = select(["shooting", "placement"], "shooting");
  const kind = select(
    [
      ["one-shot", "One-shot generation"],
      ["improvement", "Fresh-context improvement"],
    ],
    "one-shot",
  );
  const label = h("input", {
    required: "",
    placeholder: "e.g. Shooting — improved target following",
    maxlength: 200,
  });
  const gen = h("input", { placeholder: "Optional; recorded when known" });
  const objective = h("textarea", {
    placeholder: "State the objective. Required for an improvement attempt.",
  });
  const notes = h("textarea", {
    rows: 3,
    placeholder: "Optional experiment notes",
  });
  const parent = select(
    [["", "Select parent artifact"], ...modelOptions("shooting")],
    "",
  );
  const feedback = h("div", { class: "feedback-list" });
  const improvement = h(
    "div",
    { hidden: true },
    field("Previous artifact", parent),
    h("div", {}, h("p", { class: "hint" }, "Evaluation feedback"), feedback),
  );
  const selected = new Set();
  function updateFeedback() {
    selected.clear();
    const m = state.models.find((m) => m.id === parent.value);
    const runs = state.runs.filter((r) => m && r.hashes[role.value] === m.hash);
    feedback.replaceChildren(
      ...(runs.length
        ? runs.map((r) =>
            h(
              "label",
              { class: "check" },
              h("input", {
                type: "checkbox",
                value: r.id,
                onchange: (e) =>
                  e.target.checked ? selected.add(r.id) : selected.delete(r.id),
              }),
              `${r.label} · ${metric(r)} · ${r.status}`,
            ),
          )
        : [
            h(
              "span",
              { class: "hint" },
              "Select a parent with linked evaluations. Run an evaluation first if none are listed.",
            ),
          ]),
    );
  }
  parent.addEventListener("change", updateFeedback);
  role.addEventListener("change", () => {
    parent.replaceChildren(
      ...[["", "Select parent artifact"], ...modelOptions(role.value)].map(
        ([v, l]) => h("option", { value: v }, l),
      ),
    );
    updateFeedback();
  });
  kind.addEventListener("change", () => {
    improvement.hidden = kind.value !== "improvement";
    objective.required = kind.value === "improvement";
    updateFeedback();
  });
  const submit = h(
    "button",
    { type: "submit", class: "primary" },
    "Prepare and save complete prompt",
  );
  replace(
    app,
    heading(
      "EXPERIMENT / NEW ATTEMPT",
      "Prepare an attempt",
      "Create a self-contained prompt for a separate, fresh chat.",
    ),
    panel(
      "",
      h(
        "form",
        {
          onsubmit: async (e) => {
            e.preventDefault();
            submit.disabled = true;
            try {
              const a = await api("/attempts", {
                label: label.value,
                role: role.value,
                type: kind.value,
                parent_model_id: parent.value || null,
                report_ids: [...selected],
                objective: objective.value,
                generation_model: gen.value,
                notes: notes.value,
              });
              await refresh();
              location.hash = `experiments/${a.id}`;
            } finally {
              submit.disabled = false;
            }
          },
        },
        h(
          "div",
          { class: "grid-2" },
          field("Attempt label", label),
          field("Generation model", gen),
          field("Network role", role),
          field("Attempt type", kind),
        ),
        improvement,
        field("Objective", objective),
        field("Notes", notes),
        h(
          "div",
          { class: "note" },
          "One response. No tools, training, code execution, or feedback rounds. Improvements include the full prior weights and selected results and request a complete replacement from zero.",
        ),
        submit,
      ),
    ),
  );
}
async function compareSelected() {
  if (state.selected.size < 2)
    throw new Error("Select at least two evaluation runs");
  const result = await api("/compare", { report_ids: [...state.selected] });
  const area = document.querySelector("#comparison");
  area.replaceChildren(
    panel(
      "Run comparison",
      h(
        "div",
        { class: `note ${result.compatible ? "" : "warning"}` },
        result.message,
      ),
      ...result.runs.map((r) =>
        h(
          "div",
          {},
          h("h3", {}, r.label),
          h("p", {}, metric(r)),
          comparisonSettings(r),
          details("Configuration", r.configuration),
        ),
      ),
    ),
  );
  area.scrollIntoView({ behavior: "smooth" });
}
function results() {
  const list = h("div", { class: "report-list" });
  const query = h("input", {
    type: "search",
    placeholder: "Search reports…",
    "aria-label": "Search reports",
  });
  const kind = select(
    [
      ["all", "All evaluation types"],
      ["match", "Head-to-head"],
      ["shooting", "Shooting"],
      ["placement", "Placement"],
    ],
    "all",
    { "aria-label": "Evaluation type" },
  );
  function update() {
    const runs = state.runs.filter(
      (r) =>
        (kind.value === "all" || r.kind === kind.value) &&
        `${r.label} ${r.hashes.shooting} ${r.hashes.placement}`
          .toLowerCase()
          .includes(query.value.toLowerCase()),
    );
    list.replaceChildren(...runs.map((r) => runCard(r, true)));
    if (!runs.length)
      list.append(
        empty(
          "No matching evaluation reports. Existing CLI commands create reports automatically.",
        ),
      );
  }
  query.addEventListener("input", update);
  kind.addEventListener("change", update);
  update();
  replace(
    app,
    heading(
      "MEASURE / COMPARE",
      "Results",
      "Explore benchmark evidence, then step inside any recorded game.",
      button("↻ Refresh", async () => {
        await refresh();
        results();
      }),
    ),
    h(
      "div",
      { class: "toolbar" },
      query,
      kind,
      h("span", { class: "spacer" }),
      button("Compare selected", compareSelected),
    ),
    list,
    h("div", { id: "comparison" }),
    panel(
      STATIC ? "Games in this browser" : "Games started in this workspace",
      state.games.length
        ? state.games.map((g) =>
            h(
              "a",
              { class: "run-card", href: `#game/${g.id}` },
              h("h3", {}, g.label),
              h(
                "div",
                { class: "run-meta" },
                pill(g.status),
                date(g.created_at),
                `Seed ${g.configuration.seed}`,
              ),
            ),
          )
        : empty("No UI games yet. Start one from New game."),
    ),
  );
}
const svgNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}, text) {
  const el = document.createElementNS(svgNS, tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  if (text != null) el.textContent = text;
  return el;
}
function barChart(rows, percent = false) {
  const height = Math.max(110, rows.length * 44 + 24),
    svg = svgEl("svg", {
      viewBox: `0 0 640 ${height}`,
      class: "chart",
      role: "img",
      "aria-label": percent
        ? "Win rates with 95% confidence intervals"
        : "Mean shots with 95% confidence intervals",
    });
  const max = percent
    ? 1
    : Math.max(1, ...rows.map((r) => r.ci?.[1] ?? r.value ?? 0)) * 1.12;
  rows.forEach((r, i) => {
    const y = i * 44 + 20;
    svg.append(svgEl("text", { x: 0, y }, r.label));
    if (r.value != null) {
      const x = 175,
        w = (r.value / max) * 410;
      svg.append(
        svgEl("rect", {
          x,
          y: y - 12,
          width: Math.max(1, w),
          height: 18,
          rx: 3,
          class: "bar",
        }),
      );
      if (r.ci) {
        const a = x + (r.ci[0] / max) * 410,
          b = x + (r.ci[1] / max) * 410;
        svg.append(
          svgEl("line", {
            x1: a,
            x2: b,
            y1: y - 3,
            y2: y - 3,
            class: "interval",
          }),
        );
        for (const p of [a, b])
          svg.append(
            svgEl("line", {
              x1: p,
              x2: p,
              y1: y - 8,
              y2: y + 2,
              class: "interval",
            }),
          );
      }
      svg.append(
        svgEl(
          "text",
          {
            x: Math.min(
              607,
              x + Math.max(w, ((r.ci?.[1] ?? r.value) / max) * 410) + 8,
            ),
            y,
          },
          percent ? pct(r.value) : fmt(r.value),
        ),
      );
    }
  });
  return h("div", { class: "chart-scroll" }, svg);
}
function histogram(name, values) {
  const svg = svgEl("svg", {
    viewBox: "0 0 400 155",
    class: "chart",
    role: "img",
    "aria-label": `${name} shots distribution`,
  });
  const bins = new Array(20).fill(0);
  for (const [shots, count] of values)
    bins[Math.min(19, Math.floor(shots / 10))] += count;
  const maximum = Math.max(1, ...bins);
  bins.forEach((n, i) =>
    svg.append(
      svgEl("rect", {
        x: 25 + i * 18,
        y: 120 - (n / maximum) * 90,
        width: 14,
        height: (n / maximum) * 90,
        rx: 2,
        class: "bar",
      }),
    ),
  );
  svg.append(
    svgEl("line", { x1: 25, x2: 385, y1: 121, y2: 121, class: "axis" }),
  );
  for (const v of [0, 50, 100, 150, 200])
    svg.append(svgEl("text", { x: 25 + (v / 200) * 360, y: 142 }, v));
  return h("div", {}, h("p", { class: "chart-title" }, name), svg);
}
async function runDetail(id) {
  const token = state.route;
  const run = await api(`/runs/${id}`);
  if (token !== state.route) return;
  const s = run.summary;
  const rows =
    run.kind === "shooting"
      ? Object.entries(s.policies || {}).map(([name, v]) => ({
          label: name,
          value: v.overall.mean_shots,
          ci: v.overall.mean_shots_ci95,
        }))
      : Object.entries(s.matchups || {}).map(([name, v]) => ({
          label: name.replaceAll("_", " "),
          value: run.kind === "match" ? v.win_rate : v.mean_shots,
          ci: run.kind === "match" ? v.ci95 : v.mean_ci95,
        }));
  const trialArea = h("div");
  const chartTitle =
    run.kind === "match"
      ? "Win rates · 95% confidence intervals"
      : run.kind === "shooting"
        ? "Shots to clear · lower is better"
        : "Shots survived · higher is better";
  replace(
    app,
    heading(
      `RESULTS / ${run.kind.toUpperCase()}`,
      run.label,
      `${date(run.created_at)} · ${metric(run)}`,
    ),
    panel(
      "Evaluation overview",
      h("div", { class: "toolbar" }, pill(run.kind), pill(run.status)),
      comparisonSettings(run),
      run.replay_sample
        ? h(
            "p",
            { class: "note" },
            run.replay_sample.available === 0
              ? "Summary-only public archive. Metrics and distributions cover the full run; historical trial histories stay local."
              : `Public archive: ${run.replay_sample.available} sampled replays from ${run.replay_sample.total} records. Statistics and distributions cover the full run.`,
          )
        : null,
      s.failure
        ? h(
            "div",
            { class: "note warning" },
            `Incomplete run: ${JSON.stringify(s.failure)}`,
          )
        : null,
      h("h3", {}, chartTitle),
      barChart(rows, run.kind === "match"),
      h(
        "p",
        { class: "hint" },
        s.interval_method || "No summary statistics available yet.",
      ),
      run.kind === "match"
        ? table(
            ["Matchup", "Games", "First player", "Second player", "Hit rate"],
            Object.entries(s.matchups || {}).map(([k, v]) =>
              h(
                "tr",
                {},
                cell(k),
                cell(v.completed_games),
                cell(pct(v.win_rate_by_start?.first)),
                cell(pct(v.win_rate_by_start?.second)),
                cell(pct(v.hit_rate)),
              ),
            ),
          )
        : null,
      details("Full summary and paired comparisons", s),
      details("Manifest and configuration", run.manifest),
    ),
    panel(
      "Shot distributions",
      h(
        "div",
        { class: "grid-2" },
        ...Object.entries(run.distributions).map(([k, v]) => histogram(k, v)),
      ),
    ),
    panel("Recorded games / trials", trialArea),
  );
  await trialList(id, trialArea, run);
}
async function trialList(id, container, run) {
  let offset = 0;
  const phase = select(
    ["benchmark", "smoke", "http_smoke", "parity", "all"],
    "benchmark",
    { "aria-label": "Record phase" },
  );
  if (run.kind === "shooting") phase.disabled = true;
  const query = h("input", {
    type: "search",
    placeholder: "Search trial metadata…",
    "aria-label": "Search trials",
  });
  const candidate = select([["", "All policies"]], "", {
    "aria-label": "Policy",
  });
  const opponent = select([["", "All opponents"]], "", {
    "aria-label": "Opponent",
  });
  const family = select([["", "All layouts"]], "", {
    "aria-label": "Layout family",
  });
  const outcome = select(
    [
      ["", "All outcomes"],
      ["win", "Candidate wins"],
      ["loss", "Candidate losses"],
    ],
    "",
    { "aria-label": "Outcome" },
  );
  if (run.kind !== "match") outcome.disabled = true;
  const body = h("div"),
    pagination = h("div", { class: "toolbar" }),
    warnings = h("div");
  let serial = 0;
  async function load() {
    const current = ++serial;
    const params = new URLSearchParams({
      offset,
      limit: 30,
      phase: phase.value,
      query: query.value,
      candidate: candidate.value,
      opponent: opponent.value,
      family: family.value,
      outcome: outcome.value,
    });
    const data = await api(`/runs/${id}/trials?${params}`);
    if (current !== serial) return;
    for (const [key, el, label] of [
      ["candidate", candidate, "All policies"],
      ["opponent", opponent, "All opponents"],
      ["family", family, "All layouts"],
    ]) {
      const old = el.value;
      el.replaceChildren(
        ...[["", label], ...data.facets[key].map((v) => [v, v])].map(([v, l]) =>
          h("option", { value: v }, l),
        ),
      );
      el.value = old;
    }
    warnings.replaceChildren(
      ...data.warnings.map((w) => h("p", { class: "hint" }, w)),
    );
    body.replaceChildren(
      data.items.length
        ? table(
            [
              "Trial",
              "Policy / candidate",
              "Opponent / layout",
              "Phase",
              "Outcome / shots",
              "",
            ],
            data.items.map((r) =>
              h(
                "tr",
                {},
                cell(r.id),
                cell(r.candidate || r.policy),
                cell(r.opponent || r.attacker || r.family || "—"),
                cell(r.phase || "shooting"),
                cell(
                  run.kind === "match"
                    ? r.candidate_won
                      ? "Win"
                      : "Loss"
                    : `${r.shots_to_clear ?? r.attacker_shots} shots`,
                ),
                cell(
                  h(
                    "a",
                    { href: `#replay/${id}/${r.id}`, class: "button small" },
                    "▶ Replay",
                  ),
                ),
              ),
            ),
          )
        : empty(
            data.summary_only
              ? "Historical replays are kept out of this public snapshot. View the full-run charts above or start a new browser game."
              : "No records match these filters. Choose All phases to inspect smoke-only runs.",
          ),
    );
    pagination.replaceChildren(
      button("← Previous", async () => {
        offset = Math.max(0, offset - 30);
        await load();
      }),
      h(
        "span",
        { class: "hint" },
        `${data.total ? offset + 1 : 0}–${Math.min(offset + 30, data.total)} of ${data.total}`,
      ),
      button("Next →", async () => {
        offset += 30;
        await load();
      }),
    );
    pagination.firstChild.disabled = offset === 0;
    pagination.lastChild.disabled = offset + 30 >= data.total;
  }
  const search = button("Search", async () => {
    offset = 0;
    await load();
  });
  for (const el of [phase, candidate, opponent, family, outcome])
    el.addEventListener("change", () => {
      offset = 0;
      load().catch(showError);
    });
  query.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      offset = 0;
      load().catch(showError);
    }
  });
  container.replaceChildren(
    h(
      "div",
      { class: "toolbar" },
      phase,
      candidate,
      opponent,
      family,
      outcome,
      query,
      search,
    ),
    warnings,
    body,
    pagination,
  );
  await load();
}
function newGame() {
  const form = h("form");
  const selectors = [];
  const cards = [0, 1].map((seat) => {
    const kind = select(
      [
        ["weights", "Weight model"],
        ["hunt", "Hunt / target bot"],
        ["random", "Random bot"],
      ],
      seat === 0 && modelOptions("shooting").length ? "weights" : "hunt",
    );
    if (seat === 1) kind.value = "hunt";
    const model = select(
      [["", "Choose shooting weights"], ...modelOptions("shooting")],
      modelOptions("shooting")[0]?.[0] || "",
    );
    const placement = select(
      [["", "Automatic random placement"], ...modelOptions("placement")],
      "",
    );
    const wrapper = field("Shooting artifact", model);
    kind.addEventListener(
      "change",
      () => (wrapper.hidden = kind.value !== "weights"),
    );
    wrapper.hidden = kind.value !== "weights";
    selectors.push({ kind, model, placement });
    return panel(
      `Player ${seat + 1}`,
      field("Player type", kind),
      wrapper,
      field("Fleet placement", placement),
      h(
        "p",
        { class: "hint" },
        "Placement weights are independent of the shooting policy.",
      ),
    );
  });
  const seed = h("input", {
    value: "42",
    inputmode: "numeric",
    pattern: "[0-9]+",
    required: "",
  });
  const starting = select(
    [
      ["0", "Player 1"],
      ["1", "Player 2"],
    ],
    "0",
  );
  const speed = select(
    [1, 2, 4, 8, 16].map((v) => [v, `${v} shots / second`]),
    4,
  );
  const submit = h(
    "button",
    { type: "submit", class: "primary" },
    STATIC ? "▶ Start browser game" : "▶ Start live game",
  );
  form.append(
    h("div", { class: "grid-2" }, cards),
    panel(
      "Match settings",
      h(
        "div",
        { class: "grid-3" },
        field("Seed (uint64)", seed),
        field("Starts first", starting),
        field("Speed", speed),
      ),
      h(
        "div",
        { class: "note" },
        STATIC
          ? "Demonstration games run in this browser with the bundled weights. The most recent 10 replays are saved in browser storage. Closing the page stops play. Browser seeds use a separate RNG schedule from Python benchmarks."
          : "Games run on this local service and are saved automatically. Pausing stops play; closing the browser leaves the game running.",
      ),
      submit,
    ),
  );
  let requestId = crypto.randomUUID();
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    submit.disabled = true;
    api("/games", {
      request_id: requestId,
      seed: seed.value,
      starting_player: Number(starting.value),
      speed: Number(speed.value),
      players: selectors.map((s) => ({
        kind: s.kind.value,
        model_id: s.kind.value === "weights" ? s.model.value : null,
        placement_model_id: s.placement.value || null,
      })),
    })
      .then((g) => {
        location.hash = `game/${g.id}`;
      })
      .catch((error) => {
        showError(error);
        requestId = crypto.randomUUID();
      })
      .finally(() => (submit.disabled = false));
  });
  replace(
    app,
    heading(
      "PLAY / OBSERVE",
      "New game",
      "Put a fixed weight policy to the test, or watch two bots face off.",
    ),
    form,
  );
}
function viewerLibrary() {
  replace(
    app,
    heading(
      "OBSERVE / REPLAY",
      "Game viewer",
      "Open a saved game, return to a live match, or replay any evaluation trial.",
      h("a", { href: "#new", class: "button primary" }, "＋ New game"),
    ),
    panel(
      STATIC ? "Browser games" : "Workspace games",
      state.games.length
        ? state.games.map((g) =>
            h(
              "a",
              { class: "run-card", href: `#game/${g.id}` },
              h("h3", {}, g.label),
              h(
                "div",
                { class: "run-meta" },
                pill(g.status),
                date(g.created_at),
              ),
            ),
          )
        : empty("Start a match to see live play here."),
    ),
    panel(
      "Evaluation replays",
      h(
        "p",
        { class: "hint" },
        "Choose a report, filter its recorded trials, and select Replay.",
      ),
      ...state.runs.slice(0, 6).map((r) => runCard(r)),
    ),
  );
}
async function openViewer(data, live) {
  const v = {
    data,
    live,
    cursor: live ? data.history.length : 0,
    follow: live,
    playing: !live,
    speed: data.speed || 4,
    reveal: false,
  };
  state.viewer = v;
  renderViewer();
  if (live && ["running", "paused"].includes(data.status)) {
    const update = (incoming) => {
      if (state.viewer !== v || incoming.revision < v.data.revision) return;
      v.data = incoming;
      if (v.follow) v.cursor = v.data.history.length;
      renderViewer();
      if (!["running", "paused"].includes(v.data.status)) state.source?.close();
    };
    if (STATIC) {
      state.source = (await staticBackend).watchGame(data.id, update);
    } else {
      const source = new EventSource(
        `/ui-api/games/${data.id}/events?after=${data.revision}`,
      );
      state.source = source;
      source.onmessage = (event) => update(JSON.parse(event.data));
      source.onerror = () => {
        if (source.readyState !== EventSource.CLOSED) {
          const el = document.querySelector("#connection");
          if (el) el.textContent = "Reconnecting to live service…";
        }
      };
    }
  }
  let last = performance.now();
  state.timer = setInterval(() => {
    if (state.viewer !== v) return;
    const time = performance.now();
    if (!v.follow && v.playing && time - last >= 1000 / v.speed) {
      last = time;
      if (v.cursor < v.data.history.length) {
        v.cursor++;
        renderViewer();
      } else {
        v.playing = false;
        renderViewer();
      }
    }
  }, 30);
}
async function liveControl(action, speed) {
  const v = state.viewer;
  v.data = await api(`/games/${v.data.id}/control`, { action, speed });
  if (v.follow) v.cursor = v.data.history.length;
  renderViewer();
}
function seek(position) {
  const v = state.viewer;
  v.follow = false;
  v.playing = false;
  v.cursor = Math.max(0, Math.min(v.data.history.length, position));
  renderViewer();
}
function renderViewer() {
  const focus = document.activeElement;
  const focusKey = app.contains(focus)
    ? focus.getAttribute("data-focus-key") || focus.getAttribute("aria-label")
    : null;
  const v = state.viewer;
  if (!v) return;
  const d = v.data,
    history = d.history.slice(0, v.cursor),
    single = d.kind !== "match",
    active = ["running", "paused"].includes(d.status),
    atEnd = v.cursor === d.history.length;
  const boards = single ? [1] : [0, 1];
  const known = boards.some((seat) => d.fleets?.[seat]);
  const status =
    v.live && v.follow ? d.status : v.playing ? "replaying" : "paused";
  const controls = panel(
    "",
    h(
      "div",
      { class: "toolbar" },
      button("↤", () => seek(0)),
      button("←", () => seek(v.cursor - 1)),
      button(
        (v.follow ? d.status === "running" : v.playing) ? "Ⅱ Pause" : "▶ Play",
        async () => {
          if (v.live && v.follow && active)
            await liveControl(d.status === "running" ? "pause" : "play");
          else {
            v.playing = !v.playing;
            v.follow = false;
            if (v.playing && atEnd) v.cursor = 0;
            renderViewer();
          }
        },
        "primary",
      ),
      button("→", async () => {
        if (v.live && v.follow && active) await liveControl("step");
        else seek(v.cursor + 1);
      }),
      button("↦", () => seek(d.history.length)),
      select(
        [1, 2, 4, 8, 16].map((n) => [n, `${n} shots / sec`]),
        v.follow ? d.speed : v.speed,
        {
          "aria-label": "Playback speed",
          onchange: async (e) => {
            v.speed = Number(e.target.value);
            if (v.follow && active) await liveControl("speed", v.speed);
          },
        },
      ),
      h("span", { class: "spacer" }),
      v.live
        ? button("Return to live", () => {
            v.follow = true;
            v.playing = false;
            v.cursor = d.history.length;
            renderViewer();
          })
        : null,
      v.live && active
        ? button("Finish instantly", () => liveControl("finish"))
        : null,
      h("span", { class: "hint" }, `${v.cursor} / ${d.history.length} shots`),
    ),
    h("input", {
      class: "timeline",
      type: "range",
      min: 0,
      max: Math.max(1, d.history.length),
      value: v.cursor,
      "aria-label": "Replay shot position",
      oninput: (e) => {
        v.follow = false;
        v.playing = false;
        v.cursor = Number(e.target.value);
      },
      onchange: (e) => seek(Number(e.target.value)),
    }),
  );
  controls.classList.add("controls");
  const boardElements = boards.map((seat) => {
    const cells = boardState(d.history, v.cursor, seat);
    const fleet = fleetCells(d.fleets?.[seat]);
    const stats = playerStats(history, single ? 0 : seat);
    const grid = h(
      "div",
      {
        class: "board",
        role: "img",
        "aria-label": `${d.players[seat]} target board at shot ${v.cursor}`,
      },
      h("span", {}),
      ...Array.from({ length: 10 }, (_, i) =>
        h("span", { class: "axis-label" }, String.fromCharCode(65 + i)),
      ),
    );
    for (let row = 0; row < 10; row++) {
      grid.append(h("span", { class: "axis-label" }, row + 1));
      for (let col = 0; col < 10; col++) {
        const key = `${row},${col}`,
          c = cells[key] || {};
        grid.append(
          h("div", {
            class: `cell ${v.reveal && fleet.has(key) ? "fleet" : ""} ${c.state || ""} ${c.latest ? "latest" : ""}`,
            title: `${String.fromCharCode(65 + col)}${row + 1}: ${c.state || (v.reveal && fleet.has(key) ? "ship" : "unknown")}`,
            "aria-label": `${String.fromCharCode(65 + col)}${row + 1} ${c.state || "unknown"}`,
          }),
        );
      }
    }
    const nextAttacker =
      atEnd && d.result
        ? null
        : history.length
          ? 1 - history.at(-1).attacker
          : (d.configuration?.starting_player ?? d.history[0]?.attacker ?? 0);
    const card = h(
      "section",
      {
        class: `board-card ${nextAttacker === seat && !single ? "active" : ""}`,
      },
      h(
        "div",
        { class: "board-title" },
        h(
          "strong",
          {},
          single
            ? `${d.players[0]} → ${d.players[1]}`
            : `Player ${seat + 1} · ${d.players[seat]}`,
        ),
        pill(
          single
            ? "target"
            : nextAttacker === seat
              ? "next to fire"
              : "fleet board",
        ),
      ),
      grid,
      h(
        "div",
        { class: "board-stats" },
        ...[
          [stats.shots, "Shots fired"],
          [stats.hits, "Hits"],
          [pct(stats.accuracy), "Accuracy"],
          [stats.sunk, "Ships sunk"],
        ].map(([n, l]) => h("div", {}, h("strong", {}, n), l)),
      ),
      h(
        "p",
        { class: "hint" },
        d.fleets?.[seat]
          ? "Full fleet recorded · reveal optional"
          : "Full layout unavailable · shot history only",
      ),
    );
    return card;
  });
  const reveal = h("input", {
    type: "checkbox",
    "aria-label": "Reveal recorded fleets",
    checked: v.reveal,
    disabled: !known,
    onchange: (e) => {
      v.reveal = e.target.checked;
      renderViewer();
    },
  });
  const result =
    atEnd && d.result
      ? h(
          "div",
          { class: "note" },
          `Winner: ${d.players[d.result.winner]} · ${d.result.reason.replaceAll("_", " ")}`,
        )
      : null;
  const last = history.at(-1);
  const log = h(
    "div",
    { class: "move-log" },
    history.length
      ? history
          .slice(-80)
          .reverse()
          .map((e) =>
            h(
              "div",
              { class: `move-row ${e.outcome}` },
              h("span", {}, `#${e.number}`),
              h("span", {}, d.players[e.attacker]),
              h(
                "span",
                {},
                `${String.fromCharCode(65 + e.coordinate.column)}${e.coordinate.row + 1}`,
              ),
              h(
                "span",
                { class: "outcome" },
                e.outcome + (e.ship ? ` · ${e.ship}` : ""),
              ),
            ),
          )
      : empty("Play or step forward to see the opening shot."),
  );
  replace(
    app,
    heading(
      single ? "SINGLE-ATTACKER EVALUATION" : "MATCH / GAME VIEWER",
      d.label,
      single
        ? "Target practice: no competitive win rate or incoming-fire outcome is inferred."
        : "Watch the battle unfold, or inspect any point in the recorded history.",
    ),
    h(
      "div",
      { class: "toolbar" },
      pill(status),
      h("label", { class: "check" }, reveal, "Reveal recorded fleets"),
      h("span", { class: "spacer" }),
      h(
        "span",
        { id: "connection", class: "hint" },
        v.follow ? "Following live simulation" : "Historical playback",
      ),
    ),
    d.error ? h("div", { class: "note warning" }, d.error) : null,
    d.storage_warning
      ? h("div", { class: "note warning" }, d.storage_warning)
      : null,
    STATIC && v.live
      ? h(
          "div",
          { class: "note" },
          "Browser demonstration · browser-game-1.0 · not a Python benchmark result.",
          button(
            "Download replay",
            () => {
              const blob = new Blob([JSON.stringify(d, null, 2)], {
                type: "application/json",
              });
              const url = URL.createObjectURL(blob);
              const link = h("a", {
                href: url,
                download: `battleship-${d.id}.json`,
              });
              link.click();
              setTimeout(() => URL.revokeObjectURL(url), 1000);
            },
            "small",
          ),
        )
      : null,
    result,
    h("div", { class: "board-area" }, boardElements),
    h(
      "div",
      { class: "legend" },
      h("span", {}, h("i", {}), "Unknown"),
      h("span", {}, h("i", { class: "hit" }), "Hit ×"),
      h("span", {}, h("i", { class: "sunk" }), "Sunk ×"),
      h("span", {}, "Miss ·"),
      h("span", {}, "White outline: latest shot"),
    ),
    controls,
    h(
      "div",
      { class: "viewer-bottom" },
      panel("Move log · newest first", log),
      panel(
        "Game details",
        h("p", {}, `Shown: ${v.cursor} shots`),
        h(
          "p",
          { class: "hint" },
          last
            ? `Latest: ${d.players[last.attacker]} fired ${String.fromCharCode(65 + last.coordinate.column)}${last.coordinate.row + 1} · ${last.outcome}`
            : "Before the first shot.",
        ),
        v.live && d.configuration
          ? h(
              "div",
              { class: "toolbar" },
              button("Rematch", () => rematch(false)),
              button("New seed", () => rematch(true)),
            )
          : null,
        details("Configuration", d.configuration || {}),
      ),
    ),
  );
  if (focusKey) {
    const next = [
      ...app.querySelectorAll("[data-focus-key],[aria-label]"),
    ].find(
      (el) =>
        (el.getAttribute("data-focus-key") || el.getAttribute("aria-label")) ===
        focusKey,
    );
    next?.focus({ preventScroll: true });
  }
}
async function rematch(fresh) {
  const config = state.viewer.data.configuration;
  const seed = fresh
    ? String(crypto.getRandomValues(new Uint32Array(1))[0])
    : config.seed;
  const players = config.players.map((p) => ({
    kind: p.kind,
    model_id: p.model_id,
    placement_model_id: p.placement_model_id,
  }));
  const game = await api("/games", {
    request_id: crypto.randomUUID(),
    seed,
    players,
    speed: config.speed,
    starting_player: config.starting_player,
  });
  location.hash = `game/${game.id}`;
}
async function route() {
  window.scrollTo(0, 0);
  const serial = ++state.route;
  clearInterval(state.timer);
  state.source?.close();
  state.source = null;
  state.viewer = null;
  document.querySelector("#error").hidden = true;
  const [view = "experiments", id, tid] = (
    location.hash.slice(1) || "experiments"
  ).split("/");
  const navView = ["game", "replay"].includes(view)
    ? "viewer"
    : view === "prepare"
      ? "experiments"
      : view;
  document
    .querySelectorAll("nav a")
    .forEach((a) => a.classList.toggle("active", a.dataset.view === navView));
  document.querySelector("#breadcrumb").textContent =
    {
      experiments: "Experiments",
      results: "Results",
      viewer: "Game viewer",
      new: "New game",
    }[navView] || "Experiments";
  replace(app, empty("Loading workspace…"));
  try {
    await refresh();
    if (serial !== state.route) return;
    if (view === "experiments") id ? await attemptDetail(id) : experiments();
    else if (view === "prepare") prepareForm();
    else if (view === "results") id ? await runDetail(id) : results();
    else if (view === "new") newGame();
    else if (view === "viewer") viewerLibrary();
    else if (view === "game") {
      const data = await api(`/games/${id}`);
      if (serial === state.route) await openViewer(data, true);
    } else if (view === "replay") {
      const data = await api(`/runs/${id}/trials/${tid}`);
      if (serial === state.route) await openViewer(data, false);
    } else experiments();
  } catch (error) {
    replace(
      app,
      panel(
        "Unable to open this view",
        h("p", { class: "hint" }, error.message),
        h(
          "a",
          { href: "#experiments", class: "button" },
          "Back to experiments",
        ),
      ),
    );
    showError(error);
  }
}
if (STATIC) {
  const notice = h(
    "div",
    { class: "public-banner" },
    "PUBLIC SNAPSHOT · Benchmark summaries · Play new games in your browser",
  );
  document.querySelector(".topbar").after(notice);
}
window.addEventListener("hashchange", route);
window.addEventListener("keydown", (e) => {
  if (
    !state.viewer ||
    ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(
      document.activeElement.tagName,
    )
  )
    return;
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    seek(state.viewer.cursor - 1);
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    seek(state.viewer.cursor + 1);
  } else if (e.code === "Space") {
    e.preventDefault();
    const v = state.viewer;
    if (v.live && v.follow && ["running", "paused"].includes(v.data.status))
      liveControl(v.data.status === "running" ? "pause" : "play").catch(
        showError,
      );
    else {
      v.follow = false;
      v.playing = !v.playing;
      renderViewer();
    }
  }
});
route();
