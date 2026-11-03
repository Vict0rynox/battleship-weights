# GitHub Pages version

The public site runs entirely as static HTML, CSS, JavaScript, and JSON. It includes
experiment descriptions, artifact provenance, full-run summaries and charts,
summary-only benchmark data, and new games played in the browser using the bundled
shooting and placement weights. It has no API server, accounts, external model
calls, or external asset dependencies.

## Build and preview

A public snapshot of the current experiments is included under `site-data/`.
Building that snapshot requires only Python 3.10+ and its standard library:

```sh
make ui-static
make ui-static-preview
```

Open [the static preview](http://127.0.0.1:8003). The standalone output is `dist/`.
You can also run the build directly, including on a fresh checkout without `.venv`:

```sh
python3 scripts/export_ui.py build
python3 -m http.server 8003 --bind 127.0.0.1 --directory dist
```

Serve the directory over HTTP; opening `index.html` directly with a `file:` URL
cannot reliably load JavaScript modules and JSON. Assets and data use relative
URLs, so the same build works at a domain root or a GitHub project path such as
`https://USERNAME.github.io/BattleShipLLM/`. Navigation uses hash routes, including
links to individual reports and replays, without requiring server rewrites.

## Publish on GitHub

1. Create your GitHub repository and push this project, including `site-data/`,
   `scripts/export_ui.py`, browser source, and `.github/workflows/pages.yml`.
   Keep generated `dist/` ignored; the workflow builds it from the public snapshot.
2. In the repository, select **Settings → Pages → Build and deployment → Source →
   GitHub Actions**.
3. Open **Actions → Publish experiment dashboard → Run workflow** on your default
   branch. Later pushes to `main` or `master` rebuild when public data, browser
   source, exporter, or the workflow changes. If your default branch has another
   name, add it to the workflow's `push.branches` list; manual runs still work.
4. Use the URL shown by the deployment job. Configure custom domains in GitHub's
   Pages settings if needed.

The workflow builds only from versioned public data, uploads only `dist/`, and
uses a separate deployment job with the Pages/OIDC permissions. It does not
upload the entire repository. Deployment requires the repository and Pages
settings to exist; creating a local build does not publish anything.

See GitHub's [custom workflow instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [publishing source settings](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## Refresh the public experiments

After running evaluations or recording more imported artifacts locally:

```sh
make ui-snapshot
make ui-static
```

`ui-snapshot` needs the project's existing NumPy/client/server dependencies. It
explicitly reads your local reports and validated model files, then replaces the
generated public snapshot. Review and commit the resulting `site-data/` changes.
The GitHub build never reads `.battleship/` or regenerates experiments.

The exporter includes:

- Validated model tensors, labels, roles, and parameter content hashes.
- Imported/evaluated attempt metadata and links to reports by artifact hash.
- Full original summary statistics, confidence intervals, and shot distributions.
- No raw trial histories or historical replay examples by default. Browser games
  still work with the included model tensors.

The default `REPLAYS_PER_GROUP=0` exports summaries and aggregate distributions
only, keeping thousands of raw game records out of Git. This setting is preserved
on future `make ui-snapshot` runs. Historical trials remain in the ignored local
`.battleship/` directory.

If you intentionally want a replay sample later, explicitly opt in and review the
generated files before committing:

```sh
make ui-snapshot REPLAYS_PER_GROUP=1
```

Positive values select evenly spaced examples per phase, candidate, opponent,
policy, layout, starting seat, and outcome. The replay browser labels sampling
explicitly; chart statistics always cover the full original run.

The public snapshot deliberately omits player session files and credentials,
local UI games, draft attempts, raw generation prompts/responses, private notes,
absolute source paths, runtime environment values, source fingerprints, and
failure-context bodies. Report failures retain their type. Generated artifacts
under `weights/generated/` are included when valid; preparation-only drafts are
not. The original private files remain untouched.

The exporter uses marked staging/output directories. It refuses to overwrite an
unmarked directory, overwrite source/snapshot directories, or package symbolic
links from a public snapshot. JSON files are generated artifacts; edit the source
experiments and regenerate rather than editing them by hand.

## Browser games versus local games

| Capability | Public static site | Local dashboard |
| --- | --- | --- |
| Experiment overview and report comparisons | Published snapshot | All local data |
| Result charts and confidence intervals | Full exported run statistics | Full local run statistics |
| Historical replays | Omitted by default; explicit sample opt-in | Every recorded trial |
| Model/bot and model/model games | JavaScript browser simulation | Python authoritative engine |
| Placement models | Bundled, validated placement tensors | Local placement artifacts |
| Pause, step, speed, reveal, rematch | Yes | Yes |
| Save newly played games | Most recent 10 in this browser; replay download | Project files |
| Continue when page closes | No; retain interrupted replay | Yes, while service runs |
| Prepare prompts / import responses | Instructions for local workflow | Yes |
| Launch full benchmarks | Existing CLI | Existing CLI |

Browser games implement the same board rules, network shapes, input encoding,
tanh inference, masked temperature-1 sampling, and bot heuristics. They use an
independent `browser-game-1.0` pseudorandom schedule and JavaScript float64
arithmetic. The same browser seed/configuration repeats a browser game, but it
does not promise the same game as NumPy/Python. Browser demonstration results
are never added to the published benchmark statistics.

Only public target knowledge enters shooting policies. Hidden fleets are used
by the game engine to resolve shots and by the optional observer reveal toggle.
Sinking reveals ship identities and all cells; touching ships remain legal.
One shot alternates per turn, including after hits.

Browser games continue while navigating within the page. Closing or reloading
the page stops them, marking the saved replay interrupted on the next visit.
Storage is private to that browser/site path, not synchronized to GitHub or other
visitors. If browser storage fails, a visible warning explains that the game is
in-memory only; use **Download replay** to retain it. Modern JavaScript module,
BigInt, and Web Crypto support is required for browser games.

## Checks

```sh
.venv/bin/python -m pytest tests/test_ui_static.py tests/test_ui.py -q
node tests/test_ui_browser.mjs
node tests/test_ui_replay.mjs
```

The Python checks cover export sanitization, full-statistics/sample separation,
reproducible snapshots, guarded output replacement, symlink rejection, and a
standard-library-only build. Browser-engine checks cover model inference,
shooting/placement, legal shots, seed precision, reproducibility, and termination.
