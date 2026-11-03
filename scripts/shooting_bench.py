"""Local shooting benchmark: clear target fleets without incoming fire."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for package in ("protocol", "client", "server"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))

import numpy as np

from battleship_client.policies import create_policy
from battleship_client.weights import WeightPlayer, load_weight_source
from battleship_protocol import (
    FLEET, NETWORK_VERSION, OBSERVATION_VERSION, PROTOCOL_VERSION, Coordinate, ShipPlacement,
)
from battleship_server.engine import Game, validate_fleet

sys.path.insert(0, str(Path(__file__).resolve().parent))
from placement_strategies import (
    FAMILY_DESCRIPTIONS, FAMILIES, MAX_FLEET_ATTEMPTS, draw_fleet as draw_strategy_fleet,
)

VERSION = "1.0"
SCHEDULE = "battleship-shooting-1.0"
POLICIES = ("model", "random", "hunt")
BOOTSTRAPS = 10_000
MAX_UNIQUE_ATTEMPTS = 1000


def seed_for(master, *parts):
    payload = json.dumps([SCHEDULE, master, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def fixed_own_fleet():
    return [ShipPlacement(name=name, start=Coordinate(row=i, column=0), orientation="horizontal")
            for i, name in enumerate(FLEET)]


def draw_fleet(family, rng):
    return draw_strategy_fleet(family, rng, max_attempts=MAX_FLEET_ATTEMPTS)


def canonical_fleet(ships):
    fleet = validate_fleet(ships)
    return tuple((name, tuple(sorted(fleet[name]))) for name in FLEET)


def board_specs(boards, families):
    for i, family in enumerate(families):
        count = boards // len(families) + (i < boards % len(families))
        for index in range(count):
            yield family, index


def generate_layout(master, family, index, seen):
    seed = seed_for(master, family, index, "placement")
    rng = random.Random(seed)
    for _ in range(MAX_UNIQUE_ATTEMPTS):
        ships = draw_fleet(family, rng)
        key = canonical_fleet(ships)
        if key not in seen:
            seen.add(key)
            return {"board_id": f"{family}:{index}", "family": family, "index": index,
                    "placement_seed": seed,
                    "ships": [ship.model_dump(mode="json") for ship in ships]}
    raise ValueError(f"Unique layout generation exhausted retries for {family}:{index}")


def fresh_policy(name, weights, seed):
    return WeightPlayer(weights, seed) if name == "model" else create_policy(name, seed)


def checked_shot(policy, observation):
    if (observation.phase != "playing" or observation.turn != observation.player
            or observation.result is not None):
        raise ValueError("Policy requested outside an active shooting turn")
    shot = policy.choose_shot(observation)
    if (not isinstance(shot, Coordinate) or not 0 <= shot.row < 10
            or not 0 <= shot.column < 10
            or not observation.legal_shot_mask[shot.row][shot.column]):
        raise ValueError("Policy returned an illegal shot")
    return shot


def trial_metrics(events):
    outcomes = [e.outcome != "miss" for e in events]
    n = len(events)
    if not 17 <= n <= 100 or sum(outcomes) != 17 or any(e.attacker != 0 for e in events):
        raise ValueError("Invalid completed shooting trial")
    return {
        "shots_to_clear": n, "misses": n - 17,
        "first_hit": next(i for i, hit in enumerate(outcomes, 1) if hit),
        "first_sink": next(i for i, e in enumerate(events, 1) if e.outcome == "sunk"),
        "hits_at": {str(k): sum(outcomes[:k]) for k in (10, 25, 50)},
    }


def run_trial(policy, ships, own_fleet=None):
    game = Game("shooting-bench", shooting_only=True)
    game.join()
    game.place(0, fixed_own_fleet() if own_fleet is None else own_fleet)
    game.place(1, ships)
    for _ in range(100):
        observation = game.observe(0)
        game.fire(0, checked_shot(policy, observation))
        if game.result is not None:
            if (game.result.winner != 0 or game.result.reason != "fleet_destroyed"
                    or game.shots[1]):
                raise ValueError("Unexpected shooting trial termination")
            return {**trial_metrics(game.history),
                    "history": [event.model_dump(mode="json") for event in game.history]}
    raise ValueError("Shooting trial exceeded 100 shots")


def mean_ci(groups, seed, samples=None):
    """Stratified layout bootstrap: each family retains its observed sample count."""
    arrays = [np.asarray(values, dtype=np.float64) for values in groups.values() if len(values)]
    if not arrays:
        return None
    samples = BOOTSTRAPS if samples is None else samples
    total = sum(len(a) for a in arrays)
    rng = np.random.default_rng(seed)
    batch = max(1, min(256, 1_000_000 // max(map(len, arrays))))
    means = []
    for start in range(0, samples, batch):
        size = min(batch, samples - start)
        sums = np.zeros(size)
        for values in arrays:
            indexes = rng.integers(0, len(values), size=(size, len(values)))
            sums += values[indexes].sum(axis=1)
        means.extend(sums / total)
    return np.quantile(means, [0.025, 0.975]).tolist()


def policy_metrics(records, families, seed):
    if not records:
        return {"completed_trials": 0, "mean_shots": None, "mean_shots_ci95": None,
                "median_shots": None, "p90_shots": None, "mean_first_hit": None,
                "mean_first_sink": None, "mean_misses": None, "aggregate_hit_rate": None,
                "mean_hits_at": {str(k): None for k in (10, 25, 50)}}
    shots = [r["shots_to_clear"] for r in records]
    return {
        "completed_trials": len(records), "mean_shots": float(np.mean(shots)),
        "mean_shots_ci95": mean_ci(
            {f: [r["shots_to_clear"] for r in records if r["family"] == f] for f in families}, seed),
        "median_shots": float(np.median(shots)), "p90_shots": float(np.quantile(shots, 0.9)),
        "mean_first_hit": float(np.mean([r["first_hit"] for r in records])),
        "mean_first_sink": float(np.mean([r["first_sink"] for r in records])),
        "mean_misses": float(np.mean([r["misses"] for r in records])),
        "aggregate_hit_rate": 17 * len(records) / sum(shots),
        "mean_hits_at": {str(k): float(np.mean([r["hits_at"][str(k)] for r in records]))
                         for k in (10, 25, 50)},
    }


def comparison_metrics(records, families, baseline, seed):
    indexed = {policy: {r["board_id"]: r for r in records if r["policy"] == policy}
               for policy in ("model", baseline)}
    groups = {family: [] for family in families}
    for board_id in sorted(indexed["model"].keys() & indexed[baseline].keys()):
        model, other = indexed["model"][board_id], indexed[baseline][board_id]
        if model["family"] in groups:
            groups[model["family"]].append(model["shots_to_clear"] - other["shots_to_clear"])
    values = [v for group in groups.values() for v in group]
    return {"matched_layouts": len(values),
            "mean_shot_difference": float(np.mean(values)) if values else None,
            "ci95": mean_ci(groups, seed)}


def summarize(records, families, master, complete):
    report = {
        "report_version": VERSION, "status": "complete" if complete else "incomplete",
        "completed_trials": len(records),
        "primary_metric": "shots to clear all 17 ship cells; lower is better",
        "interval_method": "10000 layout-bootstrap resamples; overall stratified by family",
        "comparison_note": "model minus baseline shots on matched layouts; negative favors model",
        "policies": {}, "comparisons": {},
    }
    for policy in POLICIES:
        own = [r for r in records if r["policy"] == policy]
        report["policies"][policy] = {
            "overall": policy_metrics(own, families, seed_for(master, "statistics", policy, "overall")),
            "by_family": {
                family: policy_metrics([r for r in own if r["family"] == family], (family,),
                                       seed_for(master, "statistics", policy, family))
                for family in families},
        }
    for baseline in ("random", "hunt"):
        report["comparisons"][f"model_minus_{baseline}"] = {
            "overall": comparison_metrics(records, families, baseline,
                                           seed_for(master, "statistics", "difference", baseline, "overall")),
            "by_family": {
                family: comparison_metrics(records, (family,), baseline,
                                           seed_for(master, "statistics", "difference", baseline, family))
                for family in families},
        }
    return report


def markdown(report):
    def number(value, percent=False):
        return "—" if value is None else (f"{value:.1%}" if percent else f"{value:.2f}")

    def interval(values):
        return "—" if values is None else " to ".join(number(v) for v in values)

    lines = ["# Shooting-only benchmark", "", f"Run: **{report['status']}**. "
             f"Completed trials: {report['completed_trials']}.", "", report["primary_metric"], "",
             "No incoming fire; every completed trial destroys all five enemy ships.", "",
             "| Policy | Family | Trials | Mean shots (95% CI) | Median | P90 | First hit | First sink | Misses | Hit rate | Hits @10 / @25 / @50 |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for policy, rows in report["policies"].items():
        for family, row in [("overall", rows["overall"]), *rows["by_family"].items()]:
            checkpoints = " / ".join(number(v) for v in row["mean_hits_at"].values())
            lines.append(f"| {policy} | {family} | {row['completed_trials']} | "
                         f"{number(row['mean_shots'])} ({interval(row['mean_shots_ci95'])}) | "
                         f"{number(row['median_shots'])} | {number(row['p90_shots'])} | "
                         f"{number(row['mean_first_hit'])} | {number(row['mean_first_sink'])} | "
                         f"{number(row['mean_misses'])} | {number(row['aggregate_hit_rate'], True)} | "
                         f"{checkpoints} |")
    lines += ["", report["comparison_note"], "",
              "| Comparison | Family | Matched layouts | Mean shot difference | 95% CI |",
              "|---|---|---:|---:|---|"]
    for comparison, rows in report["comparisons"].items():
        for family, row in [("overall", rows["overall"]), *rows["by_family"].items()]:
            lines.append(f"| {comparison} | {family} | {row['matched_layouts']} | "
                         f"{number(row['mean_shot_difference'])} | {interval(row['ci95'])} |")
    lines += ["", report["interval_method"], "",
              "Intervals describe this placement suite; they do not measure defensive placement."]
    if report.get("failure"):
        lines += ["", f"Failure: {report['failure']['type']}. See summary.json for context."]
    return "\n".join(lines) + "\n"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_fingerprints():
    paths = [Path(__file__).resolve(), ROOT / "Makefile", ROOT / "scripts/placement_strategies.py"]
    for package in ("protocol", "client", "server"):
        paths.extend(sorted((ROOT / "packages" / package / "src").rglob("*.py")))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def benchmark(args, weights):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    allocation = dict.fromkeys(args.placements, 0)
    for family, _ in board_specs(args.boards, args.placements):
        allocation[family] += 1
    manifest = {
        "report_version": VERSION, "schedule_version": SCHEDULE,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "incomplete",
        "weights": str(args.weights.resolve()), "model_content_hash": weights.content_hash,
        "configuration": {"boards": args.boards, "seed": args.seed,
                          "placements": args.placements, "allocation": allocation,
                          "policies": POLICIES, "bootstrap_samples": BOOTSTRAPS,
                          "max_fleet_attempts": MAX_FLEET_ATTEMPTS,
                          "max_unique_attempts": MAX_UNIQUE_ATTEMPTS,
                          "sampling_temperature": 1.0, "turn_timeout": None,
                          "execution": "local shooting-only; no HTTP"},
        "placement_definitions": {f: FAMILY_DESCRIPTIONS[f] for f in args.placements},
        "seed_derivation": "First 8 SHA-256 bytes, big endian, of compact JSON "
                           "[schedule_version, master_seed, ...stream_parts]. "
                           "Placement parts: [family,index,'placement']; "
                           "policy parts: [family,index,'policy',policy_name]. "
                           "Statistics parts: ['statistics',policy_name,scope] or "
                           "['statistics','difference',baseline_name,scope].",
        "versions": {"network": NETWORK_VERSION, "observation": OBSERVATION_VERSION,
                     "protocol": PROTOCOL_VERSION},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "dependencies": {name: importlib.metadata.version(name)
                                     for name in ("numpy", "pydantic", "httpx")},
                    "thread_environment": {key: os.environ.get(key) for key in
                                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                            "VECLIB_MAXIMUM_THREADS")}},
        "source_sha256": source_fingerprints(),
    }
    write_json(args.output / "manifest.json", manifest)
    records, layouts = [], []
    context = {}
    failure = None
    complete = False
    print(f"Shooting benchmark: {args.boards} boards, {3 * args.boards} trials. "
          f"Output: {args.output}", flush=True)
    try:
        with (args.output / "layouts.jsonl").open("x") as targets, \
                (args.output / "trials.jsonl").open("x") as trials:
            seen = set()
            for family, index in board_specs(args.boards, args.placements):
                context = {"stage": "generation", "family": family, "index": index}
                layout = generate_layout(args.seed, family, index, seen)
                targets.write(json.dumps(layout, separators=(",", ":")) + "\n")
                targets.flush()
                layouts.append(layout)
            manifest["layouts_sha256"] = hashlib.sha256((args.output / "layouts.jsonl").read_bytes()).hexdigest()
            write_json(args.output / "manifest.json", manifest)
            print(f"Generated {len(layouts)} distinct target layouts: {allocation}", flush=True)
            for i, layout in enumerate(layouts, 1):
                ships = [ShipPlacement.model_validate(ship) for ship in layout["ships"]]
                for policy_name in POLICIES:
                    policy_seed = seed_for(args.seed, layout["family"], layout["index"], "policy", policy_name)
                    context = {"stage": "shooting", "board_id": layout["board_id"],
                               "family": layout["family"], "index": layout["index"],
                               "policy": policy_name, "placement_seed": layout["placement_seed"],
                               "policy_seed": policy_seed}
                    policy = fresh_policy(policy_name, weights, policy_seed)
                    result = run_trial(policy, ships)
                    record = {"report_version": VERSION, **context, **result}
                    trials.write(json.dumps(record, separators=(",", ":")) + "\n")
                    trials.flush()
                    records.append({k: v for k, v in record.items() if k != "history"})
                if i % 25 == 0 or i == args.boards:
                    print(f"Boards {i}/{args.boards}; completed trials {len(records)}/{3 * args.boards} "
                          f"({time.monotonic() - started:.1f}s)", flush=True)
            complete = True
    except (Exception, KeyboardInterrupt) as error:
        failure = {"type": type(error).__name__, "context": context}
    report = summarize(records, args.placements, args.seed, complete)
    report.update({"expected_trials": 3 * args.boards, "generated_layouts": len(layouts),
                   "elapsed_seconds": time.monotonic() - started, "failure": failure})
    manifest["status"] = report["status"]
    write_json(args.output / "summary.json", report)
    (args.output / "summary.md").write_text(markdown(report))
    write_json(args.output / "manifest.json", manifest)
    print(f"{report['status']}. Report: {args.output / 'summary.md'}", flush=True)
    if failure:
        print(f"Stopped: {failure['type']}; see summary.json for context.", file=sys.stderr)
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--boards", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--placements", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    # Canonical family order also fixes allocation of any remainder.
    args.placements = tuple(f for f in FAMILIES if f in args.placements)
    if args.boards < len(args.placements):
        parser.error("--boards must provide at least one board per selected family")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = ROOT / ".battleship/shooting" / f"{args.weights.stem}-{stamp}"
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a fresh directory")
    try:
        weights = load_weight_source(args.weights)
        return benchmark(args, weights)
    except (OSError, ValueError) as error:
        print(f"Benchmark preflight/output error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
