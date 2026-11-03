"""Uncensored defensive evaluation of sequential placement weights.

Trusted orchestration owns fleets; attacking policies receive public observations only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

# Share trusted HTTP lifecycle, safe reporting, and environment fingerprints.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as match_eval

import numpy as np

from battleship_client.policies import create_policy
from battleship_client.sdk import BattleshipClient
from battleship_client.weights import PlacementWeightPlayer, WeightPlayer, load_weight_source, parse_edits
from battleship_protocol import Coordinate, FLEET, ShipPlacement
from battleship_server.engine import Game, validate_fleet
from placement_strategies import FAMILY_DESCRIPTIONS, MAX_FLEET_ATTEMPTS, draw_fleet

ROOT = Path(__file__).resolve().parents[1]
VERSION = "placement-evaluation-1.1"
SCHEDULE = "battleship-placement-evaluation-1.1"
STRATEGIES = ("automatic", "edge", "center", "touching", "zero")
DEFAULT_STRATEGIES = STRATEGIES[:-1]
CANDIDATES = ("model", *DEFAULT_STRATEGIES)
ATTACKERS = ("hunt", "random", "weights")
STRATEGY_DESCRIPTIONS = {
    "model": "Supplied placement-1.0 sequential weight sampler",
    "automatic": FAMILY_DESCRIPTIONS["random"],
    **{name: FAMILY_DESCRIPTIONS[name] for name in ("edge", "center", "touching")},
    "zero": "Placement-1.0 with zero weights; uniform over currently legal actions",
}
SMOKE_TRIALS = 5
BOOTSTRAPS = 10_000


def seed_for(master, *parts):
    return match_eval.seed_for(master, *parts, schedule=SCHEDULE)


def trial_seeds(master, phase, trial, attackers):
    return {"fleet": seed_for(master, phase, trial, "fleet"),
            "attackers": {name: seed_for(master, phase, trial, "attacker", name)
                          for name in attackers}}


def fixture_fleet():
    return [ShipPlacement(name=name, start=Coordinate(row=i, column=0), orientation="horizontal")
            for i, name in enumerate(FLEET)]


def defender_shots():
    occupied = {cell for cells in validate_fleet(fixture_fleet()).values() for cell in cells}
    cells = [(r, c) for r in range(10) for c in range(10)]
    # Only the trusted script knows this fixed fixture. Its 17th hit is shot 100.
    return [Coordinate(row=r, column=c) for r, c in cells if (r, c) not in occupied] + [
        Coordinate(row=r, column=c) for r, c in cells if (r, c) in occupied]


def selected_candidates(strategies=None):
    strategies = DEFAULT_STRATEGIES if strategies is None else tuple(strategies)
    if not strategies or set(strategies) - set(STRATEGIES):
        raise ValueError("Select at least one supported comparison strategy")
    return ("model", *(name for name in STRATEGIES if name in strategies))


def selected_attackers(attackers=None, *, has_weights=False):
    if attackers is None:
        return ("hunt", "weights") if has_weights else ("hunt",)
    attackers = tuple(attackers)
    if not attackers or set(attackers) - set(ATTACKERS):
        raise ValueError("Select at least one supported attacker")
    if "weights" in attackers and not has_weights:
        raise ValueError("The weights attacker requires --attacker-weights PATH")
    if has_weights and "weights" not in attackers:
        raise ValueError("--attacker-weights is unused by the selected attackers")
    return tuple(name for name in ATTACKERS if name in attackers)


def generate_fleet(candidate, weights, seed):
    if candidate in ("model", "zero"):
        game = Game("placement-generation")
        ships = PlacementWeightPlayer(weights, seed).choose_placement(game.observe(0))
    elif candidate in STRATEGIES:
        family = "random" if candidate == "automatic" else candidate
        ships = draw_fleet(family, random.Random(seed))
    else:
        raise ValueError(f"Unknown placement strategy: {candidate}")
    validate_fleet(ships)
    return ships


def attacker_policy(name, seed, weights=None):
    return WeightPlayer(weights, seed) if name == "weights" else create_policy(name, seed)


def attack_rollout(observe, fire, policy):
    defense = defender_shots()
    for index in range(100):
        observation = observe(0)
        result = fire(0, match_eval.checked_shot(policy, observation))
        if result.phase == "finished":
            if result.result.winner != 0 or result.result.reason != "fleet_destroyed":
                raise ValueError("Defensive evaluation ended before attacker finished")
            if sum(e.attacker == 0 and e.outcome != "miss" for e in result.shot_history) != 17:
                raise ValueError("Attacker did not destroy all 17 occupied cells")
            return {"attacker_shots": index + 1,
                    "history": [e.model_dump(mode="json") for e in result.shot_history]}
        if fire(1, defense[index]).phase == "finished":
            raise ValueError("Defender fixture script ended the rollout")
    raise ValueError("Attack exceeded 100 legal shots")


def play_local(ships, attacker, seed, attacker_weights=None):
    game = Game("placement-evaluation")
    game.join()
    game.place(0, fixture_fleet())
    game.place(1, ships)

    def fire(seat, shot):
        game.fire(seat, shot)
        return game.observe(seat)

    return attack_rollout(game.observe, fire, attacker_policy(attacker, seed, attacker_weights))


def play_http(server, ships, attacker, seed, attacker_weights=None):
    try:
        with BattleshipClient(server.url) as first, BattleshipClient(server.url) as second:
            room = first.create()
            second.join(room.match_id, room.invitation)
            first.place(fixture_fleet())
            second.place(ships)
            clients = (first, second)
            return attack_rollout(lambda seat: clients[seat].observe(),
                                  lambda seat, shot: clients[seat].fire(shot),
                                  attacker_policy(attacker, seed, attacker_weights))
    finally:
        with server.store.lock:
            server.store.rooms.clear()
            server.store.creates.clear()


def interval(values, seed, *, median=False):
    if not median:
        return match_eval.bootstrap_ci(values, seed, samples=BOOTSTRAPS)
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    results = []
    batch = max(1, min(256, 1_000_000 // len(values)))
    for start in range(0, BOOTSTRAPS, batch):
        indexes = rng.integers(0, len(values), size=(min(batch, BOOTSTRAPS - start), len(values)))
        results.extend(np.median(values[indexes], axis=1))
    return np.quantile(results, [0.025, 0.975]).tolist()


def percentage_interval(model, baseline, seed):
    """Bootstrap the percentage change in means using paired trial indexes."""
    if not model or not baseline:
        return None
    model, baseline = np.asarray(model, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if len(model) != len(baseline) or np.any(baseline <= 0):
        raise ValueError("Percentage intervals require aligned positive baseline counts")
    rng = np.random.default_rng(seed)
    results = []
    batch = max(1, min(256, 1_000_000 // len(model)))
    for start in range(0, BOOTSTRAPS, batch):
        indexes = rng.integers(0, len(model), size=(min(batch, BOOTSTRAPS - start), len(model)))
        denominator = baseline[indexes].mean(axis=1)
        results.extend(100 * (model[indexes].mean(axis=1) - denominator) / denominator)
    return np.quantile(results, [0.025, 0.975]).tolist()


def summarize(records, master, attackers, complete, http_status, *, candidates=CANDIDATES):
    # An interrupted block is excluded as a whole, preserving paired comparisons
    # and the shared-fleet grouping across attacker types.
    groups = {}
    expected = {(c, a) for c in candidates for a in attackers}
    for row in records:
        if row["phase"] == "benchmark":
            groups.setdefault(row["trial"], []).append(row)
    valid = {trial: rows for trial, rows in groups.items()
             if len(rows) == len(expected) and {(r["candidate"], r["attacker"]) for r in rows} == expected}
    report = {"report_version": VERSION, "status": "complete" if complete else "incomplete",
              "http_verification": http_status, "complete_trials": len(valid),
              "incomplete_trials_excluded": len(groups) - len(valid), "matchups": {},
              "differences": {}, "metric": "attacker shots to destroy fleet; higher is better",
              "interval_method": "95% percentile bootstrap of whole trials; 10000 resamples"}
    data = {}
    for candidate in candidates:
        for attacker in attackers:
            scores = {trial: next(r["attacker_shots"] for r in rows
                                  if r["candidate"] == candidate and r["attacker"] == attacker)
                      for trial, rows in sorted(valid.items())}
            data[candidate, attacker] = scores
            values = list(scores.values())
            key = f"{candidate}_vs_{attacker}"
            report["matchups"][key] = {
                "trials": len(values), "mean_shots": float(np.mean(values)) if values else None,
                "median_shots": float(np.median(values)) if values else None,
                "p10_shots": float(np.quantile(values, .1)) if values else None,
                "p90_shots": float(np.quantile(values, .9)) if values else None,
                "mean_ci95": interval(values, seed_for(master, "statistics", key, "mean")),
                "median_ci95": interval(values, seed_for(master, "statistics", key, "median"), median=True)}
    for control in candidates[1:]:
        for attacker in attackers:
            model, baseline = data["model", attacker], data[control, attacker]
            values = [model[t] - baseline[t] for t in sorted(model.keys() & baseline.keys())]
            key = f"model_minus_{control}_vs_{attacker}"
            baseline_mean = float(np.mean(list(baseline.values()))) if baseline else None
            report["differences"][key] = {
                "aligned_trials": len(values),
                "mean_shot_difference": float(np.mean(values)) if values else None,
                "ci95": interval(values, seed_for(master, "statistics", key)),
                "percentage_improvement": (100 * float(np.mean(values)) / baseline_mean
                                           if baseline_mean else None),
                "percentage_ci95": percentage_interval(
                    list(model.values()), list(baseline.values()),
                    seed_for(master, "statistics", key, "percentage"))}
    primary = report["differences"].get("model_minus_automatic_vs_hunt")
    status = "not evaluated" if primary is None else "incomplete"
    if complete and primary is not None and primary["ci95"] is not None:
        low, high = primary["ci95"]
        status = "improved" if low > 0 else "worse" if high < 0 else "inconclusive"
    report["primary_comparison"] = {
        "comparison": "model_minus_automatic_vs_hunt", "status": status,
        "criterion": "complete run with lower paired 95% mean-difference bound strictly above zero"}
    return report


def markdown(report):
    def number(value):
        return "—" if value is None else f"{value:.2f}"

    def ci(value):
        return "—" if value is None else " to ".join(number(v) for v in value)

    lines = ["# Placement evaluation", "", f"Status: {report['status']}. "
             f"HTTP verification: {report['http_verification']}.", "", report["metric"], "",
             "| Placement vs attacker | Trials | Mean shots (95% CI) | Median shots (95% CI) | P10 | P90 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, row in report["matchups"].items():
        lines.append(f"| {key} | {row['trials']} | {number(row['mean_shots'])} "
                     f"({ci(row['mean_ci95'])}) | {number(row['median_shots'])} "
                     f"({ci(row['median_ci95'])}) | {number(row['p10_shots'])} | {number(row['p90_shots'])} |")
    lines += ["", "Model minus control (positive favors model):", ""]
    for key, row in report["differences"].items():
        lines.append(f"- {key}: {number(row['mean_shot_difference'])} "
                     f"shots (95% CI {ci(row['ci95'])}); {number(row['percentage_improvement'])}% "
                     f"(95% CI {ci(row['percentage_ci95'])}); {row['aligned_trials']} aligned trials")
    lines += ["", report["interval_method"], "",
              f"Incomplete trial blocks excluded: {report['incomplete_trials_excluded']}.",
              f"Primary comparison (model vs automatic against hunt): {report['primary_comparison']['status']}.",
              "A positive lower paired 95% confidence bound demonstrates improvement for this benchmark.",
              "Other comparisons are descriptive. No combined attacker score or competitive win rate is inferred.",
              "Small samples and collapsed intervals do not establish general playing strength."]
    if report.get("failure"):
        lines += ["", f"Failure: {report['failure']['type']}; see summary.json."]
    return "\n".join(lines) + "\n"


def evaluate(args, model, attacker_weights=None):
    candidates = selected_candidates(getattr(args, "strategies", None))
    attackers = selected_attackers(getattr(args, "attackers", None), has_weights=attacker_weights is not None)
    weights = {candidate: (model if candidate == "model" else
                          parse_edits("", role="placement") if candidate == "zero" else None)
               for candidate in candidates}
    args.output.mkdir(parents=True, exist_ok=False)
    fingerprints = match_eval.source_fingerprints()
    for name in ("evaluate_placement.py", "placement_strategies.py"):
        fingerprints[f"scripts/{name}"] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    manifest = {
        "report_version": VERSION, "schedule_version": SCHEDULE, "status": "incomplete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "placement_weights": str(args.placement_weights.resolve()),
        "content_hashes": {key: value.content_hash if value is not None else None
                           for key, value in weights.items()},
        "attacker_content_hash": attacker_weights.content_hash if attacker_weights is not None else None,
        "versions": {"placement_network": "placement-1.0", "shooting_network": "1.0",
                     "protocol": "1.0", "observation": "1.0"},
        "configuration": {"trials": args.trials, "seed": args.seed, "attackers": attackers,
                          "candidates": candidates, "smoke_trials": SMOKE_TRIALS,
                          "sampling_with_replacement": True, "max_fleet_attempts": MAX_FLEET_ATTEMPTS,
                          "bootstrap_samples": BOOTSTRAPS, "temperature": 1.0,
                          "fixture": "horizontal fleet on rows 0..4 starting column 0; attacker seat 0",
                          "defender": "water row-major, then occupied row-major; 17th hit at shot 100"},
        "strategy_definitions": {name: STRATEGY_DESCRIPTIONS[name] for name in candidates},
        "seed_derivation": "first 8 SHA-256 bytes, big endian, of compact JSON "
                           "[schedule_version, master_seed, phase, trial, fleet] or "
                           "[schedule_version, master_seed, phase, trial, attacker, name]; "
                           "candidate omitted; statistics use separately tagged streams",
        "source_sha256": fingerprints,
        "runtime": {"python": match_eval.platform.python_version(),
                    "platform": match_eval.platform.platform(),
                    "dependencies": {name: match_eval.importlib.metadata.version(name)
                                     for name in ("numpy", "pydantic", "httpx", "uvicorn", "fastapi")},
                    "thread_environment": {key: match_eval.os.environ.get(key) for key in
                                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                            "VECLIB_MAXIMUM_THREADS")}}}
    match_eval.write_json(args.output / "manifest.json", manifest)
    records, context = [], {}
    started = time.monotonic()
    failure, http_status, complete = None, "not run", False
    try:
        with (args.output / "games.jsonl").open("x") as stream:
            def run_block(phase, trial, server=None):
                nonlocal context
                seed_phase = "benchmark" if phase == "benchmark" else "smoke"
                seeds = trial_seeds(args.seed, seed_phase, trial, attackers)
                for candidate in candidates:
                    context = {"phase": phase, "trial": trial, "candidate": candidate, "seeds": seeds}
                    ships = generate_fleet(candidate, weights[candidate], seeds["fleet"])
                    fleet_hash = hashlib.sha256(json.dumps(
                        [s.model_dump(mode="json") for s in ships],
                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    for attacker in attackers:
                        context = {**context, "attacker": attacker}
                        seed = seeds["attackers"][attacker]
                        if server is None:
                            result = play_local(ships, attacker, seed, attacker_weights)
                        else:
                            result = play_http(server, ships, attacker, seed, attacker_weights)
                            if trial == 0 and result != play_local(ships, attacker, seed, attacker_weights):
                                raise ValueError("Local/HTTP placement parity mismatch")
                        record = {"report_version": VERSION, **context, "fleet_hash": fleet_hash, **result}
                        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                        stream.flush()
                        records.append({key: value for key, value in record.items() if key != "history"})

            for trial in range(SMOKE_TRIALS):
                run_block("smoke", trial)
            print("Local placement smoke passed.", flush=True)
            http_status = "running"
            with match_eval.http_server() as server:
                for trial in range(SMOKE_TRIALS):
                    run_block("http_smoke", trial, server)
            http_status = "passed"
            print("HTTP placement smoke and parity passed.", flush=True)
            for trial in range(args.trials):
                run_block("benchmark", trial)
                if (trial + 1) % 25 == 0 or trial + 1 == args.trials:
                    print(f"Placement trials: {trial + 1}/{args.trials}", flush=True)
            complete = True
    except (Exception, KeyboardInterrupt) as error:
        failure = {"type": type(error).__name__, "context": context}
        if http_status == "running":
            http_status = "failed"
    report = summarize(records, args.seed, attackers, complete, http_status, candidates=candidates)
    report.update(failure=failure, elapsed_seconds=time.monotonic() - started)
    report["rollouts_by_phase"] = {phase: sum(r["phase"] == phase for r in records)
                                   for phase in ("smoke", "http_smoke", "benchmark")}
    manifest["status"] = report["status"]
    match_eval.write_json(args.output / "manifest.json", manifest)
    match_eval.write_json(args.output / "summary.json", report)
    (args.output / "summary.md").write_text(markdown(report))
    print(f"{report['status']}: {args.output / 'summary.md'}", flush=True)
    if failure:
        print(f"Placement evaluation stopped: {failure['type']}; see summary.json.", file=sys.stderr)
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-weights", type=Path, required=True)
    parser.add_argument("--attacker-weights", type=Path)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=DEFAULT_STRATEGIES,
                        help="Comparison strategies; model is always included")
    parser.add_argument("--attackers", nargs="+", choices=ATTACKERS,
                        help="Default: hunt, plus weights when --attacker-weights is supplied")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = ROOT / ".battleship/evaluations" / f"placement-{args.placement_weights.stem}-{stamp}"
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a fresh directory")
    try:
        args.strategies = selected_candidates(args.strategies)[1:]
        args.attackers = selected_attackers(args.attackers, has_weights=args.attacker_weights is not None)
        model = load_weight_source(args.placement_weights, role="placement")
        attacker_weights = load_weight_source(args.attacker_weights) if args.attacker_weights else None
        return evaluate(args, model, attacker_weights)
    except (OSError, ValueError, ImportError) as error:
        print(f"Placement evaluation preflight/output error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
