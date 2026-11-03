"""Trusted evaluation orchestration; player policies receive observations only."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
# Permit the documented direct invocation without installing repository packages.
for package in ("protocol", "client", "server"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))

import numpy as np
import uvicorn

from battleship_client.policies import create_policy, placement_seed
from battleship_client.sdk import BattleshipClient
from battleship_client.weights import PlacementWeightPlayer, WeightPlayer, load_weight_source, parse_edits
from battleship_protocol import NETWORK_VERSION, OBSERVATION_VERSION, PROTOCOL_VERSION, Coordinate
from battleship_server.app import MatchStore, create_app
from battleship_server.engine import Game

VERSION = "1.1"
SCHEDULE = "battleship-evaluation-1.0"
TWO_MODEL_SCHEDULE = "battleship-evaluation-two-model-1.0"
CANDIDATES = ("model", "zero")
OPPONENTS = ("random", "hunt")
SMOKE_PAIRS = 5
BOOTSTRAPS = 10_000


def seed_for(master, *parts, schedule=SCHEDULE):
    payload = json.dumps([schedule, master, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def trial_seeds(master, phase, trial, opponent, *, schedule=SCHEDULE):
    # Candidate and placement streams intentionally omit candidate/opponent identity.
    return {
        "placement": seed_for(master, phase, trial, "placement", schedule=schedule),
        "candidate": seed_for(master, phase, trial, "candidate", schedule=schedule),
        "opponent": seed_for(master, phase, trial, "opponent", opponent, schedule=schedule),
    }


def players(weights, opponent, seeds):
    return (WeightPlayer(weights, seeds["candidate"]),
            create_policy(opponent, seeds["opponent"]))


def checked_shot(policy, observation):
    if (observation.phase != "playing" or observation.turn != observation.player
            or observation.result is not None):
        raise ValueError("Policy requested outside its active turn")
    shot = policy.choose_shot(observation)
    if not isinstance(shot, Coordinate):
        raise ValueError("Policy must return a Coordinate")
    if not (0 <= shot.row < 10 and 0 <= shot.column < 10
            and observation.legal_shot_mask[shot.row][shot.column]):
        raise ValueError("Policy returned an illegal shot")
    return shot


def completed_record(observation, candidate_seat):
    if (observation.phase != "finished" or observation.result is None
            or observation.result.reason != "fleet_destroyed"):
        raise ValueError("Unexpected game termination")
    history = observation.shot_history
    if not 1 <= len(history) <= 199:
        raise ValueError("Unexpected game length")
    shots = [sum(e.attacker == seat for e in history) for seat in (0, 1)]
    hits = [sum(e.attacker == seat and e.outcome != "miss" for e in history)
            for seat in (0, 1)]
    if hits[observation.result.winner] != 17:
        raise ValueError("Winner has not destroyed the fleet")
    return {
        "candidate_seat": candidate_seat,
        "winner": observation.result.winner,
        "candidate_won": observation.result.winner == candidate_seat,
        "finish_reason": observation.result.reason,
        "candidate_shots": shots[candidate_seat],
        "opponent_shots": shots[1 - candidate_seat],
        "candidate_hits": hits[candidate_seat],
        "opponent_hits": hits[1 - candidate_seat],
        "history": [e.model_dump(mode="json") for e in history],
    }


def candidate_placement(weights, seeds, observation):
    if weights is None:
        return None
    return PlacementWeightPlayer(weights, placement_seed(seeds["candidate"])).choose_placement(observation)


def play_local(weights, opponent, seeds, candidate_seat, placement_weights=None):
    game = Game("evaluation", rng=random.Random(seeds["placement"]))
    game.join()
    game.place(candidate_seat, candidate_placement(placement_weights, seeds, game.observe(candidate_seat)))
    game.place(1 - candidate_seat)
    candidate, baseline = players(weights, opponent, seeds)
    policies = {candidate_seat: candidate, 1 - candidate_seat: baseline}
    for _ in range(199):
        seat = game.turn
        observation = game.observe(seat)
        game.fire(seat, checked_shot(policies[seat], observation))
        if game.result is not None:
            return completed_record(game.observe(candidate_seat), candidate_seat)
    raise ValueError("Game exceeded 199 legal alternating shots")


class HTTPRunner:
    def __init__(self):
        self.placement_seed = 0
        self.store = MatchStore(rng_factory=lambda: random.Random(self.placement_seed))
        self.url = None

    def play(self, weights, opponent, seeds, candidate_seat, placement_weights=None):
        self.placement_seed = seeds["placement"]
        try:
            with BattleshipClient(self.url) as first, BattleshipClient(self.url) as second:
                clients = (first, second)
                room = first.create()
                second.join(room.match_id, room.invitation)
                clients[candidate_seat].place(candidate_placement(
                    placement_weights, seeds, clients[candidate_seat].observe()))
                clients[1 - candidate_seat].place()
                candidate, baseline = players(weights, opponent, seeds)
                policies = {candidate_seat: candidate, 1 - candidate_seat: baseline}
                for turn in range(199):
                    seat = turn % 2
                    observation = clients[seat].observe()
                    result = clients[seat].fire(checked_shot(policies[seat], observation))
                    if result.phase == "finished":
                        return completed_record(result, candidate_seat)
                raise ValueError("HTTP game exceeded 199 shots")
        finally:
            # Trusted lifecycle cleanup only: no public administrative endpoint.
            with self.store.lock:
                self.store.rooms.clear()
                self.store.creates.clear()


@contextmanager
def http_server():
    runner = HTTPRunner()
    sock = socket.socket()
    thread = None
    server = None
    try:
        sock.bind(("127.0.0.1", 0))
        runner.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(
            create_app(store=runner.store), log_level="error", access_log=False))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("Evaluation HTTP server did not start")
            time.sleep(0.01)
        yield runner
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None and thread.ident is not None:
            thread.join(timeout=5)
        sock.close()
        if thread is not None and thread.is_alive():
            raise RuntimeError("Evaluation HTTP server did not stop")


def paired_rows(records):
    groups = {}
    for record in records:
        groups.setdefault(record["trial"], []).append(record)
    return [sorted(group, key=lambda r: r["candidate_seat"])
            for _, group in sorted(groups.items())
            if len(group) == 2 and {r["candidate_seat"] for r in group} == {0, 1}]


def bootstrap_ci(values, seed, samples=BOOTSTRAPS):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = []
    # Bound memory even for larger user-selected trial counts.
    batch = max(1, min(256, 1_000_000 // len(values)))
    for start in range(0, samples, batch):
        indexes = rng.integers(0, len(values), size=(min(batch, samples - start), len(values)))
        means.extend(values[indexes].mean(axis=1))
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(records, master, complete, http_status, opponents=OPPONENTS,
              candidates=CANDIDATES, *, schedule=SCHEDULE):
    report = {
        "report_version": VERSION, "status": "complete" if complete else "incomplete",
        "http_verification": http_status,
        "criterion": "model vs random: lower 95% paired-bootstrap bound > 0.5",
        "strength": "incomplete", "matchups": {}, "model_minus_zero": {},
        "interval_method": "percentile bootstrap of whole pairs; 10000 resamples",
        "shot_count_note": "Shots taken at game termination; losers have incomplete searches.",
    }
    pair_scores = {}
    for candidate in candidates:
        for opponent in opponents:
            raw = [r for r in records if r["phase"] == "benchmark"
                   and r["candidate"] == candidate and r["opponent"] == opponent]
            pairs = paired_rows(raw)
            games = [r for pair in pairs for r in pair]
            scores = {pair[0]["trial"]: sum(r["candidate_won"] for r in pair) / 2
                      for pair in pairs}
            key = f"{candidate}_vs_{opponent}"
            pair_scores[key] = scores
            n = len(games)
            wins = sum(r["candidate_won"] for r in games)
            shots = sum(r["candidate_shots"] for r in games)
            report["matchups"][key] = {
                "completed_games": len(raw), "complete_pairs": len(pairs),
                "unpaired_games_excluded": len(raw) - n,
                "wins": wins, "losses": n - wins,
                "win_rate": wins / n if n else None,
                "ci95": bootstrap_ci(list(scores.values()), seed_for(master, "statistics", key, schedule=schedule)),
                "win_rate_by_start": {
                    label: (sum(r["candidate_won"] for r in games if r["candidate_seat"] == seat)
                            / len(pairs) if pairs else None)
                    for seat, label in ((0, "first"), (1, "second"))},
                "hit_rate": sum(r["candidate_hits"] for r in games) / shots if shots else None,
                "mean_shots_at_termination": {
                    role: sum(r[f"{role}_shots"] for r in games) / n if n else None
                    for role in ("candidate", "opponent")},
            }
    for opponent in opponents if "zero" in candidates else ():
        model = pair_scores[f"model_vs_{opponent}"]
        zero = pair_scores[f"zero_vs_{opponent}"]
        differences = [model[i] - zero[i] for i in sorted(model.keys() & zero.keys())]
        report["model_minus_zero"][opponent] = {
            "aligned_pairs": len(differences),
            "win_rate_difference": float(np.mean(differences)) if differences else None,
            "ci95": bootstrap_ci(differences, seed_for(master, "statistics", "difference", opponent, schedule=schedule)),
        }
    ci = report["matchups"].get("model_vs_random", {}).get("ci95")
    if complete and "random" not in opponents:
        report["strength"] = "not evaluated"
    if complete and ci is not None:
        report["strength"] = "passed" if ci[0] > 0.5 else "not demonstrated"
    return report


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def markdown(report):
    def percent(value):
        return "—" if value is None else f"{value:.1%}"

    lines = ["# Battleship evaluation", "", f"Run: **{report['status']}**. "
             f"Random strength criterion: **{report['strength']}**. "
             f"HTTP verification: **{report['http_verification']}**.", "",
             report["criterion"], "", "| Matchup | Complete pairs | Wins / losses | Win rate | 95% CI | First / second | Hit rate | Mean shots (candidate / opponent) |",
             "|---|---:|---:|---:|---|---|---:|---|"]
    for name, row in report["matchups"].items():
        ci = row["ci95"]
        interval = "—" if ci is None else " to ".join(map(percent, ci))
        starts = row["win_rate_by_start"]
        shots = row["mean_shots_at_termination"]
        shot_text = " / ".join("—" if value is None else f"{value:.1f}"
                               for value in shots.values())
        lines.append(f"| {name} | {row['complete_pairs']} | {row['wins']} / {row['losses']} | "
                     f"{percent(row['win_rate'])} | {interval} | "
                     f"{percent(starts['first'])} / {percent(starts['second'])} | "
                     f"{percent(row['hit_rate'])} | {shot_text} |")
    lines += ["", report["shot_count_note"], "", report["interval_method"]]
    if report["model_minus_zero"]:
        lines += ["", "Model minus zero-network win rate (aligned pairs):", ""]
    for opponent, row in report["model_minus_zero"].items():
        ci = row["ci95"]
        interval = "—" if ci is None else " to ".join(map(percent, ci))
        lines.append(f"- {opponent}: {percent(row['win_rate_difference'])} "
                     f"(95% CI {interval}; {row['aligned_pairs']} pairs)")
    if report.get("failure"):
        lines += ["", f"Failure: {report['failure']['type']}. See summary.json for trial context."]
    return "\n".join(lines) + "\n"


def source_fingerprints():
    paths = [Path(__file__).resolve(), ROOT / "Makefile"]
    for package in ("protocol", "client", "server"):
        paths.extend(sorted((ROOT / "packages" / package / "src").rglob("*.py")))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def evaluate(args, model, placement_weights=None):
    schedule = TWO_MODEL_SCHEDULE if placement_weights is not None else SCHEDULE
    opponents = tuple(dict.fromkeys(getattr(args, "opponents", OPPONENTS)))
    candidates = ("model",) if getattr(args, "model_only", False) else CANDIDATES
    weights = {"model": model}
    if "zero" in candidates:
        weights["zero"] = parse_edits("")
    matchups = len(candidates) * len(opponents)
    total_games = 2 * matchups * args.pairs
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "report_version": VERSION, "schedule_version": schedule,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "incomplete",
        "weights": str(args.weights.resolve()), "model_content_hash": model.content_hash,
        "placement_content_hash": placement_weights.content_hash if placement_weights is not None else None,
        "zero_content_hash": weights["zero"].content_hash if "zero" in weights else None,
        "versions": {"network": NETWORK_VERSION, "observation": OBSERVATION_VERSION,
                     "protocol": PROTOCOL_VERSION},
        "configuration": {"pairs": args.pairs, "seed": args.seed, "smoke_pairs": SMOKE_PAIRS,
                          "bootstrap_samples": BOOTSTRAPS, "opponents": opponents,
                          "candidates": candidates, "turn_timeout": None,
                          "placement_distribution": "placement-1.0" if placement_weights is not None else "engine.random_fleet",
                          "sampling_temperature": 1.0, "execution": "local",
                          "http_smoke": True},
        "seed_derivation": "first 8 SHA-256 bytes, big endian, of compact JSON "
                           "[schedule_version, master_seed, ...stream_parts]; see source",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "dependencies": {name: importlib.metadata.version(name)
                                     for name in ("numpy", "pydantic", "httpx", "uvicorn", "fastapi")},
                    "thread_environment": {key: os.environ.get(key) for key in
                                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                            "VECLIB_MAXIMUM_THREADS")}},
        "source_sha256": source_fingerprints(),
    }
    write_json(args.output / "manifest.json", manifest)
    records = []
    context = {}
    http_status = "not run"
    failure = None
    complete = False
    try:
        with (args.output / "games.jsonl").open("x") as stream:
            def run_game(phase, trial, candidate, opponent, seat, runner, seed_phase=None):
                nonlocal context
                seeds = trial_seeds(args.seed, seed_phase or phase, trial, opponent, schedule=schedule)
                context = {"phase": phase, "trial": trial, "candidate": candidate,
                           "opponent": opponent, "candidate_seat": seat, "seeds": seeds}
                if placement_weights is None:
                    result = runner(weights[candidate], opponent, seeds, seat)
                else:
                    result = runner(weights[candidate], opponent, seeds, seat, placement_weights)
                record = {"report_version": VERSION, **context, **result}
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                records.append({k: v for k, v in record.items() if k != "history"})
                return result

            for candidate in candidates:
                for opponent in opponents:
                    for trial in range(SMOKE_PAIRS):
                        for seat in (0, 1):
                            run_game("smoke", trial, candidate, opponent, seat, play_local)
            smoke_elapsed = time.monotonic() - started
            smoke_games = 2 * matchups * SMOKE_PAIRS
            estimate = smoke_elapsed / smoke_games * total_games
            print(f"Local smoke passed ({smoke_games} games, {smoke_elapsed:.1f}s). "
                  f"Estimated benchmark: {estimate / 60:.1f} minutes.", flush=True)
            http_status = "running"
            with http_server() as server:
                for candidate in candidates:
                    for opponent in opponents:
                        for trial in range(SMOKE_PAIRS):
                            for seat in (0, 1):
                                result = run_game("http_smoke", trial, candidate, opponent,
                                                  seat, server.play, "smoke")
                                if trial == 0:
                                    local = run_game("parity", trial, candidate, opponent,
                                                     seat, play_local, "smoke")
                                    if local != result:
                                        raise ValueError("Local/HTTP parity mismatch")
            http_status = "passed"
            print(f"HTTP smoke passed ({smoke_games} games); all {2 * matchups} parity replays matched.",
                  flush=True)
            done = 0
            for candidate in candidates:
                for opponent in opponents:
                    for trial in range(args.pairs):
                        for seat in (0, 1):
                            run_game("benchmark", trial, candidate, opponent, seat, play_local)
                            done += 1
                        if (trial + 1) % 25 == 0 or trial + 1 == args.pairs:
                            print(f"{candidate} vs {opponent}: {trial + 1}/{args.pairs} pairs "
                                  f"({done}/{total_games} benchmark games)", flush=True)
            complete = True
    except (Exception, KeyboardInterrupt) as error:
        # Persist context without exception strings that might contain HTTP credentials.
        failure = {"type": type(error).__name__, "context": context}
        if http_status == "running":
            http_status = "failed"
    report = summarize(records, args.seed, complete, http_status, opponents, candidates, schedule=schedule)
    report["failure"] = failure
    report["elapsed_seconds"] = time.monotonic() - started
    report["games_by_phase"] = {
        phase: sum(r["phase"] == phase for r in records)
        for phase in ("smoke", "http_smoke", "parity", "benchmark")}
    manifest["status"] = report["status"]
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "summary.json", report)
    (args.output / "summary.md").write_text(markdown(report))
    print(f"{report['status']}: random criterion {report['strength']}. "
          f"Report: {args.output / 'summary.md'}", flush=True)
    if failure:
        print(f"Evaluation stopped: {failure['type']}; see summary.json.", file=sys.stderr)
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/shooting_attempt_v1_q1.dat")
    parser.add_argument("--placement-weights", type=Path)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=OPPONENTS)
    parser.add_argument("--model-only", action="store_true", help="Omit the zero-network control")
    parser.add_argument("--output", type=Path, help="Fresh directory; default: unique .battleship/evaluations run")
    args = parser.parse_args(argv)
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        label = "-".join(dict.fromkeys(args.opponents))
        args.output = ROOT / ".battleship/evaluations" / f"{args.weights.stem}-{label}-{stamp}"
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a fresh directory")
    try:
        model = load_weight_source(args.weights)
        placement_weights = (load_weight_source(args.placement_weights, role="placement")
                             if args.placement_weights is not None else None)
        return evaluate(args, model, placement_weights)
    except (OSError, ValueError) as error:
        print(f"Evaluation preflight/output error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
