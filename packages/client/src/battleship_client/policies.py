"""Policies use authorized observations without engine imports or hidden fleet access."""

import hashlib
import json
import random
from pathlib import Path
from typing import Protocol

from battleship_protocol import Coordinate, Observation, ShipPlacement
from battleship_client.sdk import BattleshipClient


class ShotPolicy(Protocol):
    def choose_shot(self, observation: Observation) -> Coordinate: ...


class PlacementPolicy(Protocol):
    def choose_placement(self, observation: Observation) -> list[ShipPlacement] | None: ...


def legal_cells(observation: Observation) -> list[tuple[int, int]]:
    return [(r, c) for r in range(10) for c in range(10) if observation.legal_shot_mask[r][c]]


class RandomBot:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_shot(self, observation: Observation) -> Coordinate:
        choices = legal_cells(observation)
        if not choices:
            raise ValueError("No legal shots available")
        r, c = self.rng.choice(choices)
        return Coordinate(row=r, column=c)


class HuntTargetBot(RandomBot):
    def choose_shot(self, observation: Observation) -> Coordinate:
        legal = legal_cells(observation)
        if not legal:
            raise ValueError("No legal shots available")
        hits = {(r, c) for r in range(10) for c in range(10)
                if observation.target_board[r][c] == "hit"}
        scores = {}
        # Score each legal cell independently. Touching hit groups are never committed
        # to a single ship or orientation; sunk cells alone are removed from pursuit.
        for r, c in legal:
            score = 0
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if (r + dr, c + dc) in hits:
                    score += 1
                    if (r + 2 * dr, c + 2 * dc) in hits:
                        score += 2
            scores[(r, c)] = score
        best = max(scores.values())
        if best:
            choices = [cell for cell in legal if scores[cell] == best]
        else:
            choices = [cell for cell in legal if sum(cell) % 2 == 0] or legal
        r, c = self.rng.choice(choices)
        return Coordinate(row=r, column=c)


def placement_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    data = json.dumps(["battleship-placement-1.0", seed], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


class ComposedPlayer:
    def __init__(self, shooting: ShotPolicy, placement: PlacementPolicy):
        self.shooting = shooting
        self.placement = placement

    def choose_shot(self, observation: Observation) -> Coordinate:
        return self.shooting.choose_shot(observation)

    def choose_placement(self, observation: Observation) -> list[ShipPlacement] | None:
        return self.placement.choose_placement(observation)


def create_policy(kind: str, seed: int | None = None, weights_path: str | Path | None = None,
                  placement_weights_path: str | Path | None = None) -> ShotPolicy:
    """Preflight both models before session or match activity; NumPy is lazy."""
    if kind == "weights":
        if not weights_path:
            raise ValueError("The weights bot requires --weights PATH (Make: WEIGHTS=PATH).")
        from battleship_client.weights import WeightPlayer, load_weight_source
        shooting = WeightPlayer(load_weight_source(weights_path), seed)
    else:
        if weights_path is not None:
            raise ValueError("--weights is only valid for the weights bot")
        if kind == "random":
            shooting = RandomBot(seed)
        elif kind == "hunt":
            shooting = HuntTargetBot(seed)
        else:
            raise ValueError(f"Unknown bot kind: {kind}")
    if placement_weights_path is None:
        return shooting
    from battleship_client.weights import PlacementWeightPlayer, load_weight_source
    placement = PlacementWeightPlayer(load_weight_source(placement_weights_path, role="placement"),
                                      placement_seed(seed))
    return ComposedPlayer(shooting, placement)


def play_policy(client: BattleshipClient, policy: ShotPolicy, *, poll_interval: float = 0.25) -> Observation:
    observation = client.observe()
    while observation.phase != "finished":
        if not observation.ready[observation.player]:
            placement = getattr(policy, "choose_placement", None)
            ships = placement(observation) if placement is not None else None
            observation = client.place(ships)
        elif observation.phase == "playing" and observation.turn == observation.player:
            observation = client.fire(policy.choose_shot(observation))
        else:
            observation = client.wait_for_change(observation.revision, poll_interval=poll_interval)
    return observation
