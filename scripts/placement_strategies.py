"""Trusted placement generators shared by the shooting and defense benchmarks."""

from battleship_protocol import Coordinate, FLEET, ShipPlacement
from battleship_server.engine import placement_cells, random_fleet, validate_fleet

FAMILIES = ("random", "edge", "center", "touching")
MAX_FLEET_ATTEMPTS = 1000
FAMILY_DESCRIPTIONS = {
    "random": "Existing sequential random_fleet distribution; not uniform over complete fleets",
    "edge": "Every occupied cell has row or column in {0,1,8,9}",
    "center": "Every occupied cell has row and column in {2,3,4,5,6,7}",
    "touching": "Each ship after the first orthogonally touches an earlier ship",
}


def fleet_placements(fleet):
    return [ShipPlacement(name=name,
                          start=Coordinate(row=min(r for r, _ in cells),
                                           column=min(c for _, c in cells)),
                          orientation="horizontal" if len({r for r, _ in cells}) == 1
                          else "vertical")
            for name, cells in fleet.items()]


def all_placements(name):
    length = FLEET[name]
    return [ShipPlacement(name=name, start=Coordinate(row=r, column=c), orientation=orientation)
            for orientation, dr, dc in (("horizontal", 0, 1), ("vertical", 1, 0))
            for r in range(10 - dr * (length - 1))
            for c in range(10 - dc * (length - 1))]


PLACEMENTS = {name: [(ship, placement_cells(ship)) for ship in all_placements(name)]
              for name in FLEET}


def draw_fleet(family, rng, *, max_attempts=MAX_FLEET_ATTEMPTS):
    if family == "random":
        return fleet_placements(random_fleet(rng))
    if family not in FAMILIES:
        raise ValueError("Unknown placement family")
    for _ in range(max_attempts):
        occupied = set()
        ships = []
        for name in FLEET:
            choices = []
            for ship, cells in PLACEMENTS[name]:
                if occupied.intersection(cells):
                    continue
                if family == "edge" and not all(r < 2 or r >= 8 or c < 2 or c >= 8
                                                for r, c in cells):
                    continue
                if family == "center" and not all(2 <= r < 8 and 2 <= c < 8 for r, c in cells):
                    continue
                if family == "touching" and occupied and not any(
                    (r + dr, c + dc) in occupied for r, c in cells
                    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))):
                    continue
                choices.append((ship, cells))
            if not choices:
                break
            ship, cells = rng.choice(choices)
            ships.append(ship)
            occupied.update(cells)
        if len(ships) == len(FLEET):
            validate_fleet(ships)
            return ships
    raise ValueError(f"Placement generation exhausted retries for {family}")
