"""Fixed shooting and placement networks: safe edits, artifacts, and float64 inference.

This module requires the optional ``battleship-client[weights]`` extra. Nothing
in the baseline SDK imports it eagerly. All network arithmetic is float64.
"""

import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    raise ImportError(
        'Weight support requires NumPy; install "battleship-client[weights]" '
        'or run python -m pip install "./packages/client[weights]" from source.'
    ) from error

from battleship_protocol import (
    NETWORK_VERSION, OBSERVATION_VERSION, Coordinate, FLEET, Observation, ShipPlacement,
)

ARTIFACT_VERSION = "1.0"
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_STATEMENTS = 100_000
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
VALUE_LIMIT = 1_000_000
TENSOR_SHAPES = MappingProxyType({
    "W1": (405, 128), "b1": (128,), "W2": (128, 128),
    "b2": (128,), "W3": (128, 100), "b3": (100,),
})
_VERSIONS = {
    "artifact_version": ARTIFACT_VERSION,
    "network_version": NETWORK_VERSION,
    "observation_version": OBSERVATION_VERSION,
}


PLACEMENT_TENSOR_SHAPES = MappingProxyType({
    "W1": (505, 128), "b1": (128,), "W2": (128, 128),
    "b2": (128,), "W3": (128, 200), "b3": (200,),
})


@dataclass(frozen=True)
class NetworkSpec:
    shapes: Mapping[str, tuple[int, ...]]
    versions: Mapping[str, str]


NETWORK_SPECS = MappingProxyType({
    "shooting": NetworkSpec(TENSOR_SHAPES, MappingProxyType(_VERSIONS)),
    "placement": NetworkSpec(PLACEMENT_TENSOR_SHAPES, MappingProxyType({
        "artifact_version": "1.1", "role": "placement",
        "network_version": "placement-1.0", "observation_version": OBSERVATION_VERSION,
    })),
})


def network_spec(role: str) -> NetworkSpec:
    try:
        return NETWORK_SPECS[role]
    except KeyError:
        raise ValueError(f"Unknown network role: {role}") from None


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _content_hash(tensors: Mapping[str, np.ndarray], role: str = "shooting") -> str:
    digest = hashlib.sha256()
    spec = network_spec(role)
    digest.update(_json_bytes(dict(spec.versions)) + b"\n")
    for name, shape in spec.shapes.items():
        digest.update(_json_bytes([name, list(shape), "<f8"]) + b"\n")
        digest.update(tensors[name].astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class NetworkWeights:
    """Validated, owned, immutable tensors in contract order."""

    tensors: Mapping[str, np.ndarray]
    role: str = "shooting"

    def __post_init__(self):
        spec = network_spec(self.role)
        if set(self.tensors) != set(spec.shapes):
            raise ValueError("Weights must contain exactly W1, b1, W2, b2, W3, b3")
        owned = {}
        for name, shape in spec.shapes.items():
            array = self.tensors[name]
            if (not isinstance(array, np.ndarray) or array.dtype.kind != "f"
                    or array.dtype.itemsize != 8):
                raise ValueError(f"{name} must have float64 dtype")
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.isfinite(array).all() or (np.abs(array) > VALUE_LIMIT).any():
                raise ValueError(f"{name} must be finite and within +/-{VALUE_LIMIT}")
            # An immutable bytes owner prevents callers from re-enabling writes.
            data = array.astype("<f8", copy=False).tobytes()
            owned[name] = np.frombuffer(data, dtype="<f8").reshape(shape)
        object.__setattr__(self, "tensors", MappingProxyType(owned))

    @property
    def content_hash(self) -> str:
        return _content_hash(self.tensors, self.role)


class EditsError(ValueError):
    """An import failed; no partially edited weights were published."""


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    line: int
    column: int


_TOKEN = re.compile(
    r"(?P<space>[ \t\r]+)|(?P<comment>\#[^\n]*)|(?P<newline>\n)|"
    r"(?P<number>(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+))|"
    r"(?P<name>[A-Za-z_][A-Za-z_0-9]*)|(?P<operator>\+=|-=|\*=|=)|"
    r"(?P<punct>[\[\],:+-])"
)


def _tokens(text: str, first_line: int):
    position, line, column = 0, first_line, 1
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise EditsError(f"line {line}, column {column}: unexpected character")
        kind, value = match.lastgroup, match.group()
        if kind not in ("space", "comment"):
            yield _Token(kind, value, line, column)
        position = match.end()
        if kind == "newline":
            line, column = line + 1, 1
        else:
            column += len(value)
    yield _Token("eof", "", line, column)


def _extract_edits(text: str) -> tuple[str, int]:
    # Recognize Markdown fences so edits-looking text inside another block is
    # never mistaken for an import. A final unclosed edits block is an error.
    lines = text.splitlines(keepends=True)
    fence = None
    start = None
    last = None
    saw_fence = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        opening = re.fullmatch(r"(`{3,}|~{3,})([^`~]*)", stripped)
        if fence is None:
            if opening:
                saw_fence = True
                fence = opening[1]
                start = index + 1 if opening[2].strip() == "edits" else None
        elif re.fullmatch(re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*", stripped):
            if start is not None:
                last = ("".join(lines[start:index]), start + 1)
            fence, start = None, None
    if start is not None:
        raise EditsError(f"line {start}: unclosed edits block")
    if last is not None:
        return last
    if saw_fence:
        raise EditsError("line 1: response contains no closed edits block")
    return text, 1


class _Parser:
    def __init__(self, text: str, first_line: int, role: str):
        self.role = role
        self.shapes = network_spec(role).shapes
        self.tokens = iter(_tokens(text, first_line))
        self.token = next(self.tokens)

    def fail(self, message: str, token: _Token | None = None):
        token = token or self.token
        raise EditsError(f"line {token.line}, column {token.column}: {message}")

    def advance(self):
        token = self.token
        self.token = next(self.tokens)
        return token

    def take(self, value: str):
        if self.token.value != value:
            self.fail(f"expected {value!r}")
        return self.advance()

    def newlines(self):
        while self.token.kind == "newline":
            self.advance()

    def integer(self, *, step=False) -> int:
        if self.token.kind != "number" or not self.token.value.isdigit():
            self.fail("expected a nonnegative integer index")
        digits = self.advance().value.lstrip("0") or "0"
        if len(digits) > 10:
            if step:
                # Any step larger than the biggest axis selects the same cells.
                return max(max(shape) for shape in self.shapes.values()) + 1
            self.fail("index out of bounds")
        return int(digits)

    def axis(self, size: int):
        start = None if self.token.value == ":" else self.integer()
        if self.token.value != ":":
            if start >= size:
                self.fail("index out of bounds")
            return start
        self.advance()
        stop = None if self.token.value in (":", ",", "]") else self.integer()
        step = 1
        if self.token.value == ":":
            self.advance()
            if self.token.value not in (",", "]"):
                step = self.integer(step=True)
        if step <= 0 or any(bound is not None and bound > size for bound in (start, stop)):
            self.fail("slice bounds must be in bounds and step positive")
        selection = slice(start, stop, step)
        if not range(*selection.indices(size)):
            self.fail("selection must not be empty")
        return selection

    def value(self, depth=0):
        if self.token.value == "[":
            if depth >= 2:
                self.fail("lists may have at most two dimensions")
            self.advance()
            self.newlines()
            values = []
            if self.token.value != "]":
                while True:
                    values.append(self.value(depth + 1))
                    self.newlines()
                    if self.token.value != ",":
                        break
                    self.advance()
                    self.newlines()
                    if self.token.value == "]":
                        break
            self.take("]")
            return values
        sign = 1
        if self.token.value in ("+", "-"):
            sign = -1 if self.advance().value == "-" else 1
        if self.token.kind != "number":
            self.fail("expected a plain decimal number or list")
        token = self.advance()
        # Check the decimal literal before float64 rounding can hide a value
        # just outside the contract's bounds.
        literal = Decimal(token.value)
        if literal.copy_abs() > VALUE_LIMIT:
            self.fail("number must be finite and within +/-1000000", token)
        return sign * float(literal)

    def parse(self) -> NetworkWeights:
        tensors = {name: np.zeros(shape, dtype=np.float64) for name, shape in self.shapes.items()}
        count = 0
        self.newlines()
        while self.token.kind != "eof":
            origin = self.token
            count += 1
            if count > MAX_STATEMENTS:
                self.fail("too many statements")
            if self.token.kind != "name" or self.token.value not in tensors:
                self.fail("expected one of W1, b1, W2, b2, W3, b3")
            name = self.advance().value
            self.take("[")
            indexes = []
            for axis, size in enumerate(self.shapes[name]):
                if axis:
                    self.take(",")
                indexes.append(self.axis(size))
            self.take("]")
            if self.token.kind != "operator":
                self.fail("expected =, +=, -=, or *=")
            operation = self.advance().value
            value = self.value()
            if self.token.kind not in ("newline", "eof"):
                self.fail("only one statement per line is permitted")
            selected = tensors[name][tuple(indexes)]
            try:
                array = np.asarray(value, dtype=np.float64)
            except ValueError:
                self.fail("lists must be rectangular", origin)
            if array.ndim and array.shape != np.shape(selected):
                self.fail(f"list shape must match selection {np.shape(selected)}", origin)
            with np.errstate(over="ignore", invalid="ignore"):
                if operation == "=":
                    result = array
                elif operation == "+=":
                    result = selected + array
                elif operation == "-=":
                    result = selected - array
                else:
                    result = selected * array
            if not np.isfinite(result).all() or (np.abs(result) > VALUE_LIMIT).any():
                self.fail("operation result must be finite and within +/-1000000", origin)
            tensors[name][tuple(indexes)] = result
            self.newlines()
        return NetworkWeights(tensors, self.role)


def parse_edits(text: str, *, role: str = "shooting") -> NetworkWeights:
    """Import from zero, rejecting the entire import on the first invalid edit."""
    if len(text) > MAX_TEXT_BYTES or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise EditsError("line 1: edits input exceeds 8 MiB")
    source, first_line = _extract_edits(text)
    return _Parser(source, first_line, role).parse()


def save_weights(path: str | Path, weights: NetworkWeights):
    """Publish a complete artifact atomically, without replacing any destination."""
    path = Path(path)
    manifest = {**network_spec(weights.role).versions, "content_hash": weights.content_hash}
    fd, temporary = tempfile.mkstemp(prefix=".weights-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            np.savez(output, **weights.tensors, manifest=np.array(_json_bytes(manifest).decode()))
            output.flush()
            os.fsync(output.fileno())
        # Same-filesystem hard-link publication is atomic and fails if the
        # destination exists (including a symlink). No check-then-replace race.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _read_npy(data: bytes, spec: NetworkSpec = NETWORK_SPECS["shooting"]) -> np.ndarray:
    # Inspect the shape before np.load: a small NPY header can claim an enormous
    # array. All supported members are scalars or the six fixed tensor shapes.
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported NPY member version")
    if dtype.hasobject or shape not in ((), *spec.shapes.values()):
        raise ValueError("Invalid artifact array shape or object dtype")
    if dtype.itemsize > 16_384 or dtype.itemsize == 0:
        raise ValueError("Invalid artifact dtype size")
    count = 1
    for size in shape:
        count *= size
    expected = count * dtype.itemsize
    if expected > MAX_ARCHIVE_BYTES or len(data) - stream.tell() != expected:
        raise ValueError("Artifact array payload does not match its header")
    return np.load(io.BytesIO(data), allow_pickle=False)


def load_weights(path: str | Path, *, role: str = "shooting") -> NetworkWeights:
    """Validate archive structure, metadata, parameter bounds, and content hash."""
    spec = network_spec(role)
    try:
        with open(path, "rb") as source:
            if os.fstat(source.fileno()).st_size > MAX_ARCHIVE_BYTES:
                raise ValueError("Weight artifact exceeds 2 MiB")
            with zipfile.ZipFile(source) as archive:
                members = archive.infolist()
                expected = {f"{name}.npy" for name in (*spec.shapes, "manifest")}
                if len(members) != len(expected) or {item.filename for item in members} != expected:
                    raise ValueError("Unexpected, duplicate, or missing artifact members")
                if sum(item.file_size for item in members) > MAX_ARCHIVE_BYTES:
                    raise ValueError("Expanded weight artifact exceeds 2 MiB")
                arrays = {}
                for item in members:
                    with archive.open(item) as member:
                        # Bound the actual decompressor read too: untrusted ZIP
                        # metadata alone is insufficient to cap allocations.
                        data = member.read(item.file_size + 1)
                    arrays[item.filename[:-4]] = _read_npy(data, spec)
        metadata = arrays.pop("manifest")
        if metadata.shape != () or metadata.dtype.kind != "U":
            raise ValueError("Artifact manifest must be a JSON string scalar")
        manifest = json.loads(str(metadata))
        if not isinstance(manifest, dict) or set(manifest) != {*spec.versions, "content_hash"}:
            raise ValueError("Invalid artifact manifest fields")
        if any(manifest[key] != value for key, value in spec.versions.items()):
            raise ValueError("Unsupported artifact, network, or observation version")
        weights = NetworkWeights(arrays, role)
        if manifest["content_hash"] != weights.content_hash:
            raise ValueError("Weight artifact content hash mismatch")
        return weights
    except (zipfile.BadZipFile, zlib.error, EOFError, KeyError, RuntimeError, NotImplementedError,
            OverflowError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("Invalid weight artifact") from error


def load_weight_source(path: str | Path, *, role: str = "shooting") -> NetworkWeights:
    """Load .npz strictly as an artifact; all other paths are UTF-8 edits text."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return load_weights(path, role=role)
    with path.open("rb") as source:
        data = source.read(MAX_TEXT_BYTES + 1)
    if len(data) > MAX_TEXT_BYTES:
        raise EditsError("line 1: edits input exceeds 8 MiB")
    return parse_edits(data.decode("utf-8"), role=role)


def encode_observation(observation: Observation) -> np.ndarray:
    """Encode only public target knowledge, as 405 float64 binary features."""
    if observation.observation_version != OBSERVATION_VERSION:
        raise ValueError("Unsupported observation version")
    channels = {"unknown": 0, "miss": 1, "hit": 2, "sunk": 3}
    encoded = np.zeros(405, dtype=np.float64)
    if (len(observation.target_board) != 10
            or any(len(row) != 10 for row in observation.target_board)):
        raise ValueError("Target board must be 10 by 10")
    for row, cells in enumerate(observation.target_board):
        for column, state in enumerate(cells):
            if state not in channels:
                raise ValueError("Unknown target cell state")
            encoded[(row * 10 + column) * 4 + channels[state]] = 1
    for index, name in enumerate(FLEET):
        encoded[400 + index] = name in observation.remaining_enemy_ships
    return encoded


class WeightPlayer:
    """Fixed-weight shooting policy; placement is delegated to play_policy."""

    def __init__(self, weights: NetworkWeights, seed: int | None = None):
        if weights.role != "shooting":
            raise ValueError("Shooting player requires shooting weights")
        self.weights = weights
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _require_active(observation: Observation):
        if (observation.phase != "playing" or observation.turn != observation.player
                or observation.result is not None):
            raise ValueError("Weight player requires an active shooting turn")

    def logits(self, observation: Observation) -> np.ndarray:
        self._require_active(observation)
        x = encode_observation(observation)
        w = self.weights.tensors
        with np.errstate(over="ignore", invalid="ignore"):
            h1 = np.tanh(x @ w["W1"] + w["b1"])
            h2 = np.tanh(h1 @ w["W2"] + w["b2"])
            logits = h2 @ w["W3"] + w["b3"]
        if not all(np.isfinite(value).all() for value in (h1, h2, logits)):
            raise ValueError("Nonfinite network inference result")
        return logits

    def shot_probabilities(self, observation: Observation) -> np.ndarray:
        """Return 100 probabilities, with exactly zero mass on illegal cells."""
        self._require_active(observation)
        mask = np.asarray(observation.legal_shot_mask, dtype=bool)
        if mask.shape != (10, 10) or not mask.any():
            raise ValueError("Active shooting turn has no valid legal shot mask")
        legal = mask.reshape(100)
        logits = self.logits(observation)
        if logits.shape != (100,) or not np.isfinite(logits).all():
            raise ValueError("Invalid network logits")
        legal_logits = logits[legal]
        with np.errstate(under="ignore"):
            probabilities = np.exp(legal_logits - legal_logits.max())
        probabilities /= probabilities.sum()
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite shot probabilities")
        result = np.zeros(100, dtype=np.float64)
        result[legal] = probabilities
        return result

    def choose_shot(self, observation: Observation) -> Coordinate:
        probabilities = self.shot_probabilities(observation)
        index = int(self.rng.choice(100, p=probabilities))
        row, column = divmod(index, 10)
        return Coordinate(row=row, column=column)


def placement_cells(ship: ShipPlacement) -> tuple[tuple[int, int], ...]:
    """Client-only geometry; the server remains the authoritative validator."""
    dr, dc = (0, 1) if ship.orientation == "horizontal" else (1, 0)
    return tuple((ship.start.row + dr * i, ship.start.column + dc * i)
                 for i in range(FLEET[ship.name]))


def validate_placement_fleet(ships: list[ShipPlacement], *, complete: bool = False):
    names = [ship.name for ship in ships]
    if len(set(names)) != len(names) or (complete and set(names) != set(FLEET)):
        raise ValueError("Fleet must contain each ship exactly once")
    occupied = set()
    for ship in ships:
        cells = placement_cells(ship)
        if any(not (0 <= r < 10 and 0 <= c < 10) for r, c in cells):
            raise ValueError("All ships must fit on the board")
        if occupied.intersection(cells):
            raise ValueError("Ships must not overlap")
        occupied.update(cells)
    return occupied


def encode_placement(ships: list[ShipPlacement], current_ship: str) -> np.ndarray:
    """Encode an ordered partial fleet and the next ship, without match state."""
    order = list(FLEET)
    if (len(ships) >= len(order) or [ship.name for ship in ships] != order[:len(ships)]
            or current_ship != order[len(ships)]):
        raise ValueError("Placement must follow the fixed fleet order")
    validate_placement_fleet(ships)
    encoded = np.zeros(505, dtype=np.float64)
    for index, ship in enumerate(ships):
        for row, column in placement_cells(ship):
            encoded[(row * 10 + column) * 5 + index] = 1
    encoded[500 + len(ships)] = 1
    return encoded


def placement_mask(ships: list[ShipPlacement], current_ship: str) -> np.ndarray:
    encode_placement(ships, current_ship)  # Validate the prefix and ship identity.
    occupied = validate_placement_fleet(ships)
    mask = np.zeros(200, dtype=bool)
    length = FLEET[current_ship]
    for row in range(10):
        for column in range(10):
            for orientation, (dr, dc) in enumerate(((0, 1), (1, 0))):
                cells = [(row + dr * i, column + dc * i) for i in range(length)]
                mask[2 * (row * 10 + column) + orientation] = (
                    all(r < 10 and c < 10 for r, c in cells)
                    and not occupied.intersection(cells))
    return mask


class PlacementWeightPlayer:
    """Construct a complete fleet using five masked placement decisions."""

    def __init__(self, weights: NetworkWeights, seed: int | None = None):
        if weights.role != "placement":
            raise ValueError("Placement player requires placement weights")
        self.weights = weights
        self.rng = np.random.default_rng(seed)

    def logits(self, ships: list[ShipPlacement], current_ship: str) -> np.ndarray:
        x = encode_placement(ships, current_ship)
        w = self.weights.tensors
        with np.errstate(over="ignore", invalid="ignore"):
            h1 = np.tanh(x @ w["W1"] + w["b1"])
            h2 = np.tanh(h1 @ w["W2"] + w["b2"])
            logits = h2 @ w["W3"] + w["b3"]
        if not all(np.isfinite(value).all() for value in (h1, h2, logits)):
            raise ValueError("Nonfinite placement inference result")
        return logits

    def placement_probabilities(self, ships: list[ShipPlacement], current_ship: str) -> np.ndarray:
        mask = placement_mask(ships, current_ship)
        if not mask.any():
            raise ValueError("No legal placement available")
        logits = self.logits(ships, current_ship)
        if logits.shape != (200,) or not np.isfinite(logits).all():
            raise ValueError("Invalid placement logits")
        legal_logits = logits[mask]
        with np.errstate(under="ignore"):
            probabilities = np.exp(legal_logits - legal_logits.max())
        probabilities /= probabilities.sum()
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite placement probabilities")
        result = np.zeros(200, dtype=np.float64)
        result[mask] = probabilities
        return result

    def choose_placement(self, observation: Observation) -> list[ShipPlacement]:
        if (observation.phase not in ("waiting", "placement")
                or observation.ready[observation.player] or observation.result is not None):
            raise ValueError("Placement player requires an uncommitted seat")
        ships = []
        for name in FLEET:
            probabilities = self.placement_probabilities(ships, name)
            index = int(self.rng.choice(200, p=probabilities))
            cell, orientation = divmod(index, 2)
            row, column = divmod(cell, 10)
            ships.append(ShipPlacement(name=name, start=Coordinate(row=row, column=column),
                                       orientation="horizontal" if orientation == 0 else "vertical"))
        validate_placement_fleet(ships, complete=True)
        return ships
