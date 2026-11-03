"""Persistence and wire helpers for the local research UI."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:24]


def confined(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Path is outside the configured root")
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def publish(path, value, *, exclusive=False):
    """Durable atomic JSON publication, optionally without replacing a snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".ui-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def wire(value, key=""):
    """JS cannot exactly represent the benchmark's uint64 seeds."""
    if isinstance(value, dict):
        return {k: wire(v, k if 'seed' not in key.lower() else key) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v, key) for v in value]
    if isinstance(value, int) and not isinstance(value, bool) and ('seed' in key.lower() or abs(value) > 2**53 - 1):
        return str(value)
    return value
