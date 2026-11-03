import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher():
    spec = importlib.util.spec_from_file_location("launch_under_test", ROOT / "scripts" / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arguments,expected", [
    (["play", "--bot", "weights"], "requires --weights"),
    (["attach-bot", "--match", "unused", "--bot", "weights", "--weights", "missing.edits"], "No such file"),
    (["bots", "--opponent", "weights"], "requires --weights"),
    (["play", "--bot", "random", "--weights", "anything"], "only valid"),
])
def test_launcher_preflight_before_side_effects(launcher, monkeypatch, capsys, arguments, expected):
    def forbidden(*args, **kwargs):
        raise AssertionError("Launcher must not create directories, processes, or matches")
    monkeypatch.setattr(launcher, "Launcher", forbidden)
    monkeypatch.setattr(sys, "argv", ["launch.py", *arguments])
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 1
    assert expected in capsys.readouterr().err


def test_preflight_checks_invalid_opponent_before_launcher(launcher, monkeypatch, tmp_path, capsys):
    source = tmp_path / "valid.edits"
    source.write_text("b3[0]=1")
    monkeypatch.setattr(sys, "argv", ["launch.py", "bots", "--bot", "weights", "--weights", str(source),
                                      "--opponent", "weights", "--opponent-weights", "missing.npz"])
    monkeypatch.setattr(launcher, "Launcher", lambda *args: pytest.fail("premature launcher"))
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 1
    assert "missing.npz" in capsys.readouterr().err


def test_launcher_forwards_weight_path_and_seed(launcher, monkeypatch):
    # Construct without __init__, avoiding directories and external processes.
    instance = launcher.Launcher.__new__(launcher.Launcher)
    commands = []
    monkeypatch.setattr(instance, "spawn", lambda args, **kwargs: commands.append(args))
    monkeypatch.setattr(instance, "log", lambda _: None)
    instance.player(Path("seat.json"), bot="weights", seed=18, weights="path with spaces.npz")
    assert commands == [["battleship_client", "--session", "seat.json", "bot", "weights", "--seed", "18",
                          "--weights", "path with spaces.npz"]]


@pytest.mark.parametrize("target", ["play", "bot", "attach-bot", "bots"])
def test_make_forwards_paths_with_spaces(target):
    result = subprocess.run(["make", "-n", target, "MATCH=unused", "BOT=weights", "WEIGHTS=first path.edits",
                             "OPPONENT=weights", "OPPONENT_WEIGHTS=second path.npz"],
                            cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert '--weights "first path.edits"' in result.stdout
    if target == "bots":
        assert '--opponent-weights "second path.npz"' in result.stdout


def test_cli_preflight_before_session_read(tmp_path):
    result = subprocess.run([sys.executable, "-m", "battleship_client", "--session", str(tmp_path / "absent"),
                             "bot", "weights"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "requires --weights" in result.stderr
    assert "absent" not in result.stderr


def test_baseline_without_numpy_and_missing_extra_message():
    # A fresh interpreter with an import blocker simulates the base installation.
    program = '''
import importlib.abc
import sys
class NoNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("NumPy unavailable", name="numpy")
sys.meta_path.insert(0, NoNumPy())
import battleship_client
import battleship_client.cli
from battleship_client.policies import create_policy
create_policy("random", 1)
create_policy("hunt", 1)
assert "numpy" not in sys.modules
try:
    create_policy("weights", 1, "unused.edits")
except ImportError as error:
    assert "battleship-client[weights]" in str(error)
else:
    raise AssertionError("missing optional extra was ignored")
sys.argv = ["battleship", "bot", "weights", "--weights", "unused.edits"]
try:
    battleship_client.cli.main()
except SystemExit as error:
    assert error.code == 1
'''
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "battleship-client[weights]" in result.stderr


def test_packaged_client_declares_optional_numpy():
    from importlib.metadata import requires
    requirements = requires("battleship-client")
    numpy = [item for item in requirements if item.startswith("numpy")]
    assert len(numpy) == 1 and 'extra == "weights"' in numpy[0]


@pytest.mark.parametrize("arguments", [
    ["play", "--placement-weights", "missing.npz"],
    ["attach-bot", "--match", "unused", "--placement-weights", "missing.npz"],
    ["bots", "--opponent-placement-weights", "missing.npz"],
    ["play", "--opponent-placement-weights", "missing.npz"],
])
def test_placement_preflight_before_launcher(launcher, monkeypatch, capsys, arguments):
    monkeypatch.setattr(launcher, "Launcher", lambda *args: pytest.fail("premature launcher"))
    monkeypatch.setattr(sys, "argv", ["launch.py", *arguments])
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 1
    assert "missing.npz" in capsys.readouterr().err or "--opponent-placement-weights" in arguments


def test_launcher_forwards_placement_path(launcher, monkeypatch):
    instance = launcher.Launcher.__new__(launcher.Launcher)
    commands = []
    monkeypatch.setattr(instance, "spawn", lambda args, **kwargs: commands.append(args))
    monkeypatch.setattr(instance, "log", lambda _: None)
    instance.player(Path("seat.json"), bot="hunt", placement_weights="placement folder/model.npz")
    assert commands[0][-2:] == ["--placement-weights", "placement folder/model.npz"]
