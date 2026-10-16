"""Public tests must not read or write private project evidence."""

import builtins
import io
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_FIXTURE = ROOT / "tests/fixtures/session22_golden.json"


def pytest_addoption(parser):
    parser.addoption("--run-private", action="store_true", default=False,
                     help="Include private integration checks; missing artifacts are failures, not skips")


def pytest_ignore_collect(collection_path, config):
    # Avoid importing optional Streamlit just to deselect private UI checks.
    if not config.getoption("--run-private") and collection_path == ROOT / "tests/test_dashboard_ui.py":
        return True
    return None


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-private"):
        return
    public, private = [], []
    for item in items:
        (private if item.get_closest_marker("private") else public).append(item)
    items[:] = public
    config.hook.pytest_deselected(items=private)


def protected_path(value):
    """Recognize project-private paths, without opening their contents."""
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False  # e.g. an already-open file descriptor
    path = Path(os.fsdecode(value)).resolve()
    return (path == PRIVATE_FIXTURE or
            any(path.is_relative_to(ROOT / directory) for directory in
                ("data", "outputs", "tests/fixtures/private")))


@pytest.fixture(autouse=True)
def enforce_public_file_boundary(request, monkeypatch):
    # Private tests explicitly opt in; their existing golden assertions stay intact.
    if request.node.get_closest_marker("private"):
        return

    def guarded(original):
        def open_public(file, *args, **kwargs):
            if protected_path(file):
                raise AssertionError("Public tests must not access private project artifacts")
            return original(file, *args, **kwargs)
        return open_public

    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(io, "open", guarded(io.open))
