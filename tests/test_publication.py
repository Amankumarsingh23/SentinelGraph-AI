"""Publication safety boundaries, entirely synthetic and artifact free."""

from pathlib import Path
import tempfile

import pytest

from scripts.check_public_snapshot import inspect_snapshot, public_path


def test_private_paths_and_fitted_artifacts_cannot_be_allowlisted():
    with tempfile.TemporaryDirectory() as temporary:
        for relative in ("data/train.csv", "outputs/report.json", "../escape.py",
                         "tests/fixtures/session22_golden.json", ".env", "model.cbm",
                         "publication/verification.local.json", "practice.ipynb"):
            with pytest.raises(ValueError):
                public_path(relative, Path(temporary))


def test_candidate_scan_rejects_literal_golden_vectors():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "publication").mkdir()
        (root / "publication/source_allowlist.txt").write_text("example.json\n", encoding="utf-8")
        (root / "example.json").write_text('{"vectors": []}', encoding="utf-8")
        assert inspect_snapshot(root)["status"] == "FAIL"


def test_public_guard_rejects_private_paths_before_opening():
    # Existing file contents are never opened, even in an authorized workspace.
    root = Path(__file__).resolve().parents[1]
    for path in (root / "outputs/does-not-exist.json", root / "data/does-not-exist.csv",
                 root / "tests/fixtures/session22_golden.json"):
        with pytest.raises(AssertionError, match="Public tests"):
            path.open("rb")
