"""Tracking rejects incomplete reports, misleading manifests and changed artifacts."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from deep_learning.run_behavioral import metrics
from tracking.common import MODEL_NAMES, artifact_record, git_state, tracking_uri, verify_bytes
from tracking.register_session20 import REPORT_FILES, validate_reports
from tracking.verify_session21 import verify_manifest


def report_fixture(directory):
    for name in REPORT_FILES:
        (directory / name).write_text("{}", encoding="utf-8")
    config = {"feature_groups": {name: ["TransactionAmt"] for name in MODEL_NAMES}, "test_access": False}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    targets = pd.DataFrame({"TransactionID": [10, 11, 12, 13], "TransactionDT": [20, 21, 22, 23],
                            "isFraud": [0, 1, 0, 1]})
    targets.to_csv(directory / "evaluation_targets.csv", index=False)
    pd.DataFrame({"TransactionID": [1, 2], "TransactionDT": [1, 2],
                  "isFraud": [0, 1]}).to_csv(directory / "training_targets.csv", index=False)
    predictions = targets.copy()
    rows = []
    for name in MODEL_NAMES:
        (directory / f"{name}.cbm").write_bytes(b"presence-only fixture; never loaded")
        predictions[name] = [.1, .8, .2, .9]
        rows.append({"model": name, "training_rows": 2, "val_examples": 4, "features": "TransactionAmt",
                     **metrics(targets.isFraud.to_numpy(), predictions[name].to_numpy(),
                               targets.TransactionID.to_numpy())})
    predictions.to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(directory / "model_comparison.csv", index=False)


class TrackingTests(unittest.TestCase):
    def test_artifact_verification_rejects_changed_or_missing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.cbm"
            path.write_bytes(b"original")
            record = artifact_record(path, "model/model.cbm")
            verify_bytes(path, record)
            path.write_bytes(b"modified")  # Same length; checking size alone is insufficient.
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                verify_bytes(path, record)
            with self.assertRaisesRegex(ValueError, "Missing artifact"):
                verify_bytes(path.with_name("missing.cbm"), record)

    def test_report_preflight_requires_every_model_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report_fixture(directory)
            validate_reports(directory)
            (directory / "AllFive.cbm").rename(directory / "saved.cbm")
            with self.assertRaisesRegex(FileNotFoundError, "AllFive.cbm"):
                validate_reports(directory)

    def test_report_preflight_rejects_metric_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report_fixture(directory)
            table = pd.read_csv(directory / "model_comparison.csv")
            table.loc[0, "average_precision"] = .123
            table.to_csv(directory / "model_comparison.csv", index=False)
            with self.assertRaises(AssertionError):
                validate_reports(directory)

    def test_report_preflight_rejects_overlapping_training_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report_fixture(directory)
            train = pd.read_csv(directory / "training_targets.csv")
            train.loc[1, "TransactionDT"] = 20
            train.to_csv(directory / "training_targets.csv", index=False)
            with self.assertRaisesRegex(ValueError, "strictly precede"):
                validate_reports(directory)

    def test_manifest_rejects_partial_duplicate_and_wrong_scope_runs(self):
        valid = {"status": "FINISHED", "test_access": False,
                 "runs": [{"model": name, "run_id": str(i)} for i, name in enumerate(MODEL_NAMES)]}
        verify_manifest(valid)
        variants = []
        partial = copy.deepcopy(valid)
        partial["runs"].pop()
        variants.append(partial)
        duplicate = copy.deepcopy(valid)
        duplicate["runs"][1]["run_id"] = "0"
        variants.extend([duplicate, {**valid, "status": "FAILED"}, {**valid, "test_access": True}])
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                verify_manifest(variant)

    def test_unborn_git_head_is_not_claimed_as_a_training_revision(self):
        replies = [subprocess.CompletedProcess([], 128, "HEAD\n", "no commits"),
                   subprocess.CompletedProcess([], 0, "?? tracking/\n", "")]
        with patch("tracking.common.subprocess.run", side_effect=replies):
            state = git_state()
        self.assertIsNone(state["registration_commit"])
        self.assertIsNone(state["training_commit"])
        self.assertTrue(state["dirty"])

    def test_sqlite_uri_preserves_absolute_workspace_path_with_spaces(self):
        directory = Path("tracking with spaces").resolve()
        self.assertEqual(tracking_uri(directory), "sqlite:///" + (directory / "mlflow.db").as_posix())


if __name__ == "__main__":
    unittest.main()
