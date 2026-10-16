"""Release integrity must reject changed bytes, paths, scopes and provenance."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import pytest

from fastapi.testclient import TestClient

from api.main import create_app
from api.model import EXPECTED_MODEL_SHA256
from integration.release import (
    ARTIFACTS, PROTOCOL, compare_mlflow_record, contract_metadata,
    file_spec, source_paths, validate_protocol, verify_integrity, verify_specs,
)
from integration.verify_release import verify_golden_workflow


class ReleaseTests(unittest.TestCase):
    def test_modified_bytes_or_missing_files_fail_without_rewriting_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "model.cbm"
            path.write_bytes(b"original")
            specs = {"model": file_spec(path, root)}
            frozen = copy.deepcopy(specs)
            verify_specs(specs, root)
            path.write_bytes(b"modified")  # same size
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_specs(specs, root)
            path.rename(root / "moved.cbm")
            with self.assertRaisesRegex(ValueError, "Missing"):
                verify_specs(specs, root)
            self.assertEqual(specs, frozen)

    def test_absolute_or_traversing_manifest_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for bad in ["../outside.cbm", str(root / "absolute.cbm")]:
                with self.subTest(path=bad), self.assertRaisesRegex(ValueError, "escapes"):
                    verify_specs({"model": {"path": bad, "sha256": "0" * 64, "bytes": 0}}, root)

    def test_source_inventory_tracks_release_code_but_ignores_generated_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "api").mkdir()
            (root / "outputs").mkdir()
            (root / "api/main.py").write_text("pass")
            (root / "outputs/generated.py").write_text("pass")
            self.assertEqual(source_paths(root), [root / "api/main.py"])

    def test_production_claims_decision_changes_or_contract_changes_are_rejected(self):
        base = {"schema_version": 1, "stage": "development", "evaluation_status": "development_candidate_not_final_selection",
                "reserved_test_evaluated": False, "test_access": False, "final_test_ready": False,
                "training_reexecuted": False, "decision_policy": None, "score_type": "uncalibrated_model_score",
                "model_sha256": EXPECTED_MODEL_SHA256, "feature_contract": contract_metadata()}
        for key, value in [("stage", "production"), ("reserved_test_evaluated", True), ("test_access", True),
                           ("final_test_ready", True), ("decision_policy", {"threshold": .5}), ("model_sha256", "wrong")]:
            with self.subTest(field=key), self.assertRaises(ValueError):
                verify_integrity({**base, key: value})
        bad = copy.deepcopy(base)
        bad["feature_contract"]["columns"].reverse()
        with self.assertRaises(ValueError):
            verify_integrity(bad)

    def test_missing_required_artifact_inventory_is_rejected(self):
        manifest = {"schema_version": 1, "stage": "development", "evaluation_status": "development_candidate_not_final_selection",
                    "reserved_test_evaluated": False, "test_access": False, "final_test_ready": False,
                    "training_reexecuted": False, "decision_policy": None, "score_type": "uncalibrated_model_score",
                    "model_sha256": EXPECTED_MODEL_SHA256, "feature_contract": contract_metadata(), "artifacts": {}}
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_integrity(manifest)

    def test_protocol_stays_a_development_plan_without_test_optimization(self):
        protocol = json.loads(PROTOCOL.read_text())
        validate_protocol(protocol)
        for field in ["execute_reserved_test", "reserved_test_evaluated", "final_model_selection_approved"]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_protocol({**protocol, field: True})
        protocol["cost_diagnostic"]["optimize_on_test"] = True
        with self.assertRaises(ValueError):
            validate_protocol(protocol)

    def test_model_must_match_earlier_mlflow_record_and_registered_input_files(self):
        specs = {name: {"path": path, "sha256": EXPECTED_MODEL_SHA256 if name == "model" else "a" * 64, "bytes": 4}
                 for name, path in ARTIFACTS.items()}
        all_five = {"model": "AllFive", "run_id": "3", "artifacts": [
            {"artifact_path": path, **{key: specs[name][key] for key in ["sha256", "bytes"]}}
            for name, path in {"model": "model/AllFive.cbm", "predictions": "reports/predictions.csv",
                               "training_configuration": "reports/config.json"}.items()]}
        registration = {"status": "FINISHED", "test_access": False, "registration_id": "prior-registration",
                        "experiment_id": "1", "tracking_uri": "sqlite:///local.db", "input_files": {},
                        "runs": [{"model": name, "run_id": str(i)} for i, name in enumerate(
                            ["AmountOnly", "AmountPlusActivity", "AmountPlusRelationships"])] + [all_five]}
        for name in ["training_records", "validation_records", "training_graph_features", "validation_graph_features"]:
            registration["input_files"][specs[name]["path"].replace("/", "\\")] = {"sha256": specs[name]["sha256"], "bytes": 4}
        result = compare_mlflow_record(registration, specs)
        self.assertEqual(result["run_id"], "3")
        bad = copy.deepcopy(specs)
        bad["model"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "MLflow registration"):
            compare_mlflow_record(registration, bad)
        bad = copy.deepcopy(specs)
        bad["validation_graph_features"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "registration-time"):
            compare_mlflow_record(registration, bad)

    @pytest.mark.private
    def test_six_golden_requests_agree_across_http_dashboard_shap_and_monitoring(self):
        with TestClient(create_app()) as client:
            report = verify_golden_workflow(client)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["http_requests"], 6)
        self.assertTrue(report["label_input_rejected"])
