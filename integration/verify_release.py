"""One development release workflow: integrity, tests, HTTP and artifact audits."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import numpy as np
import pandas as pd

from api.model import CONTRACT_VERSION, EXPECTED_MODEL_SHA256, FEATURE_COLUMNS, load_model
from dashboard.evidence import deterministic_summary, file_hash, load_evidence_table, make_evidence_record
from dashboard.shap_evidence import checked_shap, load_shap_export
from dashboard.verify_session24 import audit_evidence
from integration.release import MANIFEST, ROOT, compare_mlflow_record, verify_integrity, write_json
from monitoring.check_drift import prepare_features as monitoring_features
from monitoring.verify_session23 import verify_outputs


def verify_registered_artifacts(manifest):
    """Check actual retrieved MLflow bytes, not just a matching local JSON claim."""
    from mlflow.tracking import MlflowClient
    spec = manifest["artifacts"]["mlflow_registration"]
    registration = json.loads((ROOT / spec["path"]).read_text())
    provenance = compare_mlflow_record(registration, manifest["artifacts"])
    client = MlflowClient(tracking_uri=provenance["tracking_uri"])
    run = client.get_run(provenance["run_id"])
    if (run.info.status != "FINISHED" or run.info.experiment_id != provenance["experiment_id"]
            or run.data.tags.get("registration_id") != provenance["registration_id"]
            or run.data.tags.get("mlflow.runName") != "AllFive"
            or run.data.tags.get("training_reexecuted") != "false"):
        raise ValueError("Earlier MLflow run identity/status differs")
    with tempfile.TemporaryDirectory(prefix="sentinelgraph_release_mlflow_") as temporary:
        for name, location in provenance["compared_artifacts"].items():
            downloaded = Path(client.download_artifacts(run.info.run_id, location, dst_path=temporary))
            expected = manifest["artifacts"][name]
            if file_hash(downloaded) != expected["sha256"] or downloaded.stat().st_size != expected["bytes"]:
                raise ValueError(f"MLflow-stored {name} differs from the release")
    return {"status": "PASS", "run_id": run.info.run_id, "retrieved_artifacts": 3,
            "provenance_scope": provenance["scope"]}


@contextmanager
def running_api(logfile):
    """Start fresh release code on loopback, and stop only our own child process."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with Path(logfile).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10, trust_env=False) as client:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Fresh release API exited during startup; inspect api_server.log")
                    try:
                        response = client.get("/health")
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Fresh release API did not become ready")
                    time.sleep(.1)
                yield client
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def verify_golden_workflow(client):
    golden = json.loads((ROOT / "tests/fixtures/session22_golden.json").read_text())
    info_response = client.get("/model-info")
    info_response.raise_for_status()
    info = info_response.json()
    if (info["model_sha256"] != EXPECTED_MODEL_SHA256 or golden["model_sha256"] != EXPECTED_MODEL_SHA256
            or info["feature_contract_version"] != CONTRACT_VERSION or info["feature_columns"] != FEATURE_COLUMNS
            or info["decision"] != "not_assigned" or info["score_type"] != "uncalibrated_model_score"):
        raise ValueError("HTTP model metadata differs from frozen contract")
    table = load_evidence_table()
    payload, lookup = load_shap_export()
    model = load_model()
    batch = pd.DataFrame([row["raw_features"] for row in golden["vectors"]])
    scores = model.classifier.predict_proba(monitoring_features(batch), thread_count=2)[:, 1]
    checks = []
    for vector, monitored_score in zip(golden["vectors"], scores):
        raw = vector["raw_features"]
        request = {"transaction_id": vector["transaction_id"], "transaction_amount": raw["TransactionAmt"],
                   "graph_card_prior_degree": raw["graph_card_prior_degree"],
                   "graph_product_prior_degree": raw["graph_product_prior_degree"],
                   "graph_pair_prior_count": raw["graph_card_product_prior_count"],
                   "graph_card_distinct_products": raw["graph_card_distinct_products"]}
        response = client.post("/predict", json=request)
        response.raise_for_status()
        body = response.json()
        evidence = make_evidence_record(table, vector["transaction_id"])
        explanation = checked_shap(evidence, payload, lookup)
        if explanation is None:
            raise ValueError("Golden transaction lacks a matching verified SHAP record")
        for score in [body["risk_score"], evidence["model_score"], explanation["model_score"], monitored_score]:
            if not math.isclose(score, vector["expected_score"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("HTTP/dashboard/SHAP/monitor score differs from independent golden evidence")
        if (body["transaction_id"] != evidence["transaction_id"] or body["model_sha256"] != evidence["model_sha256"]
                or body["feature_contract_version"] != evidence["feature_contract_version"]
                or body["score_type"] != "uncalibrated_model_score" or body["decision"] != "not_assigned"
                or "isFraud" in evidence or "No fraud decision has been assigned" not in deterministic_summary(evidence)):
            raise ValueError("Cross-component identity, label isolation or decision contract differs")
        actual_inputs = {"TransactionAmt": evidence["transaction_amount"], **evidence["historical_features"]}
        np.testing.assert_allclose([actual_inputs[name] for name in FEATURE_COLUMNS],
                                   [raw[name] for name in FEATURE_COLUMNS], rtol=0, atol=0)
        checks.append({"transaction_id": vector["transaction_id"], "status": "PASS",
                       "expected_score": vector["expected_score"], "http_score": body["risk_score"]})
    invalid = client.post("/predict", json={**request, "isFraud": 1})
    if invalid.status_code != 422:
        raise ValueError("HTTP scoring accepted a fraud label")
    return {"status": "PASS", "golden_vectors": checks, "http_requests": len(checks),
            "dashboard_shap_monitoring_agree": True, "label_input_rejected": True}


def run_tests(directory):
    result = subprocess.run([sys.executable, "-B", "-m", "pytest", "tests", "--run-private", "-q", f"--junitxml={directory / 'test_results.xml'}"],
                            cwd=ROOT, capture_output=True, text=True, timeout=180, check=False)
    log = result.stdout + result.stderr
    (directory / "tests.log").write_text(log, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError("Release test suite failed; inspect tests.log")
    match = re.search(r"\b(\d+) passed(?:,|\s)", log)
    if match is None:
        raise RuntimeError("Could not confirm successful pytest count")
    cases = list(ET.parse(directory / "test_results.xml").getroot().iter("testcase"))
    if any(case.find("skipped") is not None for case in cases):
        raise RuntimeError("Release suite skipped required verification work")
    required = ["test_api", "test_temporal_integration", "test_dashboard_evidence", "test_dashboard_ui", "test_monitoring", "test_release"]
    for module in required:
        if not any(module + "." in case.attrib.get("classname", "") for case in cases):
            raise RuntimeError(f"Required integration test module did not execute: {module}")
    temporal_count = sum("test_temporal_integration." in case.attrib.get("classname", "") for case in cases)
    if temporal_count < 5:
        raise RuntimeError("Required independent-oracle/actual-generator temporal checks did not execute")
    return {"status": "PASS", "passed_tests": int(match.group(1)), "temporal_integration_tests": temporal_count,
            "required_modules": required, "log": "tests.log", "junit": "test_results.xml"}


def verify_release(manifest_path=MANIFEST, integrity_only=False):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    directory = manifest_path.parent / "verifications" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    report = {"release_id": manifest["release_id"], "manifest_sha256": file_hash(manifest_path),
              "verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "IN_PROGRESS", "checks": {},
              "scope": "development candidate; precomputed-feature scoring and investigation demonstration",
              "test_access": False, "reserved_test_evaluated": False, "training_reexecuted": False, "final_test_ready": False}
    try:
        print("Checking frozen artifact/source/environment fingerprints...", flush=True)
        report["checks"]["integrity"] = verify_integrity(manifest)
        if integrity_only:
            report["status"] = "PASS_INTEGRITY_ONLY"
        else:
            print("Retrieving earlier MLflow model/prediction/config bytes...", flush=True)
            report["checks"]["mlflow"] = verify_registered_artifacts(manifest)
            print("Running the complete unit, temporal and UI integration suite...", flush=True)
            report["checks"]["tests"] = run_tests(directory)
            print("Auditing all saved dashboard/SHAP records and monitoring scores...", flush=True)
            report["checks"]["dashboard"] = audit_evidence()
            report["checks"]["monitoring"] = verify_outputs(write_report=False)
            print("Starting fresh loopback API for six golden cross-component requests...", flush=True)
            with running_api(directory / "api_server.log") as client:
                report["checks"]["http_golden_workflow"] = verify_golden_workflow(client)
            # Tests/retrieval may generate logs, never modify frozen release inputs.
            report["checks"]["post_workflow_integrity"] = verify_integrity(manifest)
            report["development_checklist"] = {name: "PASS" for name in [
                "artifact_integrity", "feature_contract", "golden_http_predictions", "temporal_safety",
                "evidence_integrity", "research_scope", "monitoring_parity", "candidate_evaluation_protocol_documented"]}
            report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(directory / "verification.json", report, exclusive=True)
        write_json(manifest_path.parent / "verification.json", report)
        print(f"Verification report: {directory / 'verification.json'}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--integrity-only", action="store_true", help="Check frozen files/environment only; not the full workflow")
    args = parser.parse_args()
    report = verify_release(args.manifest, args.integrity_only)
    print(f"{report['status']}: release {report['release_id']}; final test remains unevaluated.")
