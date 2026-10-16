"""Immutable development release fingerprints and earlier MLflow comparisons."""

from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import uuid

from api.model import CONTRACT_VERSION, EXPECTED_MODEL_SHA256, FEATURE_COLUMNS, FEATURE_MAPPING, PREPROCESSING
from api.schemas import PredictionResponse, TransactionRequest
from dashboard.evidence import file_hash, load_evidence_table
from tracking.common import git_state

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/integration/session25"
MANIFEST = OUTPUT / "release_manifest.json"
PROTOCOL = ROOT / "integration/evaluation_protocol.json"

# Explicit paths only: no dataset discovery and no reserved-test paths.
ARTIFACTS = {
    "model": "outputs/evaluation/session20/AllFive.cbm",
    "predictions": "outputs/evaluation/session20/predictions.csv",
    "training_configuration": "outputs/evaluation/session20/config.json",
    "training_targets": "outputs/evaluation/session20/training_targets.csv",
    "evaluation_targets": "outputs/evaluation/session20/evaluation_targets.csv",
    "development_metrics": "outputs/evaluation/session20/model_comparison.csv",
    "training_records": "data/processed/train.csv",
    "validation_records": "data/processed/validation.csv",
    "training_graph_features": "outputs/graph/train_graph_features.csv",
    "validation_graph_features": "outputs/graph/validation_graph_features.csv",
    "golden_vectors": "tests/fixtures/session22_golden.json",
    "shap_export": "outputs/dashboard/session24/shap_export.json",
    "cluster_summary": "outputs/graph/session16/summary.json",
    "cluster_components": "outputs/graph/session16/component_summary.csv",
    "cluster_memberships": "outputs/graph/session16/component_memberships.csv",
    "cluster_edges": "outputs/graph/session16/edge_evidence.csv",
    "monitoring_bins": "outputs/monitoring/session23/reference_bins.json",
    "monitoring_features": "outputs/monitoring/session23/feature_drift.csv",
    "monitoring_scores": "outputs/monitoring/session23/prediction_drift.json",
    "monitoring_details": "outputs/monitoring/session23/bin_proportions.csv",
    "monitoring_windows": "outputs/monitoring/session23/window_features.csv",
    "monitoring_summary": "outputs/monitoring/session23/run_summary.json",
    "evaluation_protocol": "integration/evaluation_protocol.json",
}
SOURCE_DIRECTORIES = ("api", "dashboard", "monitoring", "tracking", "integration", "tests",
                      "graph_models", "deep_learning", "anomaly_models", "fusion_models", "decision_models", "evaluation")


def write_json(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def contract_metadata():
    return {"version": CONTRACT_VERSION, "columns": FEATURE_COLUMNS, "numeric_dtype": "float64",
            "request_mapping": FEATURE_MAPPING, "preprocessing": PREPROCESSING,
            "missing_history": "null or omitted -> NaN; zero remains zero",
            "request_validation": TransactionRequest.model_json_schema(),
            "prediction_response": PredictionResponse.model_json_schema()}


def source_paths(root=ROOT):
    root = Path(root)
    paths = set(root.glob("*.py")) | set(root.glob("requirements*.txt"))
    if (root / "README.md").is_file():
        paths.add(root / "README.md")
    for directory in SOURCE_DIRECTORIES:
        paths.update((root / directory).rglob("*.py"))
        paths.update((root / directory).glob("*.md"))
    paths.add(root / "integration/evaluation_protocol.json")
    return sorted(path for path in paths if path.is_file())


def file_spec(path, root=ROOT):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Required release file is missing or outside the workspace: {path}")
    return {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size}


def verify_specs(specs, root=ROOT):
    root = Path(root).resolve()
    for name, spec in specs.items():
        relative = Path(spec["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Release path escapes the workspace: {name}")
        if not path.is_file():
            raise ValueError(f"Missing release artifact: {name}")
        if file_hash(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError(f"Release artifact changed: {name}")


def environment_snapshot():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"]))}


def validate_protocol(protocol):
    if (protocol["status"] != "candidate_plan_not_final_model_approval"
            or protocol["final_model_selection_approved"] is not False
            or protocol["reserved_test_evaluated"] is not False or protocol["execute_reserved_test"] is not False
            or protocol["decision_policy"] is not None or protocol["candidate_model"]["sha256"] != EXPECTED_MODEL_SHA256
            or protocol["features"] != FEATURE_COLUMNS or protocol["feature_contract_version"] != CONTRACT_VERSION
            or protocol["metrics"]["top_k"] != 100 or protocol["cost_diagnostic"]["false_positive_cost"] != 200
            or protocol["cost_diagnostic"]["optimize_on_test"] is not False):
        raise ValueError("Protocol must retain the frozen development-only scope and explicit final-test prerequisites")


def compare_mlflow_record(registration, specs):
    from tracking.verify_session21 import verify_manifest
    verify_manifest(registration)
    row = next(run for run in registration["runs"] if run["model"] == "AllFive")
    lookup = {record["artifact_path"]: record for record in row["artifacts"]}
    checks = {"model": "model/AllFive.cbm", "predictions": "reports/predictions.csv",
              "training_configuration": "reports/config.json"}
    for name, artifact in checks.items():
        expected = lookup[artifact]
        if specs[name]["sha256"] != expected["sha256"] or specs[name]["bytes"] != expected["bytes"]:
            raise ValueError(f"Release {name} differs from prior Session 21 MLflow registration")
    registered_inputs = {path.replace("\\", "/"): value for path, value in registration["input_files"].items()}
    for name in ["training_records", "validation_records", "training_graph_features", "validation_graph_features"]:
        expected = registered_inputs[specs[name]["path"]]
        if specs[name]["sha256"] != expected["sha256"] or specs[name]["bytes"] != expected["bytes"]:
            raise ValueError(f"Release input differs from registration-time file: {name}")
    if specs["model"]["sha256"] != EXPECTED_MODEL_SHA256:
        raise ValueError("MLflow record and pinned serving artifact differ")
    return {"registration_id": registration["registration_id"], "run_id": row["run_id"],
            "experiment_id": registration["experiment_id"], "tracking_uri": registration["tracking_uri"],
            "compared_artifacts": checks,
            "scope": "earlier Session 21 registration-time evidence; not original training-code provenance"}


def build_release(output=OUTPUT):
    output = Path(output)
    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    load_evidence_table()  # checks original ordered inputs, labels, IDs and pinned model
    artifacts = {name: file_spec(ROOT / path) for name, path in ARTIFACTS.items()}
    latest = json.loads((ROOT / "outputs/tracking/experiment_manifest.json").read_text())
    registration_path = ROOT / "outputs/tracking/registrations" / latest["registration_id"] / "experiment_manifest.json"
    registration = json.loads(registration_path.read_text())
    if registration != latest:
        raise ValueError("Latest MLflow pointer differs from its retained registration snapshot")
    provenance = compare_mlflow_record(registration, artifacts)
    artifacts["mlflow_registration"] = file_spec(registration_path)
    sources = {path.relative_to(ROOT).as_posix(): file_spec(path) for path in source_paths()}
    release_id = uuid.uuid4().hex
    manifest = {"schema_version": 1, "release_id": release_id, "release_name": "sentinelgraph-demo-candidate",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "development",
                "evaluation_status": "development_candidate_not_final_selection", "model_name": "AllFive",
                "model_sha256": artifacts["model"]["sha256"], "feature_contract": contract_metadata(),
                "score_type": "uncalibrated_model_score", "decision_policy": None,
                "reserved_test_evaluated": False, "test_access": False, "final_test_ready": False,
                "training_reexecuted": False, "artifacts": artifacts, "source_files": sources,
                "mlflow_provenance": provenance, "evaluation_protocol": protocol,
                "environment": environment_snapshot(), "build_git": git_state(),
                "provenance_scope": "release-build source/file fingerprints; original training revision not captured",
                "workflow_scope": "precomputed historical features -> API -> saved dashboard/SHAP; offline monitoring; no live feature store"}
    directory = output / "releases" / release_id
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "release_manifest.json"
    write_json(path, manifest, exclusive=True)
    write_json(output / "release_manifest.json", manifest)
    return path, manifest


def verify_integrity(manifest, root=ROOT, check_environment=True):
    if (manifest["schema_version"] != 1 or manifest["stage"] != "development"
            or manifest["evaluation_status"] != "development_candidate_not_final_selection"
            or manifest["reserved_test_evaluated"] is not False or manifest["test_access"] is not False
            or manifest["final_test_ready"] is not False or manifest["training_reexecuted"] is not False
            or manifest["decision_policy"] is not None or manifest["score_type"] != "uncalibrated_model_score"
            or manifest["model_sha256"] != EXPECTED_MODEL_SHA256 or manifest["feature_contract"] != contract_metadata()):
        raise ValueError("Release scope, model identity or feature contract differs")
    if set(manifest["artifacts"]) != {*ARTIFACTS, "mlflow_registration"}:
        raise ValueError("Release artifact inventory is incomplete or unexpected")
    for name, relative in ARTIFACTS.items():
        if manifest["artifacts"][name]["path"] != relative:
            raise ValueError("Required artifact path differs from the supported development workflow")
    current_sources = {path.relative_to(root).as_posix() for path in source_paths(root)}
    if set(manifest["source_files"]) != current_sources:
        raise ValueError("Release source inventory changed")
    verify_specs(manifest["artifacts"], root)
    verify_specs(manifest["source_files"], root)
    protocol = json.loads((Path(root) / ARTIFACTS["evaluation_protocol"]).read_text())
    validate_protocol(protocol)
    if manifest["evaluation_protocol"] != protocol:
        raise ValueError("Embedded and frozen evaluation protocols differ")
    registration = json.loads((Path(root) / manifest["artifacts"]["mlflow_registration"]["path"]).read_text())
    if compare_mlflow_record(registration, manifest["artifacts"]) != manifest["mlflow_provenance"]:
        raise ValueError("Earlier MLflow provenance differs from frozen release metadata")
    if check_environment and environment_snapshot() != manifest["environment"]:
        raise ValueError("Installed release environment changed; investigate before creating a new release")
    return {"status": "PASS", "artifacts": len(manifest["artifacts"]), "source_files": len(manifest["source_files"])}
