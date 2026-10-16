"""Register completed Session 20 artifacts locally in MLflow, without training."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tracking.common import (
    EXPERIMENT, MODEL_NAMES, RESULTS, TRACKING, artifact_record,
    git_state, sha256_file, tracking_uri, write_json,
)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from anomaly_models.train_anomaly_models import load_split
from deep_learning.run_behavioral import metrics
from evaluation.session20_ablations import align_evaluation, input_hash

REPORT_FILES = ["config.json", "model_comparison.csv", "predictions.csv",
                "training_targets.csv", "evaluation_targets.csv", "bootstrap_summary.json",
                "bootstrap_draws.csv", "chronological_blocks.csv", "run_summary.json",
                "artifact_verification.json"]


def validate_reports(results):
    """Complete preflight before initializing a tracking store or creating runs."""
    results = Path(results)
    required = REPORT_FILES + [f"{name}.cbm" for name in MODEL_NAMES]
    for name in required:
        if not (results / name).is_file():
            raise FileNotFoundError(f"Required Session 20 artifact missing: {results / name}")
    config = json.loads((results / "config.json").read_text())
    table = pd.read_csv(results / "model_comparison.csv")
    if len(table) != len(MODEL_NAMES) or table.model.duplicated().any() or set(table.model) != set(MODEL_NAMES):
        raise ValueError("Comparison must contain exactly one row per expected ablation model")
    if set(config["feature_groups"]) != set(MODEL_NAMES) or config.get("test_access") is not False:
        raise ValueError("Unexpected Session 20 groups or test-access scope")
    predictions = pd.read_csv(results / "predictions.csv")
    targets = pd.read_csv(results / "evaluation_targets.csv")
    train_targets = pd.read_csv(results / "training_targets.csv")
    if not predictions.TransactionID.is_unique or not train_targets.TransactionID.is_unique:
        raise ValueError("Prediction and training IDs must be unique")
    np.testing.assert_array_equal(predictions.TransactionID, targets.TransactionID)
    np.testing.assert_array_equal(predictions.TransactionDT, targets.TransactionDT)
    np.testing.assert_array_equal(predictions.isFraud, targets.isFraud)
    if train_targets.TransactionDT.max() >= targets.TransactionDT.min():
        raise ValueError("Training targets must strictly precede evaluation")
    for row in table.to_dict(orient="records"):
        if row["training_rows"] != len(train_targets) or row["val_examples"] != len(targets):
            raise ValueError("Reported counts disagree with saved target rows")
        if row["features"].split(",") != config["feature_groups"][row["model"]]:
            raise ValueError("Reported feature order differs from configuration")
        actual = metrics(predictions.isFraud.to_numpy(), predictions[row["model"]].to_numpy(),
                         predictions.TransactionID.to_numpy())
        for key, value in actual.items():
            np.testing.assert_allclose(value, row[key], rtol=1e-10, atol=1e-12)
    return config, table, predictions, train_targets


def verify_current_inputs(results, config, predictions, train_targets):
    """Reproduce saved predictions, not training. Check stored ordered-input hashes."""
    full_train = load_split("train")
    train = full_train.tail(config["train_targets"])
    evaluation = align_evaluation(load_split("validation"), pd.read_csv(config["reference"]))
    np.testing.assert_array_equal(train.TransactionID, train_targets.TransactionID)
    np.testing.assert_array_equal(train.isFraud, train_targets.isFraud)
    np.testing.assert_array_equal(evaluation.TransactionID, predictions.TransactionID)
    np.testing.assert_array_equal(evaluation.isFraud, predictions.isFraud)
    features = config["feature_groups"]["AllFive"]
    for frame, key in [(train, "training_input_hash"), (evaluation, "evaluation_input_hash")]:
        if input_hash(frame, ["TransactionID", "TransactionDT", "isFraud", *features]) != config[key]:
            raise ValueError(f"Current ordered inputs differ from saved training-time {key}")
    errors = {}
    for name in MODEL_NAMES:
        model = CatBoostClassifier()
        model.load_model(str(Path(results) / f"{name}.cbm"))
        columns = config["feature_groups"][name]
        if model.feature_names_ != columns or model.tree_count_ != config["shared_catboost_parameters"]["iterations"]:
            raise ValueError(f"Model feature order/tree count mismatch: {name}")
        scores = model.predict_proba(evaluation[columns].replace([np.inf, -np.inf], np.nan))[:, 1]
        np.testing.assert_allclose(scores, predictions[name], rtol=1e-10, atol=1e-12)
        errors[name] = float(np.max(np.abs(scores - predictions[name])))
    return {"status": "PASS", "ordered_training_inputs_match": True,
            "ordered_evaluation_inputs_match": True, "max_prediction_errors": errors,
            "training_reexecuted": False, "test_access": False}


def environment_snapshot():
    packages = {dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "executable": sys.executable, "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower()))}


def source_files():
    sources = list(ROOT.glob("*.py")) + list(ROOT.glob("requirements*.txt"))
    for directory in ["deep_learning", "graph_models", "anomaly_models", "fusion_models",
                      "decision_models", "evaluation", "tracking", "tests"]:
        sources.extend((ROOT / directory).rglob("*.py"))
    return sorted(sources)


def main():
    # Tracking is optional for the artifact-free report-validation tests.
    import mlflow
    from mlflow.tracking import MlflowClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--store-dir", type=Path, default=TRACKING)
    args = parser.parse_args()
    print("Preflight: checking artifacts, metrics, selected input fingerprints and saved-model predictions...", flush=True)
    config, table, predictions, train_targets = validate_reports(args.results)
    audit = verify_current_inputs(args.results, config, predictions, train_targets)
    input_paths = [ROOT / "data/processed/train.csv", ROOT / "data/processed/validation.csv",
                   ROOT / "outputs/graph/train_graph_features.csv", ROOT / "outputs/graph/validation_graph_features.csv",
                   Path(config["reference"])]
    inputs = {str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size,
              "hash_scope": "current registration-time file; original whole-file training hash not recorded"}
              for path in input_paths}
    registration_id = uuid.uuid4().hex
    directory = args.store_dir.resolve() / "registrations" / registration_id
    provenance = directory / "provenance"
    provenance.mkdir(parents=True)
    git = git_state()
    environment = environment_snapshot()
    write_json(provenance / "registration_environment.json", environment)
    write_json(provenance / "git_state.json", git)
    write_json(provenance / "input_files.json", inputs)
    write_json(provenance / "prediction_preflight.json", audit)
    (provenance / "requirements_snapshot.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in environment["packages"].items()) + "\n", encoding="utf-8"
    )
    code_hashes = {}
    for source in source_files():
        relative = source.relative_to(ROOT)
        target = provenance / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        code_hashes[relative.as_posix()] = sha256_file(source)
    write_json(provenance / "source_fingerprints.json", {
        "scope": "registration-time source snapshot; original training code revision/hashes unavailable",
        "files": code_hashes,
    })
    # Snapshot historical result bytes before logging, so tracking owns its copies.
    reports = directory / "reports"
    reports.mkdir()
    for filename in REPORT_FILES:
        shutil.copy2(args.results / filename, reports / filename)
    uri = tracking_uri(args.store_dir)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    artifact_location = (args.store_dir.resolve() / "artifacts").as_uri()
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT, artifact_location=artifact_location)
    else:
        if experiment.lifecycle_stage != "active":
            raise ValueError("The named tracking experiment is not active")
        experiment_id = experiment.experiment_id
    manifest = {"registration_id": registration_id, "status": "IN_PROGRESS", "experiment_name": EXPERIMENT,
                "experiment_id": experiment_id, "tracking_uri": uri,
                "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                "record_type": "registration of existing completed training artifacts; no retraining",
                "training_configuration": config, "registration_git": git, "input_files": inputs,
                "training_time_input_fingerprints": {key: config[key] for key in ["training_input_hash", "evaluation_input_hash"]},
                "preflight": audit, "test_access": False, "runs": []}
    shared = [(path, "provenance/" + path.relative_to(provenance).as_posix())
              for path in sorted(provenance.rglob("*")) if path.is_file()]
    shared.extend((reports / filename, "reports/" + filename) for filename in REPORT_FILES)
    try:
        for name in MODEL_NAMES:
            row = table.set_index("model").loc[name]
            parameters = {**config["shared_catboost_parameters"], "model_family": "CatBoost",
                          "feature_group": name, "feature_names": json.dumps(config["feature_groups"][name]),
                          "training_rows": len(train_targets), "evaluation_rows": len(predictions),
                          "preprocessing": config["preprocessing"], "bootstrap_repeats": config["bootstrap_repeats"],
                          "seed": config["seed"]}
            numeric = {key: float(row[key]) for key in ["average_precision", "precision_at_100", "recall_at_100",
                       "validation_prevalence", "delta_ap_vs_amount_only", "training_seconds", "inference_seconds"]}
            tags = {"registration_id": registration_id, "record_type": "historical_artifact_registration",
                    "evaluation_scope": "reused_development_validation", "training_reexecuted": "false",
                    "registration_git_commit": git["registration_commit"] or "unavailable_no_commit",
                    "original_training_git_commit": "not_recorded", "card1_semantics": "anonymized_proxy_not_verified_customer",
                    "test_access": "false"}
            with mlflow.start_run(experiment_id=experiment_id, run_name=name, tags=tags) as run:
                mlflow.log_params(parameters)
                mlflow.log_metrics(numeric)
                model_path = args.results / f"{name}.cbm"
                payloads = [*shared, (model_path, "model/" + model_path.name)]
                artifacts = []
                for source, destination in payloads:
                    mlflow.log_artifact(str(source), artifact_path=str(Path(destination).parent).replace("\\", "/"))
                    artifacts.append(artifact_record(source, destination))
                record = {"model": name, "run_id": run.info.run_id, "parameters": {key: str(value) for key, value in parameters.items()},
                          "metrics": numeric, "tags": tags, "artifacts": artifacts}
                record_path = directory / f"{name}_run_record.json"
                write_json(record_path, record)
                mlflow.log_artifact(str(record_path), artifact_path="provenance")
                record["artifacts"].append(artifact_record(record_path, "provenance/" + record_path.name))
                manifest["runs"].append(record)
            print(f"Registered {name}: run_id={run.info.run_id}, AP={numeric['average_precision']:.6f}", flush=True)
        manifest["status"] = "FINISHED"
    except Exception:
        manifest["status"] = "FAILED"
        write_json(directory / "experiment_manifest.json", manifest)
        raise
    manifest_path = directory / "experiment_manifest.json"
    write_json(manifest_path, manifest)
    write_json(args.store_dir / "experiment_manifest.json", manifest)
    print(f"\nManifest: {manifest_path}")
    print(f"Tracking URI: {uri}")
    print("Four historical results registered. No model was retrained and no test data was read.")


if __name__ == "__main__":
    main()
