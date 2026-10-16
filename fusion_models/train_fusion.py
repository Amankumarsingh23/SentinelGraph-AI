"""Session 18: chronological holdout stacking on frozen anomaly scores.

This is a development comparison on reused validation data, not walk-forward
OOF stacking or an independent final test. The reserved test files are not read.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anomaly_models.train_anomaly_models import (
    FEATURES, Autoencoder, autoencoder_scores, load_split, raw_features,
)
from deep_learning.run_behavioral import metrics, write_json
from evaluate_risk import chronological_midpoint

ANOMALY = ROOT / "outputs/anomaly/session17"
SCORE_COLUMNS = {"isolation_forest": "IsolationForestScore", "autoencoder": "AutoencoderScore"}
STACKS = {
    "catboost_autoencoder": ["CatBoostScore", "AutoencoderScore"],
    "catboost_isolation_forest": ["CatBoostScore", "IsolationForestScore"],
    "catboost_all_anomalies": ["CatBoostScore", "AutoencoderScore", "IsolationForestScore"],
}


def merge_anomaly_scores(validation, predictions):
    """Align two saved score files and labels to raw validation by unique ID."""
    if not validation.TransactionID.is_unique or validation.TransactionID.isna().any():
        raise ValueError("Validation TransactionIDs must be unique and nonmissing")
    expected_ids = None
    result = None
    for name, column in SCORE_COLUMNS.items():
        table = predictions[name]
        if table.empty or table.TransactionID.isna().any() or not table.TransactionID.is_unique:
            raise ValueError(f"{name}: need nonempty, unique, nonmissing prediction IDs")
        if not np.isfinite(table.anomaly_score.to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: anomaly scores must be finite")
        if "model" in table and not table.model.eq(name).all():
            raise ValueError(f"{name}: incorrect saved model name")
        if expected_ids is None:
            expected_ids = set(table.TransactionID)
            if not table.TransactionID.isin(validation.TransactionID).all():
                raise ValueError("Saved anomaly IDs must belong to validation")
            result = validation.loc[validation.TransactionID.isin(expected_ids)].copy()
        elif set(table.TransactionID) != expected_ids:
            raise ValueError("Both anomaly models must score exactly the same IDs")
        renamed = table[["TransactionID", "isFraud", "anomaly_score"]].rename(
            columns={"isFraud": "saved_label", "anomaly_score": column}
        )
        result = result.merge(renamed, on="TransactionID", validate="one_to_one")
        if not result.isFraud.eq(result.saved_label).all():
            raise ValueError(f"{name}: saved labels disagree with original validation labels")
        result = result.drop(columns="saved_label")
    return result.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


def holdout_partition(frame):
    """Move the midpoint after timestamp ties, reusing the Phase 1 policy."""
    mid = chronological_midpoint(frame.TransactionDT.to_numpy())
    meta, evaluation = frame.iloc[:mid].copy(), frame.iloc[mid:].copy()
    for name, part in [("meta-training", meta), ("evaluation", evaluation)]:
        if not part.isFraud.isin([0, 1]).all() or part.isFraud.nunique() != 2:
            raise ValueError(f"{name} must have both binary label classes")
    return meta, evaluation


def fit_stacker(meta, columns, seed=42):
    """Accept only the designated meta-training table; scaling stays inside fit."""
    stacker = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=seed,
    ))
    stacker.fit(meta[columns], meta.isFraud)
    return stacker


def verify_frozen_anomalies(train, scored, anomaly_dir):
    """Audit recorded training provenance and reproduce scores without any refit."""
    config = json.loads((anomaly_dir / "config.json").read_text())
    targets = pd.read_csv(anomaly_dir / "train_targets.csv")
    val_targets = pd.read_csv(anomaly_dir / "validation_targets.csv")
    if config.get("test_access") is not False or config["features"] != FEATURES:
        raise ValueError("Anomaly experiment provenance/features do not match Session 17")
    if not targets.TransactionID.is_unique or not targets.TransactionID.isin(train.TransactionID).all():
        raise ValueError("Anomaly training targets must be unique rows from the training period")
    source_train = train.set_index("TransactionID").loc[targets.TransactionID].reset_index()
    np.testing.assert_array_equal(source_train.TransactionDT, targets.TransactionDT)
    np.testing.assert_array_equal(source_train.isFraud, targets.isFraud)
    if source_train.TransactionDT.max() >= scored.TransactionDT.min():
        raise ValueError("Frozen anomaly models must be fitted strictly before score targets")
    np.testing.assert_array_equal(scored.TransactionID, val_targets.TransactionID)
    np.testing.assert_array_equal(scored.isFraud, val_targets.isFraud)
    scaler = joblib.load(anomaly_dir / "scaler.joblib")
    training_inputs = raw_features(source_train)
    expected_scaler = StandardScaler().fit(training_inputs)
    np.testing.assert_allclose(scaler.mean_, expected_scaler.mean_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(scaler.scale_, expected_scaler.scale_, rtol=1e-10, atol=1e-10)
    if scaler.n_samples_seen_ != len(source_train):
        raise ValueError("Anomaly scaler fit-row count differs from recorded training targets")
    X = scaler.transform(raw_features(scored)).astype(np.float32)
    forest = joblib.load(anomaly_dir / "isolation_forest.joblib")
    checkpoint = torch.load(anomaly_dir / "autoencoder.pt", map_location="cpu", weights_only=True)
    model = Autoencoder(checkpoint["input_size"], checkpoint["hidden_size"], checkpoint["latent_size"])
    model.load_state_dict(checkpoint["state_dict"])
    actual = {"IsolationForestScore": -forest.score_samples(X),
              "AutoencoderScore": autoencoder_scores(model, X, config["batch_size"])}
    errors = {}
    for column, values in actual.items():
        np.testing.assert_allclose(values, scored[column], rtol=1e-6, atol=1e-7)
        errors[column] = float(np.max(np.abs(values - scored[column])))
    return {"status": "PASS", "training_rows": len(source_train), "scored_rows": len(scored),
            "training_only_scaler": "PASS", "max_score_errors": errors,
            "training_end_time": float(source_train.TransactionDT.max()), "refit": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-targets", type=int, default=50_000)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--anomaly-dir", type=Path, default=ANOMALY)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion/session18")
    args = parser.parse_args()
    if min(args.train_targets, args.iterations, args.threads) < 1:
        parser.error("Training target limit, tree iterations and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    full_train, validation = load_split("train"), load_split("validation")
    predictions = {name: pd.read_csv(args.anomaly_dir / f"{name}_predictions.csv")
                   for name in SCORE_COLUMNS}
    scored = merge_anomaly_scores(validation, predictions)
    if full_train.TransactionDT.max() >= scored.TransactionDT.min():
        raise ValueError("All base-model training data must precede the scored validation subset")
    audit = verify_frozen_anomalies(full_train, scored, args.anomaly_dir)
    write_json(args.output / "frozen_anomaly_audit.json", audit)
    train = full_train.tail(args.train_targets).copy()
    if not train.isFraud.isin([0, 1]).all() or train.isFraud.nunique() != 2:
        raise ValueError("CatBoost training needs both classes")
    cat_parameters = {"iterations": args.iterations, "depth": 6, "learning_rate": 0.05,
                      "loss_function": "Logloss", "random_seed": args.seed,
                      "thread_count": args.threads, "verbose": False, "allow_writing_files": False}
    catboost = CatBoostClassifier(**cat_parameters)
    # Native missing-value handling; no scaler is learned for these five raw inputs.
    X_train = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    X_scored = scored[FEATURES].replace([np.inf, -np.inf], np.nan)
    timer = time.perf_counter()
    catboost.fit(X_train, train.isFraud)
    cat_seconds = time.perf_counter() - timer
    timer = time.perf_counter()
    scored["CatBoostScore"] = catboost.predict_proba(X_scored)[:, 1]
    cat_inference = time.perf_counter() - timer
    catboost.save_model(str(args.output / "catboost.cbm"))
    meta, evaluation = holdout_partition(scored)
    scored["role"] = np.where(np.arange(len(scored)) < len(meta), "meta_train", "evaluation")
    scored[["TransactionID", "TransactionDT", "isFraud", "role", "CatBoostScore",
            "AutoencoderScore", "IsolationForestScore"]].to_csv(args.output / "base_scores.csv", index=False)
    train[["TransactionID", "TransactionDT", "isFraud"]].to_csv(args.output / "base_training_targets.csv", index=False)
    comparison, coefficients = [], []
    evaluated = evaluation[["TransactionID", "TransactionDT", "isFraud"]].copy()

    def record(name, scores, training_seconds, inference_seconds):
        evaluated[name] = scores
        comparison.append({"model": name, **metrics(evaluation.isFraud.to_numpy(), scores,
                                                   evaluation.TransactionID.to_numpy()),
                           "training_seconds": training_seconds, "inference_seconds": inference_seconds})

    record("catboost", evaluation.CatBoostScore.to_numpy(), cat_seconds, cat_inference)
    record("autoencoder", evaluation.AutoencoderScore.to_numpy(), None, None)
    record("isolation_forest", evaluation.IsolationForestScore.to_numpy(), None, None)
    for name, columns in STACKS.items():
        timer = time.perf_counter()
        stacker = fit_stacker(meta, columns, args.seed)
        training_seconds = time.perf_counter() - timer
        timer = time.perf_counter()
        scores = stacker.predict_proba(evaluation[columns])[:, 1]
        record(name, scores, training_seconds, time.perf_counter() - timer)
        joblib.dump(stacker, args.output / f"{name}.joblib")
        scaler, logistic = stacker.steps[0][1], stacker.steps[1][1]
        for column, coefficient, mean, scale in zip(columns, logistic.coef_[0], scaler.mean_, scaler.scale_):
            coefficients.append({"model": name, "score_feature": column, "standardized_coefficient": coefficient,
                                 "scaler_mean": mean, "scaler_scale": scale,
                                 "standardized_intercept": float(logistic.intercept_[0])})
    comparison = pd.DataFrame(comparison)
    cat_ap = comparison.loc[comparison.model == "catboost", "average_precision"].item()
    comparison["delta_ap_vs_catboost"] = comparison.average_precision - cat_ap
    comparison.to_csv(args.output / "model_comparison.csv", index=False)
    evaluated.to_csv(args.output / "evaluation_predictions.csv", index=False)
    pd.DataFrame(coefficients).to_csv(args.output / "stacker_coefficients.csv", index=False)
    base_columns = ["CatBoostScore", "AutoencoderScore", "IsolationForestScore"]
    for role, part in [("meta_train", meta), ("evaluation", evaluation)]:
        for method in ["pearson", "spearman"]:
            part[base_columns].corr(method=method).to_csv(args.output / f"{role}_{method}_correlations.csv")
    partitions = {}
    for name, part in [("base_train", train), ("meta_train", meta), ("evaluation", evaluation)]:
        partitions[name] = {"rows": len(part), "frauds": int(part.isFraud.sum()),
                            "prevalence": float(part.isFraud.mean()),
                            "start_time": float(part.TransactionDT.min()),
                            "end_time": float(part.TransactionDT.max())}
    hashes = {}
    for filename in ["config.json", "train_targets.csv", "validation_targets.csv", "scaler.joblib",
                     "isolation_forest.joblib", "autoencoder.pt", "isolation_forest_predictions.csv",
                     "autoencoder_predictions.csv"]:
        with (args.anomaly_dir / filename).open("rb") as stream:
            hashes[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(args.output / "config.json", {
        **vars(args), "output": str(args.output), "anomaly_dir": str(args.anomaly_dir),
        "features": FEATURES, "catboost": cat_parameters, "stacks": STACKS,
        "meta_model": {"scaler_fit": "meta_train only", "model": "LogisticRegression",
                       "C": 1.0, "solver": "lbfgs", "class_weight": None, "max_iter": 1000},
        "boundary_policy": "chronological midpoint advanced past timestamp ties",
        "partitions": partitions, "frozen_anomaly_sha256": hashes, "test_access": False,
        "experiment": "development chronological holdout stacking; reused validation; not OOF",
    })
    write_json(args.output / "run_summary.json", {
        "completed_models": comparison.model.tolist(), "failures": [],
        "runtime_seconds": time.perf_counter() - start, "test_access": False,
        "timing": "CatBoost inference scores all saved validation IDs; stack inference scores evaluation only; frozen anomaly training times not rerun",
    })
    print(json.dumps(partitions, indent=2))
    print("\nSAME-ROW EVALUATION COMPARISON")
    print(comparison.to_string(index=False))
    print("\nDevelopment holdout stacking only; the evaluation half was previously included in Session 17 reports.")


if __name__ == "__main__":
    main()
