"""Session 19: frozen CatBoost calibration and chronological alert-policy cost.

All results are development diagnostics on previously examined validation data.
Costs are illustrative dataset-amount units; the reserved test is never opened.
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
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from anomaly_models.train_anomaly_models import FEATURES, load_split
from deep_learning.run_behavioral import metrics, write_json
from evaluate_risk import alert_ranking, chronological_midpoint

FUSION = ROOT / "outputs/fusion/session18"
ANOMALY = ROOT / "outputs/anomaly/session17"
OUTPUT = ROOT / "outputs/decisions/session19"
CAPACITIES = [0, 25, 50, 100, 200, 500, 1000]


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def aligned_saved_scores(frame, saved, column):
    """Return saved scores in frame order after strict ID/label/time checks."""
    for data in [frame, saved]:
        if not data.TransactionID.is_unique or data.TransactionID.isna().any():
            raise ValueError("Score alignment requires unique nonmissing IDs")
    if len(frame) != len(saved) or not saved.TransactionID.isin(frame.TransactionID).all():
        raise ValueError("Saved and expected prediction IDs must match exactly")
    ordered = saved.set_index("TransactionID").loc[frame.TransactionID]
    if not np.array_equal(ordered.isFraud.to_numpy(), frame.isFraud.to_numpy()):
        raise ValueError("Saved prediction labels differ from original validation")
    if "TransactionDT" in ordered and not np.array_equal(
        ordered.TransactionDT.to_numpy(), frame.TransactionDT.to_numpy()
    ):
        raise ValueError("Saved prediction timestamps differ from original validation")
    values = ordered[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Saved scores must be finite")
    return values


def load_frozen_inputs(fusion_dir=FUSION, anomaly_dir=ANOMALY):
    """Restore exact CatBoost inputs and verify existing scores, without fitting."""
    config = json.loads((fusion_dir / "config.json").read_text())
    model = CatBoostClassifier()
    model.load_model(str(fusion_dir / "catboost.cbm"))
    if config["features"] != FEATURES or model.feature_names_ != FEATURES:
        raise ValueError("Saved CatBoost feature order differs from the input pipeline")
    targets = pd.read_csv(anomaly_dir / "validation_targets.csv")
    if targets.empty or not targets.TransactionID.is_unique or targets.TransactionID.isna().any():
        raise ValueError("Need the unique Session 17 validation target IDs")
    validation = load_split("validation")
    if not targets.TransactionID.isin(validation.TransactionID).all():
        raise ValueError("Recorded validation targets are absent from original data")
    frame = validation.set_index("TransactionID").loc[targets.TransactionID].reset_index()
    np.testing.assert_array_equal(frame.isFraud, targets.isFraud)
    np.testing.assert_array_equal(frame.TransactionDT, targets.TransactionDT)
    frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    if not frame.isFraud.isin([0, 1]).all():
        raise ValueError("Need binary evaluation labels")
    trained = pd.read_csv(fusion_dir / "base_training_targets.csv")
    if trained.TransactionDT.max() >= frame.TransactionDT.min() or trained.TransactionID.isin(frame.TransactionID).any():
        raise ValueError("The frozen CatBoost training period must strictly precede validation")
    X = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    frame["CatBoostScore"] = model.predict_proba(X)[:, 1]
    saved_base = fusion_dir / "base_scores.csv"
    if saved_base.exists():
        scores = aligned_saved_scores(frame, pd.read_csv(saved_base), "CatBoostScore")
        np.testing.assert_allclose(frame.CatBoostScore, scores, rtol=1e-9, atol=1e-12)
        source = "verified Session 18 base_scores.csv against the frozen model"
    else:
        source = "regenerated all out-of-sample scores using frozen Session 18 CatBoost; no refit"
    return model, frame, {"status": "PASS", "score_source": source,
                          "catboost_sha256": file_hash(fusion_dir / "catboost.cbm"),
                          "training_end_time": float(trained.TransactionDT.max()),
                          "features": FEATURES, "test_access": False, "refit": False}


def three_way_partition(frame):
    """Split at halves, preserving complete timestamp groups at both boundaries."""
    eval_start = chronological_midpoint(frame.TransactionDT.to_numpy())
    cal_end = chronological_midpoint(frame.TransactionDT.iloc[:eval_start].to_numpy())
    parts = {"calibration": frame.iloc[:cal_end].copy(),
             "policy_development": frame.iloc[cal_end:eval_start].copy(),
             "evaluation": frame.iloc[eval_start:].copy()}
    for name, part in parts.items():
        if part.empty or not part.isFraud.isin([0, 1]).all() or part.isFraud.nunique() != 2:
            raise ValueError(f"{name} needs both binary label classes")
    return parts


def score_logits(probabilities):
    """Sigmoid calibration uses CatBoost log-odds, not anomaly scores."""
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("CatBoost probabilities must be finite and between zero and one")
    clipped = np.clip(values, 1e-12, 1 - 1e-12)
    return (np.log(clipped) - np.log1p(-clipped)).reshape(-1, 1)


def fit_calibrator(calibration):
    """Only the designated earlier calibration block is accepted for fitting."""
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(score_logits(calibration.CatBoostScore), calibration.isFraud)
    return model


def cost_report(labels, amounts, alerts, review_cost=200.0):
    """Realized proxy = cost per false positive + full amount of missed fraud.

    True positives/negatives cost zero under this illustrative assumption.
    Amount units are those of the dataset, not asserted to be INR or another currency.
    """
    labels, amounts, alerts = np.asarray(labels), np.asarray(amounts, dtype=float), np.asarray(alerts)
    if not (labels.ndim == amounts.ndim == alerts.ndim == 1) or not (len(labels) == len(amounts) == len(alerts)):
        raise ValueError("Cost inputs must be aligned one-dimensional arrays")
    if not np.isin(labels, [0, 1]).all() or alerts.dtype != np.bool_:
        raise ValueError("Need binary labels and Boolean alert decisions")
    if not np.isfinite(amounts).all() or (amounts < 0).any() or not np.isfinite(review_cost) or review_cost < 0:
        raise ValueError("Amounts and false-positive cost must be finite and nonnegative")
    fraud, legitimate = labels == 1, labels == 0
    tp, fp = int((fraud & alerts).sum()), int((legitimate & alerts).sum())
    missed_amount = float(amounts[fraud & ~alerts].sum())
    caught_amount = float(amounts[fraud & alerts].sum())
    return {"alerts": int(alerts.sum()), "true_positives": tp, "false_positives": fp,
            "missed_frauds": int((fraud & ~alerts).sum()),
            "true_negatives": int((legitimate & ~alerts).sum()),
            "precision": tp / int(alerts.sum()) if alerts.any() else 0.0,
            "recall": tp / int(fraud.sum()) if fraud.any() else 0.0,
            "caught_fraud_amount": caught_amount, "missed_fraud_amount": missed_amount,
            "false_positive_cost": float(review_cost * fp),
            "realized_cost_proxy": float(missed_amount + review_cost * fp)}


def capacity_mask(frame, k, score_column="calibrated_probability"):
    if not isinstance(k, (int, np.integer)) or k < 0 or k > len(frame):
        raise ValueError("Capacity must be an integer between zero and row count")
    ranking = alert_ranking(frame[score_column], frame.CatBoostScore, frame.TransactionID)
    alerts = np.zeros(len(frame), dtype=bool)
    alerts[ranking[:k]] = True
    return alerts


def capacity_table(policy_development, capacities=CAPACITIES, review_cost=200.0):
    """Compare prespecified counts using development labels only, including 0/all."""
    if policy_development.empty:
        raise ValueError("Policy development needs nonempty data")
    counts = sorted({0, len(policy_development), *(min(k, len(policy_development)) for k in capacities)})
    rows = []
    for k in counts:
        rows.append({"capacity": k, "alert_fraction": k / len(policy_development),
                     **cost_report(policy_development.isFraud, policy_development.TransactionAmt,
                                   capacity_mask(policy_development, k), review_cost)})
    table = pd.DataFrame(rows)
    table["is_minimum"] = np.isclose(table.realized_cost_proxy, table.realized_cost_proxy.min(), rtol=0, atol=1e-8)
    return table


def select_capacity(table):
    """Cost ties prefer fewer alerts. No evaluation labels enter selection."""
    return int(table.loc[table.is_minimum].sort_values("capacity").iloc[0].capacity)


def transfer_capacity(k, source_rows, target_rows):
    """Preserve the selected fraction, using deterministic nearest half-up rounding."""
    if source_rows < 1 or target_rows < 1 or not 0 <= k <= source_rows:
        raise ValueError("Need valid source/target sizes and capacity")
    return min(target_rows, (2 * k * target_rows + source_rows) // (2 * source_rows))


def reliability_bins(labels, probabilities):
    rows = []
    labels, p = np.asarray(labels), np.asarray(probabilities)
    groups = np.minimum((p * 10).astype(int), 9)
    for i in range(10):
        mask = groups == i
        rows.append({"bin_lower": i / 10, "bin_upper": (i + 1) / 10, "count": int(mask.sum()),
                     "mean_probability": float(p[mask].mean()) if mask.any() else np.nan,
                     "observed_fraud_rate": float(labels[mask].mean()) if mask.any() else np.nan})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-cost", type=float, default=200.0)
    parser.add_argument("--fusion-dir", type=Path, default=FUSION)
    parser.add_argument("--anomaly-dir", type=Path, default=ANOMALY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not np.isfinite(args.review_cost) or args.review_cost < 0:
        parser.error("Review cost must be finite and nonnegative")
    start = time.perf_counter()
    _, frame, audit = load_frozen_inputs(args.fusion_dir, args.anomaly_dir)
    parts = three_way_partition(frame)
    calibration, development, evaluation = parts.values()
    # Verify that the final block is exactly Session 18's evaluation IDs/labels.
    saved_evaluation = pd.read_csv(args.fusion_dir / "evaluation_predictions.csv")
    np.testing.assert_allclose(aligned_saved_scores(evaluation, saved_evaluation, "catboost"),
                               evaluation.CatBoostScore, rtol=1e-9, atol=1e-12)
    calibrator = fit_calibrator(calibration)
    frame["calibrated_probability"] = calibrator.predict_proba(score_logits(frame.CatBoostScore))[:, 1]
    parts = three_way_partition(frame)
    calibration, development, evaluation = parts.values()
    # Invalid costs must stop policy evaluation rather than silently impute money.
    for part in parts.values():
        cost_report(part.isFraud, part.TransactionAmt, np.zeros(len(part), dtype=bool), args.review_cost)
    candidates = capacity_table(development, review_cost=args.review_cost)
    chosen_k = select_capacity(candidates)
    eval_k = transfer_capacity(chosen_k, len(development), len(evaluation))
    selected_alerts = capacity_mask(evaluation, eval_k)
    policies = []
    for name, alerts in [
        ("selected_alert_fraction", selected_alerts),
        ("alert_none", np.zeros(len(evaluation), dtype=bool)),
        ("alert_all", np.ones(len(evaluation), dtype=bool)),
    ]:
        policies.append({"policy": name, **cost_report(evaluation.isFraud, evaluation.TransactionAmt,
                                                      alerts, args.review_cost)})
    probability_metrics, bins = [], []
    for name, column in [("raw_catboost", "CatBoostScore"), ("sigmoid_calibrated", "calibrated_probability")]:
        probability_metrics.append({"score_type": name,
            **metrics(evaluation.isFraud.to_numpy(), evaluation[column].to_numpy(), evaluation.TransactionID.to_numpy()),
            "brier_score": float(brier_score_loss(evaluation.isFraud, evaluation[column]))})
        bins.extend({"score_type": name, **row} for row in reliability_bins(evaluation.isFraud, evaluation[column]))
    # A separate, predeclared K=100 comparison addresses the Session 18 tradeoff.
    # It does not select or revise the frozen alert fraction above.
    fixed_capacity = []
    for name in ["catboost", "catboost_autoencoder"]:
        scores = aligned_saved_scores(evaluation, saved_evaluation, name)
        ranked = alert_ranking(scores, scores, evaluation.TransactionID)
        alerts = np.zeros(len(evaluation), dtype=bool)
        alerts[ranked[:min(100, len(evaluation))]] = True
        fixed_capacity.append({"model": name, "fixed_capacity": min(100, len(evaluation)),
                               **cost_report(evaluation.isFraud, evaluation.TransactionAmt, alerts, args.review_cost)})
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, args.output / "sigmoid_calibrator.joblib")
    candidates.to_csv(args.output / "policy_development_costs.csv", index=False)
    pd.DataFrame(probability_metrics).to_csv(args.output / "probability_metrics.csv", index=False)
    pd.DataFrame(policies).to_csv(args.output / "evaluation_policy_costs.csv", index=False)
    pd.DataFrame(fixed_capacity).to_csv(args.output / "fixed_top100_model_costs.csv", index=False)
    pd.DataFrame(bins).to_csv(args.output / "reliability_bins.csv", index=False)
    roles = np.concatenate([np.full(len(part), name) for name, part in parts.items()])
    frame["role"] = roles
    frame[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "role", "CatBoostScore",
           "calibrated_probability"]].to_csv(args.output / "development_scores.csv", index=False)
    decisions = evaluation[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "CatBoostScore",
                            "calibrated_probability"]].copy()
    decisions["selected_policy_alert"] = selected_alerts
    decisions.to_csv(args.output / "evaluation_decisions.csv", index=False)
    partition_info = {name: {"rows": len(part), "frauds": int(part.isFraud.sum()),
                            "start_time": float(part.TransactionDT.min()), "end_time": float(part.TransactionDT.max())}
                      for name, part in parts.items()}
    write_json(args.output / "policy.json", {
        "selected_development_k": chosen_k, "development_rows": len(development),
        "selected_alert_fraction": chosen_k / len(development), "evaluation_k": eval_k,
        "evaluation_rows": len(evaluation), "tie_break": "minimum cost, then fewer alerts",
        "transfer": "equal fraction of rows; nearest half-up rounding; not daily capacity",
        "review_cost": args.review_cost, "units": "illustrative compatible dataset-amount units, not asserted currency",
    })
    write_json(args.output / "config.json", {
        "partitions": partition_info, "prespecified_capacities": CAPACITIES,
        "extra_baselines": "zero alerts and all alerts included as policy candidates",
        "calibrator": {"input": "logit of clipped frozen CatBoost probability", "clip_epsilon": 1e-12,
                       "method": "LogisticRegression sigmoid", "C": 1.0, "max_iter": 1000,
                       "slope": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])},
        "cost": "review_cost * false_positives + sum(missed fraud amounts); TP/TN cost zero",
        "review_cost": args.review_cost, "audit": audit, "test_access": False,
        "experiment": "reused development validation; no model refit, test tuning or deployment claim",
    })
    write_json(args.output / "run_summary.json", {"status": "PASS", "runtime_seconds": time.perf_counter() - start,
                                                 "test_access": False, "base_model_refit": False})
    print(json.dumps(partition_info, indent=2))
    print("\nEVALUATION PROBABILITY METRICS\n" + pd.DataFrame(probability_metrics).to_string(index=False))
    print(f"\nSelected capacity: {chosen_k}/{len(development)} -> {eval_k}/{len(evaluation)} evaluation alerts")
    print(pd.DataFrame(policies).to_string(index=False))
    print("\nFIXED TOP-100 COMPARISON\n" + pd.DataFrame(fixed_capacity).to_string(index=False))
    print("\nCosts are illustrative dataset-amount units. These are development results; test data was not opened.")


if __name__ == "__main__":
    main()
