"""Calibration and threshold development on validation only; never test tuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_score, recall_score, f1_score,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/processed"
OUT = ROOT / "outputs/evaluation"
FALSE_POSITIVE_COST = 200


def chronological_midpoint(times):
    """Keep equal timestamps on the calibration side of the boundary."""
    times = np.asarray(times)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Calibration times must be finite and chronological")
    mid = len(times) // 2
    while 0 < mid < len(times) and times[mid] == times[mid - 1]:
        mid += 1
    if mid == 0 or mid == len(times):
        raise ValueError("Need two nonempty, time-separated validation portions")
    return mid


def alert_ranking(probabilities, raw_scores, transaction_ids):
    """Descending probability, descending raw score, ascending unique ID.

    No labels enter ranking. Precision@K and Recall@K use the same order.
    """
    p, raw, ids = map(np.asarray, (probabilities, raw_scores, transaction_ids))
    if not (len(p) == len(raw) == len(ids)) or len(np.unique(ids)) != len(ids):
        raise ValueError("Ranking requires aligned arrays and unique transaction IDs")
    if not np.isfinite(p).all() or not np.isfinite(raw).all():
        raise ValueError("Ranking scores must be finite")
    return np.lexsort((ids, -raw, -p))


def metric_report(labels, probabilities, raw_scores, ids):
    labels = np.asarray(labels)
    predicted = np.asarray(probabilities) >= 0.5
    k = min(100, len(labels))
    if not k:
        raise ValueError("Metrics require nonempty observations")
    selected = alert_ranking(probabilities, raw_scores, ids)[:k]
    hits = int(labels[selected].sum())
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "classification_threshold": 0.5, "alerts": int(predicted.sum()),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "k": k, "frauds_in_top_k": hits, "precision_at_k": hits / k,
        "recall_at_k": hits / int(labels.sum()) if labels.sum() else None,
    }


def threshold_costs(labels, probabilities, amounts):
    """Illustrative REALIZED cost, not validated or predicted financial loss.

    sum(amount of missed fraud) + 200 * number of legitimate transactions alerted.
    Assumes full loss on missed fraud and zero residual cost for caught fraud.
    """
    labels, probabilities, amounts = map(np.asarray, (labels, probabilities, amounts))
    if not (len(labels) == len(probabilities) == len(amounts)):
        raise ValueError("Cost inputs must be aligned")
    if not np.isfinite(amounts).all() or np.any(amounts < 0):
        raise ValueError("Cost calculation requires finite, nonnegative amounts")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("Cost calculation requires binary labels and finite scores")
    policies = [("alert_none", np.nan, np.zeros(len(labels), dtype=bool)),
                ("alert_all", np.nan, np.ones(len(labels), dtype=bool))]
    policies += [("threshold", t / 100, probabilities >= t / 100)
                 for t in range(5, 100, 5)]
    rows = []
    for policy, threshold, predicted in policies:
        missed = (labels == 1) & ~predicted
        false_alarms = (labels == 0) & predicted
        rows.append({
            "policy": policy, "threshold": threshold, "alerts": int(predicted.sum()),
            "realized_cost_proxy": float(amounts[missed].sum() + FALSE_POSITIVE_COST * false_alarms.sum()),
            "false_positives": int(false_alarms.sum()), "missed_frauds": int(missed.sum()),
        })
    result = pd.DataFrame(rows)
    result["is_minimum"] = np.isclose(
        result.realized_cost_proxy, result.realized_cost_proxy.min(), rtol=0, atol=1e-8,
    )
    return result


def main():
    # Pure metric helpers are public-testable without the optional training stack.
    from lightgbm import LGBMClassifier

    OUT.mkdir(parents=True, exist_ok=True)
    numeric = ["TransactionAmt", "TransactionDT"]
    categorical = ["ProductCD", "card4", "card6"]
    features = numeric + categorical
    columns = features + ["TransactionID", "isFraud"]
    train = pd.read_csv(DATA / "train.csv", usecols=columns).sort_values(
        ["TransactionDT", "TransactionID"]
    ).tail(100_000).copy()
    val = pd.read_csv(DATA / "validation.csv", usecols=columns).sort_values(
        ["TransactionDT", "TransactionID"]
    ).reset_index(drop=True)
    if train.TransactionDT.max() >= val.TransactionDT.min():
        raise ValueError("Training must precede validation")
    if not train.TransactionID.is_unique or not val.TransactionID.is_unique or train.TransactionID.isin(val.TransactionID).any():
        raise ValueError("Transaction IDs must be unique and separated")
    mid = chronological_midpoint(val.TransactionDT)
    for col in categorical:
        train[col] = train[col].fillna("Missing").astype(str)
        val[col] = val[col].fillna("Missing").astype(str)
        dtype = pd.CategoricalDtype(categories=pd.Index(train[col].unique()))
        train[col] = train[col].astype(dtype)
        val[col] = val[col].astype(dtype)
    model = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        random_state=42, verbosity=-1,
    )
    model.fit(train[features], train["isFraud"])
    raw_scores = model.predict_proba(val[features])[:, 1]
    cal_labels = val.isFraud.iloc[:mid].to_numpy()
    labels = val.isFraud.iloc[mid:].to_numpy()
    if len(np.unique(cal_labels)) != 2 or len(np.unique(labels)) != 2:
        raise ValueError("Both validation portions require both classes")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_scores[:mid], cal_labels)
    scores = raw_scores[mid:]
    probabilities = calibrator.predict(scores)
    ids = val.TransactionID.iloc[mid:].to_numpy()
    metrics = pd.DataFrame([
        {"score_type": "raw", **metric_report(labels, scores, scores, ids)},
        {"score_type": "calibrated", **metric_report(labels, probabilities, scores, ids)},
    ])
    metrics.to_csv(OUT / "metrics.csv", index=False)
    results = threshold_costs(labels, probabilities, val.TransactionAmt.iloc[mid:].to_numpy())
    results.to_csv(OUT / "threshold_costs.csv", index=False)
    # Development reliability bins are descriptive, not fitting inputs.
    bins = []
    for kind, p in [("raw", scores), ("calibrated", probabilities)]:
        bin_id = np.minimum((p * 10).astype(int), 9)
        for b in range(10):
            mask = bin_id == b
            bins.append({"score_type": kind, "bin_lower": b / 10, "bin_upper": (b + 1) / 10,
                         "count": int(mask.sum()),
                         "mean_probability": float(p[mask].mean()) if mask.any() else np.nan,
                         "observed_fraud_rate": float(labels[mask].mean()) if mask.any() else np.nan})
    pd.DataFrame(bins).to_csv(OUT / "calibration_bins.csv", index=False)
    summary = {
        "training_rows": len(train), "calibration_rows": mid, "development_rows": len(labels),
        "calibration_frauds": int(cal_labels.sum()), "development_frauds": int(labels.sum()),
        "calibration_end_time": int(val.TransactionDT.iloc[mid - 1]),
        "development_start_time": int(val.TransactionDT.iloc[mid]),
        "probability_min": float(probabilities.min()), "probability_max": float(probabilities.max()),
        "distinct_calibrated_probabilities": int(len(np.unique(probabilities))),
        "ranking": "probability descending, raw score descending, TransactionID ascending",
        "cost_equation": "sum(missed fraud amounts) + 200 * false positives; illustrative realized proxy, not validated financial loss",
        "minimum_cost_thresholds": results.loc[results.is_minimum & (results.policy == "threshold"), "threshold"].tolist(),
        "minimum_cost_baselines": results.loc[results.is_minimum & (results.policy != "threshold"), "policy"].tolist(),
        "test_used": False,
    }
    (OUT / "calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nDEVELOPMENT METRICS\n", metrics.to_string(index=False))
    print("\nILLUSTRATIVE REALIZED COST PROXY (not validated financial loss)\n", results.to_string(index=False))
    print("\nALL MINIMUM-COST POLICIES\n", results.loc[results.is_minimum].to_string(index=False))
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
