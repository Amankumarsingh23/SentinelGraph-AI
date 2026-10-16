"""Session 19: native CatBoost SHAP on the actual frozen Session 18 model."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from catboost import Pool
from scipy.special import expit

from deep_learning.run_behavioral import write_json
from evaluate_risk import alert_ranking
from decision_models.evaluate_decisions import (
    ANOMALY, FUSION, OUTPUT, FEATURES, load_frozen_inputs, three_way_partition,
)


def shap_attributions(model, X):
    """Return native SHAP contributions only after checking raw-score additivity."""
    if list(X.columns) != model.feature_names_:
        raise ValueError("SHAP inputs must use the exact fitted feature names/order")
    shap_values = model.get_feature_importance(Pool(X), type="ShapValues")
    contributions, base = shap_values[:, :-1], shap_values[:, -1]
    raw = model.predict(X, prediction_type="RawFormulaVal").reshape(-1)
    reconstructed = base + contributions.sum(axis=1)
    np.testing.assert_allclose(raw, reconstructed, rtol=1e-7, atol=1e-5)
    probabilities = model.predict_proba(X)[:, 1]
    np.testing.assert_allclose(probabilities, expit(reconstructed), rtol=1e-7, atol=1e-8)
    return contributions, base, raw, reconstructed, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--fusion-dir", type=Path, default=FUSION)
    parser.add_argument("--anomaly-dir", type=Path, default=ANOMALY)
    parser.add_argument("--output", type=Path, default=OUTPUT / "shap")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    model, frame, audit = load_frozen_inputs(args.fusion_dir, args.anomaly_dir)
    evaluation = three_way_partition(frame)["evaluation"]
    ranked = alert_ranking(evaluation.CatBoostScore, evaluation.CatBoostScore, evaluation.TransactionID)
    selected = evaluation.iloc[ranked[:args.count]].copy()
    X = selected[FEATURES].replace([np.inf, -np.inf], np.nan)
    contributions, baseline, raw, rebuilt, probabilities = shap_attributions(model, X)
    args.output.mkdir(parents=True, exist_ok=True)
    transactions = selected[["TransactionID", "TransactionDT", "isFraud"]].copy()
    transactions["catboost_probability"] = probabilities
    transactions["raw_log_odds"] = raw
    transactions["shap_baseline"] = baseline
    transactions["reconstructed_raw_score"] = rebuilt
    transactions["absolute_additivity_error"] = np.abs(raw - rebuilt)
    transactions.to_csv(args.output / "explained_transactions.csv", index=False)
    rows = []
    for i, (_, transaction) in enumerate(selected.iterrows()):
        order = sorted(range(len(FEATURES)), key=lambda j: (-abs(contributions[i, j]), FEATURES[j]))
        for rank, j in enumerate(order, 1):
            rows.append({"TransactionID": int(transaction.TransactionID), "attribution_rank": rank,
                         "feature": FEATURES[j], "input_value": X.iloc[i, j],
                         "shap_contribution_log_odds": contributions[i, j]})
    pd.DataFrame(rows).to_csv(args.output / "feature_attributions.csv", index=False)
    write_json(args.output / "verification.json", {
        "status": "PASS", "transactions": len(selected), "features": FEATURES,
        "max_absolute_raw_score_error": float(np.max(np.abs(raw - rebuilt))),
        "max_absolute_probability_error": float(np.max(np.abs(probabilities - expit(rebuilt)))),
        "selection": "top frozen CatBoost scores on Session 18 evaluation IDs; ID tie break; no label selection",
        "scale": "raw binary CatBoost log-odds; not calibrated/fused probability",
        "audit": audit, "causal_explanation": False, "test_access": False,
    })
    print(transactions.to_string(index=False))
    print("\nNATIVE SHAP CONTRIBUTIONS (raw log-odds):")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSHAP additive consistency and sigmoid reconstruction: PASS")
    print("Attributions explain the frozen CatBoost output; they do not establish causes or fraudulent intent.")


if __name__ == "__main__":
    main()
