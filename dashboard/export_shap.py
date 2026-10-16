"""Export keyed native SHAP for the exact Session 20 AllFive evaluation model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from catboost import Pool
import numpy as np
from scipy.special import expit

from api.model import CONTRACT_VERSION, FEATURE_COLUMNS, load_model, prepare_features
from dashboard.evidence import SHAP_PATH, load_evidence_table, make_evidence_record


def export_shap(path=SHAP_PATH):
    table = load_evidence_table()
    model = load_model()
    inputs = prepare_features(table)
    shap = model.classifier.get_feature_importance(Pool(inputs), type="ShapValues", thread_count=2)
    raw = model.classifier.predict(inputs, prediction_type="RawFormulaVal", thread_count=2)
    probabilities = model.classifier.predict_proba(inputs, thread_count=2)[:, 1]
    totals = shap[:, -1] + shap[:, :-1].sum(axis=1)
    if not np.isfinite(shap).all():
        raise ValueError("SHAP values must be finite")
    np.testing.assert_allclose(totals, raw, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(expit(totals), probabilities, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(probabilities, table.model_score, rtol=1e-9, atol=1e-10)
    records = []
    # Build records once, instead of repeatedly filtering the full table.
    for i in range(len(table)):
        selected = table.iloc[[i]]
        evidence = make_evidence_record(selected, int(selected.TransactionID.iloc[0]))
        records.append({"transaction_id": evidence["transaction_id"], "input_sha256": evidence["input_sha256"],
                        "model_name": "AllFive", "model_sha256": model.sha256,
                        "feature_contract_version": CONTRACT_VERSION, "output_scale": "raw_log_odds",
                        "base_value": float(shap[i, -1]), "raw_output": float(raw[i]),
                        "model_score": float(probabilities[i]),
                        "feature_contributions": dict(zip(FEATURE_COLUMNS, shap[i, :-1].tolist()))})
    payload = {"model_name": "AllFive", "model_sha256": model.sha256, "feature_contract_version": CONTRACT_VERSION,
               "feature_columns": FEATURE_COLUMNS, "output_scale": "raw_log_odds", "sources": table.attrs["provenance"]["sources"],
               "max_additivity_error": float(np.max(np.abs(totals - raw))),
               "max_saved_score_error": float(np.max(np.abs(probabilities - table.model_score))),
               "test_access": False, "training_reexecuted": False, "causal_explanation": False, "records": records}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(records):,} keyed SHAP records: {path}")
    print(f"Maximum raw additivity error: {payload['max_additivity_error']:.3g}; saved-score error: {payload['max_saved_score_error']:.3g}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SHAP_PATH)
    export_shap(parser.parse_args().output)
