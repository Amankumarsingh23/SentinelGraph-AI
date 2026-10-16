"""Audit all saved dashboard records and keyed SHAP without training or UI writes."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from api.model import load_model, prepare_features
from dashboard.evidence import (
    OUTPUT, PREDICTIONS, SHAP_PATH, deterministic_summary, file_hash,
    load_clusters, load_evidence_table, make_evidence_record,
)
from dashboard.shap_evidence import checked_shap, load_shap_export


def audit_evidence():
    table = load_evidence_table()
    payload, lookup = load_shap_export()
    if set(lookup) != set(table.TransactionID):
        raise ValueError("SHAP export does not cover exactly the displayed evaluation IDs")
    saved = pd.read_csv(PREDICTIONS).set_index("TransactionID")
    np.testing.assert_array_equal(table.model_score, saved.loc[table.TransactionID, "AllFive"])
    model = load_model()
    scores = model.classifier.predict_proba(prepare_features(table), thread_count=2)[:, 1]
    raw = model.classifier.predict(prepare_features(table), prediction_type="RawFormulaVal", thread_count=2)
    np.testing.assert_allclose(scores, table.model_score, rtol=1e-9, atol=1e-10)
    errors = []
    for position in range(len(table)):
        selected = table.iloc[[position]]
        evidence = make_evidence_record(selected, int(selected.TransactionID.iloc[0]))
        if "isFraud" in evidence or "isFraud" in table or evidence["decision"] != "not_assigned" or evidence["calibrated"] is not False:
            raise ValueError("Investigator evidence exposes an outcome or assigned policy")
        explanation = checked_shap(evidence, payload, lookup)
        np.testing.assert_allclose(explanation["raw_output"], raw[position], rtol=1e-9, atol=1e-8)
        errors.append(abs(explanation["base_value"] + sum(explanation["feature_contributions"].values()) - raw[position]))
        text = deterministic_summary(evidence)
        if "No fraud decision has been assigned" not in text or "uncalibrated" not in text:
            raise ValueError("Summary omits the score/decision limitations")
        json.dumps(evidence, allow_nan=False)
    components, members, edges, cluster_summary = load_clusters()
    overlap = int(members.TransactionID.isin(table.TransactionID).sum())
    if overlap:
        raise ValueError("Unexpected overlap between separate training-cluster and validation-evidence populations")
    report = {"status": "PASS", "evaluation_records": len(table), "verified_shap_records": len(lookup),
              "saved_scores_reproduced": len(scores), "max_raw_additivity_error": float(max(errors)),
              "max_saved_score_error": float(np.max(np.abs(scores - table.model_score))),
              "retrospective_components": len(components), "retrospective_covered_transactions": len(members),
              "retrospective_sample_rows": cluster_summary["sample_rows"], "retrospective_edges": len(edges),
              "cluster_evaluation_id_overlap": overlap, "retrospective_clusters_joined_to_evidence": False,
              "model_sha256": model.sha256, "shap_export_sha256": file_hash(SHAP_PATH),
              "sources": table.attrs["provenance"]["sources"], "test_access": False,
              "training_reexecuted": False, "llm_used": False, "decision_policy_applied": False}
    return report


def main():
    report = audit_evidence()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ["sources", "shap_export_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
