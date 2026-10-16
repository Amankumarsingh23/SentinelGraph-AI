"""Only exact transaction/model/input matches may display SHAP attributions."""

import json
import math
from pathlib import Path

from scipy.special import expit

from api.model import CONTRACT_VERSION, EXPECTED_MODEL_SHA256, FEATURE_COLUMNS
from dashboard.evidence import SHAP_PATH


def load_shap_export(path=SHAP_PATH):
    payload = json.loads(Path(path).read_text())
    if (payload["model_sha256"] != EXPECTED_MODEL_SHA256 or payload["model_name"] != "AllFive"
            or payload["feature_contract_version"] != CONTRACT_VERSION
            or payload["feature_columns"] != FEATURE_COLUMNS or payload["output_scale"] != "raw_log_odds"):
        raise ValueError("SHAP export model, feature contract or output scale differs")
    rows = payload["records"]
    lookup = {row["transaction_id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("SHAP export contains duplicate TransactionIDs")
    return payload, lookup


def checked_shap(evidence, payload, lookup):
    if (payload["model_sha256"] != evidence["model_sha256"]
            or payload["feature_contract_version"] != evidence["feature_contract_version"]
            or payload["sources"] != evidence["provenance"]["sources"]):
        raise ValueError("SHAP export provenance does not match the selected evidence")
    row = lookup.get(evidence["transaction_id"])
    if row is None:
        return None
    if (row["transaction_id"] != evidence["transaction_id"] or row["input_sha256"] != evidence["input_sha256"]
            or row["model_sha256"] != evidence["model_sha256"] or row["feature_contract_version"] != CONTRACT_VERSION
            or row["output_scale"] != "raw_log_odds" or set(row["feature_contributions"]) != set(FEATURE_COLUMNS)):
        raise ValueError("SHAP record ID, input, model or contract differs from the selection")
    values = [row["base_value"], row["raw_output"], row["model_score"], *row["feature_contributions"].values()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("SHAP explanation contains nonfinite values")
    total = row["base_value"] + sum(row["feature_contributions"].values())
    if (not math.isclose(total, row["raw_output"], rel_tol=1e-9, abs_tol=1e-8)
            or not math.isclose(float(expit(total)), evidence["model_score"], rel_tol=1e-9, abs_tol=1e-10)
            or not math.isclose(row["model_score"], evidence["model_score"], rel_tol=1e-9, abs_tol=1e-10)):
        raise ValueError("SHAP contributions do not reconstruct the selected saved model score")
    return row
