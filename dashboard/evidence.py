"""Match saved evidence by ID and time, without exposing retrospective labels."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from api.model import CONTRACT_VERSION, EXPECTED_MODEL_SHA256, FEATURE_COLUMNS, MODEL_PATH, PREPROCESSING

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs/evaluation/session20/predictions.csv"
CONFIG = ROOT / "outputs/evaluation/session20/config.json"
VALIDATION = ROOT / "data/processed/validation.csv"
GRAPH = ROOT / "outputs/graph/validation_graph_features.csv"
OUTPUT = ROOT / "outputs/dashboard/session24"
SHAP_PATH = OUTPUT / "shap_export.json"
CLUSTERS = ROOT / "outputs/graph/session16"
GRAPH_FEATURES = FEATURE_COLUMNS[1:]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_signature(paths):
    return tuple((str(path), Path(path).stat().st_mtime_ns, Path(path).stat().st_size) for path in paths)


def finite_number(value):
    return None if pd.isna(value) or not math.isfinite(float(value)) else float(value)


def align_evidence(predictions, raw, graph):
    """Labels are checked and fingerprinted internally, then removed."""
    predictions = predictions[["TransactionID", "TransactionDT", "isFraud", "AllFive"]].copy()
    raw = raw[["TransactionID", "TransactionDT", "isFraud", "TransactionAmt", "card1", "ProductCD", "card4"]].copy()
    graph = graph[["TransactionID", "TransactionDT", *GRAPH_FEATURES]].copy()
    for name, frame in [("predictions", predictions), ("validation", raw), ("graph", graph)]:
        if frame.empty or frame.TransactionID.isna().any() or not frame.TransactionID.is_unique:
            raise ValueError(f"{name} IDs must be nonempty, unique and nonmissing")
        if not np.isfinite(frame.TransactionDT.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} timestamps must be finite")
    if not predictions.TransactionID.isin(raw.TransactionID).all() or not predictions.TransactionID.isin(graph.TransactionID).all():
        raise ValueError("Every predicted TransactionID needs validation and graph evidence")
    merged = predictions.merge(raw, on="TransactionID", validate="one_to_one", suffixes=("_prediction", ""))
    merged = merged.merge(graph, on="TransactionID", validate="one_to_one", suffixes=("", "_graph"))
    if not (merged.TransactionDT.eq(merged.TransactionDT_prediction).all() and merged.TransactionDT.eq(merged.TransactionDT_graph).all()):
        raise ValueError("Prediction, validation and graph timestamps must match")
    if not merged.isFraud.isin([0, 1]).all() or not merged.isFraud.eq(merged.isFraud_prediction).all():
        raise ValueError("Saved retrospective labels disagree with authoritative validation records")
    scores = merged.AllFive.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Saved model scores must be finite and between zero and one")
    merged = merged.sort_values(["TransactionDT", "TransactionID"], kind="stable")
    values = pd.util.hash_pandas_object(merged[["TransactionID", "TransactionDT", "isFraud", *FEATURE_COLUMNS]], index=False).to_numpy()
    fingerprint = hashlib.sha256(values.tobytes()).hexdigest()
    result = merged[["TransactionID", "TransactionDT", "TransactionAmt", "card1", "ProductCD", "card4",
                     *GRAPH_FEATURES, "AllFive"]].rename(columns={"AllFive": "model_score"})
    result = result.sort_values(["model_score", "TransactionID"], ascending=[False, True], kind="stable").reset_index(drop=True)
    result.attrs["ordered_input_sha256"] = fingerprint
    return result


def load_evidence_table():
    # Only the validation split is opened. Session 16 and Session 19 artifacts
    # are deliberately absent from these joins.
    predictions = pd.read_csv(PREDICTIONS)
    raw = pd.read_csv(VALIDATION, usecols=["TransactionID", "TransactionDT", "isFraud", "TransactionAmt", "card1", "ProductCD", "card4"])
    graph = pd.read_csv(GRAPH, usecols=["TransactionID", "TransactionDT", *GRAPH_FEATURES])
    table = align_evidence(predictions, raw, graph)
    config = json.loads(CONFIG.read_text())
    if config["feature_groups"]["AllFive"] != FEATURE_COLUMNS or config.get("test_access") is not False:
        raise ValueError("Session 20 feature contract or evaluation scope differs")
    if table.attrs["ordered_input_sha256"] != config["evaluation_input_hash"]:
        raise ValueError("Current evidence differs from the original Session 20 ordered evaluation inputs")
    if len(table) != config["evaluation_rows"] or file_hash(MODEL_PATH) != EXPECTED_MODEL_SHA256:
        raise ValueError("Evaluation row count or frozen model fingerprint differs")
    table.attrs["provenance"] = {
        "model_name": "AllFive", "model_sha256": EXPECTED_MODEL_SHA256,
        "feature_contract_version": CONTRACT_VERSION, "preprocessing": PREPROCESSING,
        "ordered_input_sha256": table.attrs["ordered_input_sha256"],
        "sources": {str(path.relative_to(ROOT)): file_hash(path) for path in [PREDICTIONS, CONFIG, VALIDATION, GRAPH]},
        "scope": "saved Session 20 reused development evaluation; original point-in-time Session 13 features",
        "retrospective_labels": "verified internally and excluded from investigator evidence",
    }
    return table


def make_evidence_record(table, transaction_id):
    selected = table.loc[table.TransactionID.eq(transaction_id)]
    if len(selected) != 1:
        raise ValueError("Selected TransactionID must have exactly one evidence record")
    row = selected.iloc[0]
    observed = {key: None if pd.isna(row[key]) else row[key].item() if isinstance(row[key], np.generic) else row[key]
                for key in ["card1", "ProductCD", "card4"]}
    record = {"transaction_id": int(row.TransactionID), "transaction_time": float(row.TransactionDT),
              "transaction_amount": finite_number(row.TransactionAmt), "model_name": "AllFive",
              "model_score": float(row.model_score), "model_sha256": EXPECTED_MODEL_SHA256,
              "feature_contract_version": CONTRACT_VERSION, "calibrated": False, "decision": "not_assigned",
              "historical_features": {key: finite_number(row[key]) for key in GRAPH_FEATURES},
              "observed_attributes": observed, "provenance": table.attrs.get("provenance", {}),
              "limitations": ["card1 is an anonymized proxy, not a verified customer identity",
                              "Historical counts describe strictly earlier observed events",
                              "No fraud decision policy applied; score is uncalibrated",
                              "Retrospective components are a separate research view, not inference-time evidence"]}
    record["input_sha256"] = input_fingerprint(record)
    return record


def input_fingerprint(record):
    payload = {"transaction_id": record["transaction_id"], "transaction_time": record["transaction_time"],
               "TransactionAmt": record["transaction_amount"], **record["historical_features"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def deterministic_summary(evidence):
    score = float(evidence["model_score"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Summary needs a valid recorded model score")
    features = evidence["historical_features"]
    def number(value):
        return "unavailable" if finite_number(value) is None else f"{float(value):g}"
    pair = features.get("graph_card_product_prior_count", features.get("graph_pair_prior_count"))
    return (f"Transaction {int(evidence['transaction_id'])} has an uncalibrated model score of {score:.6f}. "
            f"Recorded amount: {number(evidence.get('transaction_amount'))}. "
            f"Prior card-proxy observations: {number(features.get('graph_card_prior_degree'))}; "
            f"prior product observations: {number(features.get('graph_product_prior_degree'))}; "
            f"prior matching card-product transactions: {number(pair)}; "
            f"distinct previously observed products for the card proxy: {number(features.get('graph_card_distinct_products'))}. "
            "Unavailable values are unknown, not zero. card1 is an anonymized proxy and does not identify a verified customer. "
            "These observations do not establish fraud or causality. No fraud decision has been assigned.")


def load_clusters(directory=CLUSTERS):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    components = pd.read_csv(directory / "component_summary.csv")
    members = pd.read_csv(directory / "component_memberships.csv", dtype={"card_key": str, "addr_key": str})
    edges = pd.read_csv(directory / "edge_evidence.csv", dtype={"card_key": str, "addr_key": str})
    if (summary["test_access"] is not False or summary["validation_access"] is not False
            or summary["prospective_features"] is not False or not components.component_id.is_unique
            or not members.TransactionID.is_unique or len(members) != summary["eligible_transactions"]
            or len(components) != summary["components"] or set(members.component_id) != set(components.component_id)
            or set(edges.component_id) != set(components.component_id)):
        raise ValueError("Retrospective component scope or membership counts differ")
    for row in components.itertuples():
        group = members.loc[members.component_id.eq(row.component_id)]
        links = edges.loc[edges.component_id.eq(row.component_id)]
        pairs = set(zip(group.card_key, group.addr_key))
        if (len(group) != row.transaction_count or group.card_key.nunique() != row.card_proxy_count
                or group.addr_key.nunique() != row.address_proxy_count or len(pairs) != row.unique_pair_count
                or len(links) != row.unique_pair_count or set(zip(links.card_key, links.addr_key)) != pairs
                or int(group.isFraud.sum()) != row.fraud_count
                or not math.isclose(float(group.isFraud.mean()), row.fraud_rate, abs_tol=1e-12)
                or group.TransactionDT.min() != row.first_seen or group.TransactionDT.max() != row.last_seen):
            raise ValueError("Component records do not reproduce their retrospective summary")
        for link in links.itertuples():
            observations = group.loc[group.card_key.eq(link.card_key) & group.addr_key.eq(link.addr_key)]
            if (len(observations) != link.transaction_count or observations.TransactionDT.min() != link.first_seen
                    or observations.TransactionDT.max() != link.last_seen):
                raise ValueError("Retrospective edge counts/times disagree with observed memberships")
    return components, members.drop(columns="isFraud"), edges, summary
