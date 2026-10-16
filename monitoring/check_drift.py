"""Session 23: retrospective drift on training/validation inputs, without labels."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from api.model import CONTRACT_VERSION, FEATURE_COLUMNS, PREPROCESSING, load_model, prepare_features
from monitoring.drift_metrics import (
    EPS, bin_counts, decode_edges, distribution_summary,
    encode_edges, reference_edges, smoothed_proportions,
)

DATA = ROOT / "data/processed"
GRAPH = ROOT / "outputs/graph"
OUTPUT = ROOT / "outputs/monitoring/session23"
FEATURES = FEATURE_COLUMNS.copy()


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def align_features(raw, graph):
    """Require an exact one-to-one ID population and matching original times."""
    raw = raw[["TransactionID", "TransactionDT", "TransactionAmt"]].copy()
    graph = graph[["TransactionID", "TransactionDT", *FEATURES[1:]]].copy()
    for name, frame in [("raw", raw), ("graph", graph)]:
        if frame.empty or frame.TransactionID.isna().any() or not frame.TransactionID.is_unique:
            raise ValueError(f"{name} IDs must be nonempty, unique and nonmissing")
        if not np.isfinite(frame.TransactionDT.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} timestamps must be finite")
    if len(raw) != len(graph) or not raw.TransactionID.isin(graph.TransactionID).all():
        raise ValueError("Raw and graph inputs must contain the same TransactionIDs")
    merged = raw.merge(graph, on="TransactionID", how="left", validate="one_to_one", suffixes=("", "_graph"))
    if not merged.TransactionDT.eq(merged.TransactionDT_graph).all():
        raise ValueError("Graph timestamps differ from original transactions")
    merged[FEATURES] = merged[FEATURES].apply(pd.to_numeric, errors="raise")
    return merged.drop(columns="TransactionDT_graph").sort_values(
        ["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


def load_features(split, data=DATA, graph=GRAPH):
    if split not in ("train", "validation"):
        raise ValueError("Monitoring permits only train and validation; reserved test is forbidden")
    raw = pd.read_csv(Path(data) / f"{split}.csv", usecols=["TransactionID", "TransactionDT", "TransactionAmt"])
    features = pd.read_csv(Path(graph) / f"{split}_graph_features.csv",
                           usecols=["TransactionID", "TransactionDT", *FEATURES[1:]])
    return align_features(raw, features)


def select_windows(train, validation, reference_rows, current_rows):
    if min(reference_rows, current_rows) < 1 or len(train) < reference_rows or len(validation) < current_rows:
        raise ValueError("Need positive requested window sizes and enough rows in both splits")
    if train.TransactionDT.max() >= validation.TransactionDT.min() or train.TransactionID.isin(validation.TransactionID).any():
        raise ValueError("Training must strictly precede validation with disjoint IDs")
    return train.tail(reference_rows).copy(), validation.head(current_rows).copy()


def window_hash(frame):
    # Explicitly serialize ordered IDs, timestamps and raw numerical values.
    content = frame[["TransactionID", "TransactionDT", *FEATURES]].to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compare_distribution(name, reference, current, edges, epsilon):
    expected = smoothed_proportions(reference, edges, epsilon)
    actual = smoothed_proportions(current, edges, epsilon)
    contributions = (actual - expected) * np.log(actual / expected)
    ref_counts, cur_counts = bin_counts(reference, edges), bin_counts(current, edges)
    detail = []
    serialized = encode_edges(edges)
    for index, (ref_count, cur_count) in enumerate(zip(ref_counts, cur_counts)):
        missing = index == len(ref_counts) - 1
        detail.append({"feature": name, "bin": index, "kind": "missing_or_nonfinite" if missing else "numeric",
                       "left": None if missing else serialized[index], "right": None if missing else serialized[index + 1],
                       "reference_count": int(ref_count), "current_count": int(cur_count),
                       "reference_proportion": float(expected[index]), "current_proportion": float(actual[index]),
                       "psi_contribution": float(contributions[index])})
    ref_stats, cur_stats = distribution_summary(reference), distribution_summary(current)
    summary = {"feature": name, "psi": float(contributions.sum()), "numeric_bins": len(edges) - 1,
               **{f"reference_{key}": value for key, value in ref_stats.items()},
               **{f"current_{key}": value for key, value in cur_stats.items()},
               "missing_rate_delta": cur_stats["missing_rate"] - ref_stats["missing_rate"],
               "missing_or_nonfinite_rate_delta": cur_stats["missing_or_nonfinite_rate"] - ref_stats["missing_or_nonfinite_rate"],
               "empty_reference_numeric_bins": int((ref_counts[:-1] == 0).sum())}
    return summary, detail


def run_monitoring(reference_rows=10000, current_rows=10000, n_bins=10, epsilon=EPS,
                   output=OUTPUT, reference_bins=None, data=DATA, graph=GRAPH):
    # Loading columns explicitly avoids even parsing fraud labels.
    reference, current = select_windows(load_features("train", data, graph), load_features("validation", data, graph),
                                        reference_rows, current_rows)
    model = load_model()
    reference_scores = model.classifier.predict_proba(prepare_features(reference), thread_count=2)[:, 1]
    current_scores = model.classifier.predict_proba(prepare_features(current), thread_count=2)[:, 1]
    for scores in (reference_scores, current_scores):
        if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
            raise ValueError("Frozen model produced invalid scores")
    ref_hash = window_hash(reference)
    if reference_bins is None:
        edges = {name: reference_edges(reference[name], n_bins) for name in FEATURES}
        edges["risk_score"] = reference_edges(reference_scores, n_bins)
        baseline = {"schema_version": 1, "reference_split": "train", "reference_rows": len(reference),
                    "reference_window_sha256": ref_hash, "model_sha256": model.sha256,
                    "feature_contract_version": CONTRACT_VERSION, "nominal_quantile_bins": n_bins,
                    "quantile_method": "linear; duplicate cuts collapsed; special constant/missing handling",
                    "epsilon": epsilon, "smoothing": "(raw proportion + epsilon) / (1 + categories * epsilon)",
                    "intervals": "numpy.histogram: left closed/right open, final right edge included; separate missing/nonfinite category",
                    "edges": {name: encode_edges(value) for name, value in edges.items()}}
    else:
        baseline = json.loads(Path(reference_bins).read_text())
        if (baseline["schema_version"] != 1 or baseline["reference_split"] != "train"
                or baseline["reference_rows"] != len(reference) or baseline["reference_window_sha256"] != ref_hash
                or baseline["model_sha256"] != model.sha256 or baseline["feature_contract_version"] != CONTRACT_VERSION
                or baseline["epsilon"] != epsilon or baseline["nominal_quantile_bins"] != n_bins
                or set(baseline["edges"]) != {*FEATURES, "risk_score"}):
            raise ValueError("Saved bins do not match this reference, model or monitoring configuration")
        edges = {name: decode_edges(value) for name, value in baseline["edges"].items()}
        # Recompute only from the designated reference to detect corrupt baselines;
        # the current window never participates in fitting or validating cuts.
        for name, ref in [*((name, reference[name]) for name in FEATURES), ("risk_score", reference_scores)]:
            if not np.array_equal(edges[name], reference_edges(ref, n_bins)):
                raise ValueError("Saved cuts differ from the designated training reference")
    summaries, details = [], []
    for name in FEATURES:
        summary, bins = compare_distribution(name, reference[name], current[name], edges[name], epsilon)
        summaries.append(summary)
        details.extend(bins)
    score_summary, bins = compare_distribution("risk_score", reference_scores, current_scores, edges["risk_score"], epsilon)
    details.extend(bins)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / "feature_drift.csv", index=False)
    pd.DataFrame(details).to_csv(output / "bin_proportions.csv", index=False)
    write_json(output / "reference_bins.json", baseline)
    write_json(output / "prediction_drift.json", {**score_summary, "model_sha256": model.sha256,
               "score_type": "uncalibrated_model_score", "reference_scores_in_sample": True,
               "scope": "retrospective score-distribution comparison, not performance or fraud prevalence"})
    windows = []
    for role, frame, scores in [("reference", reference, reference_scores), ("current", current, current_scores)]:
        window = frame.copy()
        window.insert(0, "role", role)
        window["risk_score"] = scores
        windows.append(window)
    pd.concat(windows, ignore_index=True).to_csv(output / "window_features.csv", index=False)
    input_paths = [Path(data) / f"{split}.csv" for split in ("train", "validation")]
    input_paths += [Path(graph) / f"{split}_graph_features.csv" for split in ("train", "validation")]
    summary = {"status": "PASS", "reference": {"split": "train", "selection": "latest chronological rows",
               "rows": len(reference), "time_min": float(reference.TransactionDT.min()),
               "time_max": float(reference.TransactionDT.max()), "window_sha256": ref_hash},
               "current": {"split": "validation", "selection": "earliest chronological rows", "rows": len(current),
               "time_min": float(current.TransactionDT.min()), "time_max": float(current.TransactionDT.max()),
               "window_sha256": window_hash(current)}, "model_sha256": model.sha256,
               "feature_contract_version": CONTRACT_VERSION, "feature_columns": FEATURES, "preprocessing": PREPROCESSING,
               "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "catboost": version("catboost")},
               "reference_bins_reused": reference_bins is not None,
               "input_files": {str(path.resolve()): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in input_paths},
               "labels_loaded": False, "test_access": False, "training_reexecuted": False,
               "automatic_retraining": False, "serving_model_changed": False,
               "scope": "offline retrospective input/score drift; no concept-drift or performance claim",
               "limitations": ["Training reference was used to fit this model; score reference is in-sample",
                               "Historical graph counters grow over time; changes can reflect expected history accumulation",
                               "PSI depends on cuts, smoothing and sample sizes; no universal alarm threshold is applied",
                               "Validation has been reused for development; no real label-arrival history is available"]}
    write_json(output / "run_summary.json", summary)
    print(pd.DataFrame(summaries)[["feature", "psi", "reference_missing_rate", "current_missing_rate"]].to_string(index=False))
    print(f"Prediction PSI: {score_summary['psi']:.8f}")
    print(f"Reference/current rows: {len(reference):,} / {len(current):,}; reports: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-rows", type=int, default=10000)
    parser.add_argument("--current-rows", type=int, default=10000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=EPS)
    parser.add_argument("--reference-bins", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.reference_rows, args.current_rows) < 1 or args.bins < 2 or not np.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("Use positive window sizes/finite epsilon, and at least two bins")
    run_monitoring(args.reference_rows, args.current_rows, args.bins, args.epsilon,
                   args.output, args.reference_bins)


if __name__ == "__main__":
    main()
