"""Audit generated drift reports, reference-only cuts and serving score parity."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from api.model import CONTRACT_VERSION, load_model, prepare_features
from monitoring.check_drift import FEATURES, OUTPUT, sha256_file, window_hash, write_json
from monitoring.drift_metrics import decode_edges, reference_edges


def verify_outputs(output=OUTPUT, write_report=True):
    output = Path(output)
    summary = json.loads((output / "run_summary.json").read_text())
    baseline = json.loads((output / "reference_bins.json").read_text())
    prediction = json.loads((output / "prediction_drift.json").read_text())
    features = pd.read_csv(output / "feature_drift.csv", float_precision="round_trip")
    details = pd.read_csv(output / "bin_proportions.csv", float_precision="round_trip")
    windows = pd.read_csv(output / "window_features.csv", float_precision="round_trip")
    if (summary["status"] != "PASS" or summary["labels_loaded"] is not False
            or summary["test_access"] is not False or summary["training_reexecuted"] is not False
            or "isFraud" in windows or set(windows.role) != {"reference", "current"}):
        raise ValueError("Unexpected monitoring scope or window contents")
    reference = windows.loc[windows.role.eq("reference")].copy()
    current = windows.loc[windows.role.eq("current")].copy()
    for name, frame in [("reference", reference), ("current", current)]:
        if len(frame) != summary[name]["rows"] or not frame.TransactionID.is_unique:
            raise ValueError("Window counts/IDs differ from summary")
        if window_hash(frame) != summary[name]["window_sha256"]:
            raise ValueError("Saved feature window fingerprint differs from source selection")
    if reference.TransactionDT.max() >= current.TransactionDT.min() or reference.TransactionID.isin(current.TransactionID).any():
        raise ValueError("Reference/current windows overlap")
    if (len(features) != len(FEATURES) or features.feature.duplicated().any()
            or set(features.feature) != set(FEATURES) or set(details.feature) != {*FEATURES, "risk_score"}
            or set(baseline["edges"]) != {*FEATURES, "risk_score"}):
        raise ValueError("Report feature groups are incomplete or duplicated")
    model = load_model()
    if (baseline["reference_split"] != "train" or summary["reference"]["split"] != "train"
            or summary["current"]["split"] != "validation" or baseline["reference_window_sha256"] != window_hash(reference)
            or baseline["reference_rows"] != len(reference) or baseline["feature_contract_version"] != CONTRACT_VERSION
            or baseline["model_sha256"] != model.sha256 or summary["model_sha256"] != model.sha256
            or prediction["model_sha256"] != model.sha256):
        raise ValueError("Reference or model provenance mismatch")
    for frame in (reference, current):
        scores = model.classifier.predict_proba(prepare_features(frame), thread_count=2)[:, 1]
        np.testing.assert_allclose(scores, frame.risk_score, rtol=1e-10, atol=1e-12)
    for filename, expected in summary["input_files"].items():
        if sha256_file(filename) != expected["sha256"] or Path(filename).stat().st_size != expected["bytes"]:
            raise ValueError("An original monitoring input file changed")
    audited = []
    for name in [*FEATURES, "risk_score"]:
        edges = decode_edges(baseline["edges"][name])
        np.testing.assert_array_equal(edges, reference_edges(reference[name], baseline["nominal_quantile_bins"]))
        records = details.loc[details.feature.eq(name)].sort_values("bin")
        expected_counts = []
        for frame in (reference, current):
            values = frame[name].to_numpy(dtype=float)
            finite = np.isfinite(values)
            # Independent count/formula replay from the saved observations.
            counts = np.r_[np.histogram(values[finite], bins=edges)[0], (~finite).sum()]
            expected_counts.append(counts)
        ref_counts, cur_counts = expected_counts
        np.testing.assert_array_equal(records.reference_count, ref_counts)
        np.testing.assert_array_equal(records.current_count, cur_counts)
        eps = baseline["epsilon"]
        expected = (ref_counts / len(reference) + eps) / (1 + len(ref_counts) * eps)
        actual = (cur_counts / len(current) + eps) / (1 + len(cur_counts) * eps)
        np.testing.assert_allclose(records.reference_proportion, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(records.current_proportion, actual, rtol=1e-12, atol=1e-12)
        contribution = (actual - expected) * np.log(actual / expected)
        np.testing.assert_allclose(records.psi_contribution, contribution, rtol=1e-12, atol=1e-12)
        row = prediction if name == "risk_score" else features.set_index("feature").loc[name]
        np.testing.assert_allclose(row["psi"], contribution.sum(), rtol=1e-12, atol=1e-12)
        for role, frame in [("reference", reference), ("current", current)]:
            if row[f"{role}_rows"] != len(frame):
                raise ValueError("Report row counts differ from saved windows")
            np.testing.assert_allclose(row[f"{role}_missing_rate"], frame[name].isna().mean(), atol=1e-12)
        audited.append({"feature": name, "psi": float(contribution.sum()), "status": "PASS"})
    report = {"status": "PASS", "reference_rows": len(reference), "current_rows": len(current),
              "scores_reproduced": len(windows), "model_sha256": model.sha256, "checks": audited,
              "training_only_reference_edges": True, "labels_loaded": False,
              "test_access": False, "training_reexecuted": False}
    if write_report:
        write_json(output / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    report = verify_outputs(parser.parse_args().output)
    print(f"PASS: five feature reports, score PSI, reference-only cuts, counts, input hashes and {report['scores_reproduced']:,} saved scores")


if __name__ == "__main__":
    main()
