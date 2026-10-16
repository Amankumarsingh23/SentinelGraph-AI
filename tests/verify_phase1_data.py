"""Optional real-data regression: python tests/verify_phase1_data.py.

Reads artifacts without training or rewriting datasets. Writes only a small
verification report under outputs/. Unit tests remain independent of this data.
"""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ["TransactionID", "TransactionDT", "TransactionAmt", "card1", "isFraud"]
FEATURES = ["card1_prior_count", "card1_prior_mean_amount",
            "card1_time_since_previous", "card1_amount_ratio", "card1_missing"]


def main():
    backup = ROOT / "outputs/phase1_before_repair"
    hashes = backup / "split_hashes.json"
    if hashes.exists():
        for relative, expected in json.loads(hashes.read_text()).items():
            with (ROOT / relative).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            assert actual == expected, f"Original split changed: {relative}"
    raw = pd.read_csv(ROOT / "data/raw/train_transaction.csv", usecols=BASE)[BASE]
    raw = raw.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    assert raw.TransactionID.is_unique
    splits, features, changes = [], [], {}
    for name in ["train", "validation", "test"]:
        split = pd.read_csv(ROOT / f"data/processed/{name}.csv", usecols=BASE)[BASE]
        feature = pd.read_csv(ROOT / f"data/processed/{name}_features.csv", usecols=BASE + FEATURES)
        pd.testing.assert_frame_equal(split, feature[BASE], check_dtype=False)
        splits.append(split)
        features.append(feature[FEATURES])
        previous = backup / f"{name}_features.csv"
        if previous.exists():
            old = pd.read_csv(previous, usecols=FEATURES)[FEATURES]
            different = ~np.isclose(old.to_numpy(), feature[FEATURES].to_numpy(),
                                    rtol=1e-9, atol=1e-8, equal_nan=True)
            changes[name] = int(different.any(axis=1).sum())
    pd.testing.assert_frame_equal(pd.concat(splits, ignore_index=True), raw, check_dtype=False)
    assert splits[0].TransactionDT.max() < splits[1].TransactionDT.min()
    assert splits[1].TransactionDT.max() < splits[2].TransactionDT.min()
    # Independent sequential oracle: read a whole timestamp batch, emit features
    # from previous state, and only then update that state. No cumulative pandas
    # calculations are shared with the implementation under test.
    expected = np.full((len(raw), 5), np.nan)
    expected[:, 4] = 1
    valid = raw.loc[np.isfinite(raw.card1)].sort_values(["card1", "TransactionDT"])
    rows = valid[["card1", "TransactionDT", "TransactionAmt"]].itertuples(name=None)
    entity = None
    for (card, time), batch in itertools.groupby(rows, key=lambda row: (row[1], row[2])):
        batch = list(batch)
        if card != entity:
            entity, count, observed, total, previous_time = card, 0, 0, 0., None
        mean = total / observed if observed else np.nan
        gap = time - previous_time if previous_time is not None else np.nan
        for index, _, _, amount in batch:
            ratio = amount / mean if np.isfinite(amount) and np.isfinite(mean) and mean != 0 else np.nan
            expected[index] = [count, mean, gap, ratio, 0]
        count += len(batch)
        amounts = [row[3] for row in batch if np.isfinite(row[3])]
        observed += len(amounts)
        total += sum(amounts)
        previous_time = time
    actual = pd.concat(features, ignore_index=True).to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8, equal_nan=True)
    assert not np.isinf(actual).any()
    # Repair must preserve the historical experiment's costs, despite new labels
    # for the cost field and the added baseline-policy rows.
    old_costs = backup / "threshold_costs.csv"
    if old_costs.exists():
        old = pd.read_csv(old_costs)
        new = pd.read_csv(ROOT / "outputs/evaluation/threshold_costs.csv")
        new = new.loc[new.policy == "threshold"].reset_index(drop=True)
        np.testing.assert_allclose(old.expected_cost_proxy, new.realized_cost_proxy)
        np.testing.assert_array_equal(old.false_positives, new.false_positives)
        np.testing.assert_array_equal(old.missed_frauds, new.missed_frauds)
    report = {"status": "PASS", "rows": len(raw), "split_rows": [len(s) for s in splits],
              "original_split_hashes_checked": hashes.exists(),
              "independent_feature_oracle": "PASS", "changed_feature_rows": changes}
    target = ROOT / "outputs/phase1_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
