"""Session 13: strictly historical degrees and associations, without fraud labels.

Counters represent a growing heterogeneous transaction/attribute graph. card1 is
an anonymized proxy, not a verified customer. Only train and validation are read.
"""

from collections import defaultdict
from itertools import groupby
import json
from numbers import Real
from pathlib import Path
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUTPUT = ROOT / "outputs/graph"
COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt", "card1",
           "ProductCD", "card4", "isFraud"]
FEATURES = [
    "graph_card_prior_degree",
    "graph_product_prior_degree",
    "graph_card_distinct_products",
    "graph_card_product_prior_count",
    "graph_new_product_for_card",
    "graph_card4_prior_degree",
    "graph_card_missing",
    "graph_product_missing",
    "graph_card4_missing",
]


def observed_key(value):
    """Missing/nonfinite attributes do not create a shared missing-value node."""
    if pd.isna(value) or (isinstance(value, Real) and not np.isfinite(value)):
        return None
    return value


def temporal_graph_features(frame):
    """Return features aligned with original input row positions and index.

    Input requires ID, finite time, card1, ProductCD and card4 only. TransactionAmt
    and isFraud are deliberately not read. For each timestamp, emit ALL features
    first and update ALL relationships second. No labels or later edges enter
    history. Missing attributes have unavailable features (NaN) and flags.
    """
    if frame.TransactionID.isna().any() or not frame.TransactionID.is_unique:
        raise ValueError("TransactionIDs must be nonmissing and unique")
    times = pd.to_numeric(frame.TransactionDT, errors="raise")
    if not np.isfinite(times.to_numpy(dtype=float)).all():
        raise ValueError("TransactionDT must be finite and nonmissing")
    ordered = frame[["TransactionID", "card1", "ProductCD", "card4"]].copy()
    ordered["TransactionDT"] = times.to_numpy()
    ordered["position"] = np.arange(len(frame))
    ordered = ordered.sort_values(["TransactionDT", "TransactionID"], kind="stable")
    features = np.full((len(frame), len(FEATURES)), np.nan)
    card_count = defaultdict(int)
    product_count = defaultdict(int)
    card4_count = defaultdict(int)
    pair_count = defaultdict(int)
    card_products = defaultdict(set)
    rows = ordered[["TransactionDT", "card1", "ProductCD", "card4", "position"]]

    # itertools.groupby streams consecutive equal-time rows after sorting. It is
    # the same timestamp-batch policy as Pandas groupby, with less frame overhead.
    for _, timestamp_rows in groupby(rows.itertuples(index=False, name=None), key=lambda row: row[0]):
        batch = [(observed_key(card), observed_key(product), observed_key(network), position)
                 for _, card, product, network, position in timestamp_rows]

        # Stage 1: read G_<t for every transaction at this time; do not update it.
        for card, product, network, position in batch:
            features[position, 6:] = [card is None, product is None, network is None]
            if card is not None:
                features[position, 0] = card_count.get(card, 0)
                features[position, 2] = len(card_products.get(card, ()))
            if product is not None:
                features[position, 1] = product_count.get(product, 0)
            if card is not None and product is not None:
                features[position, 3] = pair_count.get((card, product), 0)
                features[position, 4] = product not in card_products.get(card, ())
            if network is not None:
                features[position, 5] = card4_count.get(network, 0)

        # Stage 2: only now add this timestamp's observed edges/associations.
        for card, product, network, _ in batch:
            if card is not None:
                card_count[card] += 1
            if product is not None:
                product_count[product] += 1
            if card is not None and product is not None:
                pair_count[(card, product)] += 1
                card_products[card].add(product)
            if network is not None:
                card4_count[network] += 1

    return pd.DataFrame(features, index=frame.index, columns=FEATURES)


def main():
    start = time.perf_counter()
    parts = []
    for split in ["train", "validation"]:
        frame = pd.read_csv(DATA / f"{split}.csv", usecols=COLUMNS)[COLUMNS]
        if frame.empty or not frame.isFraud.isin([0, 1]).all():
            raise ValueError(f"{split} must be nonempty with binary evaluation labels")
        frame["split"] = split
        parts.append(frame)
    if parts[0].TransactionDT.max() >= parts[1].TransactionDT.min():
        raise ValueError("Training must strictly precede validation")
    combined = pd.concat(parts, ignore_index=True).sort_values(
        ["TransactionDT", "TransactionID"], kind="stable"
    ).reset_index(drop=True)
    # No subsampling: earlier training and validation INPUTS remain available.
    features = temporal_graph_features(combined)
    result = pd.concat([combined, features], axis=1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"rows": len(result), "test_access": False, "features": FEATURES,
              "history_policy": "strictly earlier timestamps; online validation inputs; no labels",
              "proxy": "card1 is anonymized, not a verified customer ID", "splits": {}}
    for split in ["train", "validation"]:
        subset = result.loc[result.split == split].drop(columns="split")
        target = OUTPUT / f"{split}_graph_features.csv"
        subset.to_csv(target, index=False)
        report["splits"][split] = {
            "rows": len(subset), "unique_ids": int(subset.TransactionID.nunique()),
            "start_time": float(subset.TransactionDT.min()),
            "end_time": float(subset.TransactionDT.max()),
            "missing_feature_values": {name: int(subset[name].isna().sum()) for name in FEATURES},
            "new_card_product_rows": int(subset.graph_new_product_for_card.eq(1).sum()),
        }
        print(f"\n{split.upper()}: {len(subset):,} rows -> {target}", flush=True)
        print(subset[FEATURES[:6]].describe().round(3).to_string(), flush=True)
    report["runtime_seconds"] = time.perf_counter() - start
    (OUTPUT / "temporal_graph_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"\nTemporal graph features completed in {report['runtime_seconds']:.2f}s.")
    print("Reserved test split was not opened. isFraud is retained for evaluation only.")


if __name__ == "__main__":
    main()
