"""Session 16: retrospective, label-free discovery of proxy-linked components.

Connected anonymized attributes are candidate investigation evidence, not proof
of coordinated fraud. The completed sample graph is not safe for past-time scoring.
"""

import argparse
from numbers import Real
from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd
from graph_models.temporal_graph_features import observed_key

DATA = ROOT / "data/processed/train.csv"
OUTPUT = ROOT / "outputs/graph/session16"
ATTRIBUTES = ["TransactionID", "TransactionDT", "card1", "addr1"]
STRUCTURE_COLUMNS = ["component_id", "transaction_count", "card_proxy_count",
                     "address_proxy_count", "unique_pair_count", "first_seen", "last_seen"]
EDGE_COLUMNS = ["component_id", "card_key", "addr_key", "transaction_count",
                "first_seen", "last_seen"]


def proxy_key(value):
    """Canonical categorical keys; missing attributes never create shared nodes."""
    value = observed_key(value)
    if value is None:
        return None
    if isinstance(value, Real) and float(value).is_integer():
        return str(int(value))
    return str(value)


def discover_components(frame, min_cards=2, max_cards=10):
    """Discover and structurally rank components without reading any label column.

    Only transactions whose own card/address pair survives filtering receive a
    membership. Sharing an included card through an excluded address is not enough.
    """
    if min_cards < 1 or min_cards > max_cards:
        raise ValueError("Need 1 <= min_cards <= max_cards")
    attributes = frame[ATTRIBUTES].copy()
    if attributes.TransactionID.isna().any() or not attributes.TransactionID.is_unique:
        raise ValueError("TransactionIDs must be nonmissing and unique")
    attributes["TransactionDT"] = pd.to_numeric(attributes.TransactionDT, errors="raise")
    if not np.isfinite(attributes.TransactionDT.to_numpy(dtype=float)).all():
        raise ValueError("TransactionDT must be finite and nonmissing")
    attributes = attributes.sort_values(["TransactionDT", "TransactionID"], kind="stable")
    attributes["card_key"] = attributes.card1.map(proxy_key)
    attributes["addr_key"] = attributes.addr1.map(proxy_key)
    pairs = attributes.dropna(subset=["card_key", "addr_key"])
    degrees = pairs.groupby("addr_key", sort=True).card_key.nunique().rename(
        "distinct_card_count"
    ).reset_index()
    degrees["eligible"] = degrees.distinct_card_count.between(min_cards, max_cards)
    eligible_addresses = degrees.loc[degrees.eligible, "addr_key"]
    members = pairs.loc[pairs.addr_key.isin(eligible_addresses)].copy()

    # Repeated transactions form one observed pair edge with count/time evidence.
    associations = members.groupby(["card_key", "addr_key"], sort=True).agg(
        transaction_count=("TransactionID", "size"),
        first_seen=("TransactionDT", "min"), last_seen=("TransactionDT", "max"),
    ).reset_index()
    graph = nx.Graph()
    for row in associations.itertuples(index=False):
        card, address = ("card1_proxy", row.card_key), ("addr1_proxy", row.addr_key)
        graph.add_node(card, node_type="card1_proxy", bipartite=0)
        graph.add_node(address, node_type="addr1_proxy", bipartite=1)
        graph.add_edge(card, address, relation="co_observed_in_transaction",
                       transaction_count=int(row.transaction_count),
                       first_seen=row.first_seen, last_seen=row.last_seen)

    # IDs depend only on graph keys, never fraud outcomes or input row order.
    connected = sorted(tuple(sorted(nodes)) for nodes in nx.connected_components(graph))
    node_to_component = {node: f"C{index:04d}" for index, nodes in enumerate(connected, 1)
                         for node in nodes}
    card_to_component = {key: component for (kind, key), component in node_to_component.items()
                         if kind == "card1_proxy"}
    members["component_id"] = members.card_key.map(card_to_component)
    associations["component_id"] = associations.card_key.map(card_to_component)
    component_rows = []
    for component, group in members.groupby("component_id", sort=True):
        component_rows.append({
            "component_id": component, "transaction_count": len(group),
            "card_proxy_count": group.card_key.nunique(),
            "address_proxy_count": group.addr_key.nunique(),
            "unique_pair_count": int(associations.component_id.eq(component).sum()),
            "first_seen": group.TransactionDT.min(), "last_seen": group.TransactionDT.max(),
        })
    components = pd.DataFrame(component_rows, columns=STRUCTURE_COLUMNS)
    components = components.sort_values(
        ["transaction_count", "card_proxy_count", "unique_pair_count", "component_id"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    components.insert(0, "structural_rank", np.arange(1, len(components) + 1))
    return graph, components, members, associations[EDGE_COLUMNS], degrees


def evaluate_components(components, members, sample):
    """Join labels AFTER discovery; never change component membership or ranking.

    Enrichment denominator is prevalence across the entire selected sample,
    including transactions excluded from graph discovery. Undefined enrichment
    for a zero-fraud sample is NaN, not infinity or fabricated zero.
    """
    if sample.empty or not sample.TransactionID.is_unique or sample.TransactionID.isna().any():
        raise ValueError("Evaluation needs a nonempty sample with unique IDs")
    if not sample.isFraud.isin([0, 1]).all():
        raise ValueError("Evaluation labels must be binary")
    labeled = members.merge(sample[["TransactionID", "isFraud"]], on="TransactionID",
                            how="left", validate="one_to_one")
    if labeled.isFraud.isna().any():
        raise ValueError("Every membership must have an aligned evaluation label")
    fraud_counts = labeled.groupby("component_id").isFraud.sum()
    report = components.copy()
    report["fraud_count"] = report.component_id.map(fraud_counts).astype(int)
    report["fraud_rate"] = report.fraud_count / report.transaction_count
    prevalence = float(sample.isFraud.mean())
    report["fraud_enrichment"] = report.fraud_rate / prevalence if prevalence else np.nan
    return report, labeled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--min-cards", type=int, default=2)
    parser.add_argument("--max-cards", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.limit < 1 or args.min_cards < 1 or args.min_cards > args.max_cards:
        parser.error("Need positive limit and 1 <= min-cards <= max-cards")
    start = time.perf_counter()
    sample = pd.read_csv(DATA, usecols=ATTRIBUTES + ["isFraud"])
    sample = sample.sort_values(["TransactionDT", "TransactionID"], kind="stable").head(args.limit)
    graph, structure, members, edges, degrees = discover_components(
        sample, args.min_cards, args.max_cards
    )
    components, labeled_members = evaluate_components(structure, members, sample)
    args.output.mkdir(parents=True, exist_ok=True)
    components.to_csv(args.output / "component_summary.csv", index=False)
    labeled_members.to_csv(args.output / "component_memberships.csv", index=False)
    edges.to_csv(args.output / "edge_evidence.csv", index=False)
    degrees.to_csv(args.output / "address_proxy_degrees.csv", index=False)
    missing_pairs = sample.card1.map(proxy_key).isna() | sample.addr1.map(proxy_key).isna()
    summary = {
        "sample_rows": len(sample), "sample_frauds": int(sample.isFraud.sum()),
        "sample_prevalence": float(sample.isFraud.mean()),
        "sample_start_time": float(sample.TransactionDT.min()),
        "sample_end_time": float(sample.TransactionDT.max()),
        "rows_missing_usable_pair": int(missing_pairs.sum()),
        "eligible_transactions": len(members), "excluded_transactions": len(sample) - len(members),
        "eligible_frauds": int(labeled_members.isFraud.sum()),
        "eligible_prevalence": float(labeled_members.isFraud.mean()) if len(members) else None,
        "observed_address_proxies": len(degrees),
        "eligible_address_proxies": int(degrees.eligible.sum()),
        "addresses_below_minimum": int((degrees.distinct_card_count < args.min_cards).sum()),
        "addresses_above_maximum": int((degrees.distinct_card_count > args.max_cards).sum()),
        "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
        "components": len(components), "is_bipartite": nx.is_bipartite(graph),
        "min_distinct_cards": args.min_cards, "max_distinct_cards": args.max_cards,
        "ranking": "transaction_count DESC, card_proxy_count DESC, unique_pair_count DESC, component_id ASC",
        "enrichment_baseline": "all selected training transactions, including excluded rows",
        "label_use": "retrospective evaluation only, after graph construction and structural ranking",
        "test_access": False, "validation_access": False,
        "prospective_features": False, "runtime_seconds": time.perf_counter() - start,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nTOP COMPONENTS BY STRUCTURE (labels shown retrospectively):")
    print(components.head(10).to_string(index=False) if len(components) else "No eligible components.")
    print(f"\nSaved reports to {args.output}")
    print("Proxy-linked clusters are investigation candidates, not verified fraud rings.")
    print("Completed-sample components are retrospective; do not use them for earlier-event predictions.")


if __name__ == "__main__":
    main()
