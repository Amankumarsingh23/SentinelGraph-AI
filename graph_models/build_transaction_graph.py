"""Session 12: descriptive bipartite graph of training transactions/card1 proxies.

This completed sample graph is not a per-event historical snapshot. Its degrees
must not be used as past-time prediction features. No fraud labels are loaded.
"""

import argparse
import math
from pathlib import Path

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/train.csv"
COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt", "card1"]


def build_transaction_graph(frame):
    """Preserve missing-proxy transactions as isolated transaction nodes.

    Typed tuple identifiers prevent collisions between transaction IDs and
    card-related proxy values. A card1 proxy is not a verified cardholder.
    """
    if frame.TransactionID.isna().any() or not frame.TransactionID.is_unique:
        raise ValueError("TransactionIDs must be nonmissing and unique")
    if not frame.TransactionDT.map(math.isfinite).all():
        raise ValueError("TransactionDT must be finite")
    graph = nx.Graph()
    for row in frame[COLUMNS].itertuples(index=False):
        transaction = ("transaction", int(row.TransactionID))
        amount = float(row.TransactionAmt)
        graph.add_node(
            transaction, node_type="transaction", bipartite=0,
            transaction_time=float(row.TransactionDT),
            amount=amount if math.isfinite(amount) else None,
        )
        if pd.isna(row.card1) or not math.isfinite(row.card1):
            continue
        # IEEE-CIS card1 values are numerical anonymized categories, not measures.
        card = float(row.card1)
        category = str(int(card)) if card.is_integer() else str(card)
        proxy = ("card1_proxy", category)
        graph.add_node(proxy, node_type="card1_proxy", bipartite=1)
        graph.add_edge(transaction, proxy, relation="has_card1_proxy")
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000,
                        help="Number of earliest training transactions (default: 2000)")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    frame = pd.read_csv(DATA, usecols=COLUMNS)
    frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="stable").head(args.limit)
    if frame.empty:
        raise ValueError("Training data contains no transactions")
    graph = build_transaction_graph(frame)
    transactions = [node for node, data in graph.nodes(data=True)
                    if data["node_type"] == "transaction"]
    proxies = [node for node, data in graph.nodes(data=True)
               if data["node_type"] == "card1_proxy"]
    component_sizes = sorted((len(nodes) for nodes in nx.connected_components(graph)), reverse=True)
    print("Descriptive IEEE-CIS training-sample graph")
    print("Selected transactions:", len(frame))
    print("TransactionDT range:", frame.TransactionDT.min(), "to", frame.TransactionDT.max())
    print("Number of nodes:", graph.number_of_nodes())
    print("Transaction nodes:", len(transactions))
    print("Anonymized card1 proxy nodes:", len(proxies))
    print("Number of edges:", graph.number_of_edges())
    print("Is bipartite:", nx.is_bipartite(graph))
    print("Transactions without a usable card1:", sum(graph.degree(node) == 0 for node in transactions))
    print("Connected components:", len(component_sizes))
    print("Largest component size:", component_sizes[0])
    print("\nFive highest-degree card1 proxies (selected sample only):")
    ranked = sorted(proxies, key=lambda node: (-graph.degree(node), node))[:5]
    for node in ranked:
        print(f"{node}: degree={graph.degree(node)}")
    print("\ncard1 is an anonymized grouping proxy, not a verified customer ID.")
    print("Completed-sample degrees are descriptive; they are not past-time prediction features.")


if __name__ == "__main__":
    main()
