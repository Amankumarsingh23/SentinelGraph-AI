"""Hand-calculated component discovery, provenance and label separation checks."""

import unittest
import networkx as nx
import numpy as np
import pandas as pd

from graph_models.discover_candidate_clusters import discover_components, evaluate_components


def sample():
    return pd.DataFrame({
        "TransactionID": np.arange(1, 9), "TransactionDT": np.arange(10, 90, 10),
        "card1": [1, 1, 2, 2, 3, 4, 5, 6],
        "addr1": [10, 10, 10, 20, 20, 30, 30, 40],
        "isFraud": [0, 1, 0, 1, 0, 1, 0, 1],
    })


class CandidateClusterTests(unittest.TestCase):
    def test_hand_calculated_components_and_fraud_enrichment(self):
        frame = sample()
        graph, components, members, edges, degrees = discover_components(frame)
        self.assertTrue(nx.is_bipartite(graph))
        self.assertEqual(graph.number_of_nodes(), 8)
        self.assertEqual(graph.number_of_edges(), 6)
        self.assertEqual(components.transaction_count.tolist(), [5, 2])
        self.assertEqual(components.card_proxy_count.tolist(), [3, 2])
        self.assertEqual(components.address_proxy_count.tolist(), [2, 1])
        self.assertEqual(degrees.distinct_card_count.tolist(), [2, 2, 2, 1])
        self.assertEqual(len(members), 7)
        report, _ = evaluate_components(components, members, frame)
        np.testing.assert_allclose(report.fraud_rate, [.4, .5])
        np.testing.assert_allclose(report.fraud_enrichment, [.8, 1.0])
        self.assertEqual(edges.transaction_count.sum(), len(members))

    def test_labels_cannot_change_graph_membership_or_rank(self):
        frame = sample()
        original = discover_components(frame)
        frame["isFraud"] = 1 - frame.isFraud
        changed = discover_components(frame)
        self.assertEqual(dict(original[0].nodes(data=True)), dict(changed[0].nodes(data=True)))
        self.assertEqual(list(original[0].edges(data=True)), list(changed[0].edges(data=True)))
        for before, after in zip(original[1:], changed[1:]):
            pd.testing.assert_frame_equal(before, after)
        without_labels = discover_components(frame.drop(columns="isFraud"))
        pd.testing.assert_frame_equal(original[1], without_labels[1])

    def test_address_filter_uses_distinct_cards_and_own_pair(self):
        frame = sample()
        # Card 1 also has an excluded address. Mapping card -> component must not
        # accidentally assign this excluded transaction to the included cluster.
        frame = pd.concat([frame, pd.DataFrame({
            "TransactionID": [9], "TransactionDT": [90], "card1": [1],
            "addr1": [50], "isFraud": [1],
        })], ignore_index=True)
        _, _, members, _, degrees = discover_components(frame)
        self.assertNotIn(9, members.TransactionID.tolist())
        self.assertEqual(degrees.set_index("addr_key").loc["10", "distinct_card_count"], 2)
        frame.loc[4, "addr1"] = 10  # addr 10 now has three distinct cards.
        _, _, members, _, _ = discover_components(frame, min_cards=2, max_cards=2)
        self.assertEqual(members.TransactionID.tolist(), [6, 7])

    def test_missing_nonfinite_keys_do_not_create_relationships(self):
        frame = sample().astype({"card1": float, "addr1": float})
        frame.loc[0, "card1"] = np.nan
        frame.loc[1, "card1"] = np.inf
        frame.loc[5, "addr1"] = -np.inf
        _, _, members, _, _ = discover_components(frame)
        self.assertTrue(set(members.TransactionID).isdisjoint({1, 2, 6}))
        self.assertFalse(members[["card_key", "addr_key"]].isna().any().any())

    def test_edge_evidence_and_typed_identifiers(self):
        graph, _, _, edges, _ = discover_components(sample())
        edge = edges.loc[(edges.card_key == "1") & (edges.addr_key == "10")].iloc[0]
        self.assertEqual((edge.transaction_count, edge.first_seen, edge.last_seen), (2, 10, 20))
        frame = sample()
        frame.loc[0:1, "card1"] = 10
        graph, _, _, _, _ = discover_components(frame)
        self.assertIn(("card1_proxy", "10"), graph)
        self.assertIn(("addr1_proxy", "10"), graph)
        self.assertEqual(nx.number_of_selfloops(graph), 0)

    def test_row_order_does_not_change_ids_or_reports(self):
        frame = sample()
        original = discover_components(frame)
        changed = discover_components(frame.sample(frac=1, random_state=42))
        for before, after in zip(original[1:], changed[1:]):
            pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
        report, members = evaluate_components(original[1], original[2], frame.sample(frac=1, random_state=3))
        self.assertEqual(members.TransactionID.tolist(), list(range(1, 8)))
        self.assertEqual(report.fraud_count.tolist(), [2, 1])

    def test_no_eligible_components_and_zero_fraud_baseline(self):
        frame = sample()
        graph, components, members, edges, _ = discover_components(frame, min_cards=9, max_cards=10)
        self.assertEqual(graph.number_of_nodes(), 0)
        report, _ = evaluate_components(components, members, frame)
        self.assertEqual(len(report), 0)
        self.assertEqual(len(edges), 0)
        _, components, members, _, _ = discover_components(frame)
        frame["isFraud"] = 0
        report, _ = evaluate_components(components, members, frame)
        self.assertTrue(report.fraud_enrichment.isna().all())
        self.assertFalse(np.isinf(report.fraud_enrichment).any())

    def test_invalid_ids_times_labels_and_bounds(self):
        frame = sample()
        with self.assertRaises(ValueError):
            discover_components(frame, min_cards=10, max_cards=2)
        frame.loc[0, "TransactionID"] = 2
        with self.assertRaises(ValueError):
            discover_components(frame)
        frame = sample().astype({"TransactionDT": float})
        frame.loc[0, "TransactionDT"] = np.inf
        with self.assertRaises(ValueError):
            discover_components(frame)
        frame = sample()
        _, components, members, _, _ = discover_components(frame)
        frame.loc[0, "isFraud"] = 2
        with self.assertRaises(ValueError):
            evaluate_components(components, members, frame)


if __name__ == "__main__":
    unittest.main()
