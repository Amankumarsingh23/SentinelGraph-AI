"""Check graph direction, temporal isolation, batching and target alignment."""

import unittest
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from deep_learning.behavioral_data import prepare_data
from graph_models.train_temporal_gnn import TemporalGNN, build_neighborhoods


class TemporalGNNTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "TransactionID": [101, 102, 103], "TransactionDT": [10, 20, 20],
            "TransactionAmt": [100., 200., 300.], "card1": [1., 1., 1.],
            "isFraud": [0, 1, 0],
        })
        self.val = pd.DataFrame({
            "TransactionID": [104, 105], "TransactionDT": [30, 40],
            "TransactionAmt": [np.inf, 400.], "card1": [1., np.nan],
            "isFraud": [1, 0],
        })
        self.frame, self.features, self.parts, _ = prepare_data(
            self.train, self.val, train_targets=1, max_length=20
        )

    def graphs(self, split):
        return build_neighborhoods(self.frame, self.features, self.parts[split])

    def test_current_first_and_historical_edges_only(self):
        graph = self.graphs("train")[0]
        # Target 103 at time 20 sees 101, but not its time-20 peer 102.
        self.assertEqual(graph.event_rows.tolist(), [2, 0])
        self.assertEqual(graph.edge_index.tolist(), [[1], [0]])
        self.assertEqual(graph.target_id.item(), 103)
        self.assertEqual(graph.y.item(), 0)
        self.assertEqual(graph.x[0, 1].item(), 0)
        self.assertGreater(graph.x[1, 1].item(), 0)
        self.assertTrue((graph.event_time[1:] < graph.event_time[0]).all())

    def test_online_history_and_isolated_missing_group(self):
        graphs = self.graphs("validation")
        self.assertEqual(graphs[0].event_rows.tolist(), [3, 0, 1, 2])
        self.assertEqual(graphs[1].num_nodes, 1)
        self.assertEqual(tuple(graphs[1].edge_index.shape), (2, 0))
        for graph in graphs:
            self.assertTrue(torch.isfinite(graph.x).all())
            self.assertTrue(graph.validate(raise_on_error=True))

    def test_disconnected_batch_target_selection_and_shapes(self):
        torch.set_num_threads(2)
        batch = next(iter(DataLoader(self.graphs("validation"), batch_size=2)))
        self.assertEqual(batch.ptr.tolist(), [0, 4, 5])
        self.assertEqual(batch.target_id.tolist(), [104, 105])
        self.assertEqual(batch.y.tolist(), [1., 0.])
        for kind in ["sage", "gat"]:
            model = TemporalGNN(kind)
            output = model(batch)
            self.assertEqual(tuple(output.shape), (2,))
            self.assertTrue(torch.isfinite(output).all())
            torch.nn.BCEWithLogitsLoss()(output, batch.y).backward()

    def test_labels_do_not_enter_node_features(self):
        original = self.graphs("validation")
        changed = self.frame.copy()
        changed["isFraud"] = 1 - changed.isFraud
        part = dict(self.parts["validation"])
        part["labels"] = 1 - part["labels"]
        other = build_neighborhoods(changed, self.features, part)
        for before, after in zip(original, other):
            torch.testing.assert_close(before.x, after.x)
            torch.testing.assert_close(before.edge_index, after.edge_index)

    def test_validation_amount_cannot_change_training_scaling(self):
        changed = self.val.copy()
        changed["TransactionAmt"] = 1e20
        _, features, _, _ = prepare_data(self.train, changed, train_targets=1)
        np.testing.assert_array_equal(self.features[:3, 0], features[:3, 0])

    def test_future_or_wrong_group_history_rejected(self):
        part = dict(self.parts["train"])
        part["windows"] = part["windows"].copy()
        part["windows"][0, 0] = 4
        with self.assertRaises(ValueError):
            build_neighborhoods(self.frame, self.features, part)
        frame = self.frame.copy()
        frame.loc[0, "card1"] = 2
        with self.assertRaises(ValueError):
            build_neighborhoods(frame, self.features, self.parts["train"])


if __name__ == "__main__":
    unittest.main()
