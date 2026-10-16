"""Alignment and preprocessing safeguards for unsupervised anomaly learning."""

import unittest
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anomaly_models.train_anomaly_models import (
    FEATURES, Autoencoder, align_graph_features, autoencoder_scores,
    load_split, raw_features, reconstruction_error,
)


def example():
    frame = pd.DataFrame({"TransactionID": [1, 2, 3], "TransactionDT": [10, 20, 30],
                          "TransactionAmt": [100., 200., 300.], "isFraud": [0, 1, 0]})
    for index, feature in enumerate(FEATURES[1:]):
        frame[feature] = [0., float(index + 1), float(index + 2)]
    return frame


class AnomalyTests(unittest.TestCase):
    def test_merge_aligns_features_by_id_not_file_position(self):
        frame = example()
        raw = frame[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]]
        graph = frame[["TransactionID", "TransactionDT", *FEATURES[1:]]]
        actual = align_graph_features(raw.iloc[::-1], graph.iloc[[1, 2, 0]])
        pd.testing.assert_frame_equal(actual[frame.columns], frame)

    def test_merge_rejects_missing_duplicate_and_stale_rows(self):
        frame = example()
        raw = frame[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]]
        graph = frame[["TransactionID", "TransactionDT", *FEATURES[1:]]]
        for invalid in [graph.iloc[:2], pd.concat([graph, graph.iloc[:1]])]:
            with self.assertRaises(ValueError):
                align_graph_features(raw, invalid)
        stale = graph.copy()
        stale.loc[0, "TransactionDT"] = 999
        with self.assertRaises(ValueError):
            align_graph_features(raw, stale)

    def test_missing_nonfinite_and_negative_inputs_are_finite(self):
        frame = example()
        frame.loc[0, FEATURES] = [np.nan, np.inf, -np.inf, -3, 0]
        values = raw_features(frame)
        np.testing.assert_array_equal(values[0], np.zeros(5))
        np.testing.assert_allclose(values[1, 0], np.log1p(200))
        self.assertTrue(np.isfinite(values).all())

    def test_preprocessing_is_label_independent_and_scaler_is_frozen(self):
        frame = example()
        original = raw_features(frame)
        frame["isFraud"] = 1 - frame.isFraud
        np.testing.assert_array_equal(original, raw_features(frame))
        scaler = StandardScaler().fit(original)
        mean, scale = scaler.mean_.copy(), scaler.scale_.copy()
        scaler.transform(np.full((2, 5), 1e10))
        np.testing.assert_array_equal(scaler.mean_, mean)
        np.testing.assert_array_equal(scaler.scale_, scale)
        with self.assertRaises(ValueError):
            load_split("test")

    def test_reconstruction_shape_per_row_error_and_batches(self):
        torch.manual_seed(42)
        model = Autoencoder()
        values = np.zeros((7, 5), dtype=np.float32)
        self.assertEqual(tuple(model(torch.from_numpy(values)).shape), (7, 5))
        batched = autoencoder_scores(model, values, batch_size=3)
        self.assertEqual(batched.shape, (7,))
        self.assertTrue(np.isfinite(batched).all())
        torch.testing.assert_close(
            reconstruction_error(torch.tensor([[2., 4.]]), torch.tensor([[2.5, 3.]])),
            torch.tensor([.625]),
        )

    def test_isolation_score_orientation(self):
        values = np.arange(20, dtype=float).reshape(10, 2)
        forest = IsolationForest(n_estimators=10, random_state=42).fit(values)
        abnormal = -forest.score_samples(values)
        self.assertTrue(np.isfinite(abnormal).all())
        self.assertEqual(np.argmax(abnormal), np.argmin(forest.score_samples(values)))


if __name__ == "__main__":
    unittest.main()
