"""Hand-calculated sequence contracts; no real data required."""

import unittest
import numpy as np
import pandas as pd
import torch

from deep_learning.behavioral_data import (
    Preprocessor, event_features_and_windows, materialize, prepare_data,
    truncate_windows,
)
from deep_learning.behavioral_models import make_model, forward


def miniature():
    return pd.DataFrame({
        "TransactionID": ["A", "E", "B", "C", "D"],
        "card1": ["G1", "G2", "G1", "G1", "G1"],
        "TransactionDT": [10, 15, 20, 20, 30],
        "TransactionAmt": [100, 900, 200, 300, 400],
        "isFraud": [0, 0, 1, 0, 1],
    })


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.frame = miniature()
        self.raw, self.windows, self.lengths = event_features_and_windows(self.frame, 20, 10)

    def ids(self, row):
        indices = self.windows[row, :self.lengths[row]]
        return self.frame.TransactionID.iloc[indices].tolist()

    def test_strictly_past_history_and_current(self):
        for i, length in enumerate(self.lengths):
            indices = self.windows[i, :length]
            self.assertEqual(indices[-1], i)
            self.assertTrue((self.frame.TransactionDT.iloc[indices[:-1]] <
                             self.frame.TransactionDT.iloc[i]).all())
        self.assertEqual(self.ids(4), ["A", "B", "C", "D"])

    def test_timestamp_peers_cannot_see_each_other(self):
        self.assertEqual(self.ids(2), ["A", "B"])
        self.assertEqual(self.ids(3), ["A", "C"])
        np.testing.assert_allclose(self.raw[[2, 3], 4], np.log1p([10, 10]))
        np.testing.assert_allclose(self.raw[[2, 3], 6], np.log1p([1, 1]))
        self.assertAlmostEqual(self.raw[4, 6], np.log1p(3))

    def test_distinct_groups(self):
        self.assertEqual(self.ids(1), ["E"])
        self.assertNotIn("E", self.ids(4))

    def test_missing_groups_have_no_shared_history(self):
        self.frame.loc[[0, 2, 4], "card1"] = np.nan
        _, windows, lengths = event_features_and_windows(self.frame, 20, 10)
        for i in [0, 2, 4]:
            self.assertEqual(lengths[i], 1)
            self.assertEqual(windows[i, 0], i)

    def test_nonfinite_numeric_groups_are_missing(self):
        frame = self.frame.copy()
        frame["card1"] = [np.inf, np.nan, -np.inf, 1, 1]
        raw, _, lengths = event_features_and_windows(frame, 20, 10)
        np.testing.assert_array_equal(lengths[:3], 1)
        np.testing.assert_array_equal(raw[:3, 3], 1)

    def test_padding_and_recent_truncation(self):
        windows, lengths = truncate_windows(self.windows, self.lengths, 2)
        np.testing.assert_array_equal(windows[4], [3, 4])
        np.testing.assert_array_equal(lengths, [1, 1, 2, 2, 2])
        values = materialize(np.nan_to_num(self.raw), windows)
        np.testing.assert_array_equal(values[0, 1], 0)

    def test_length_one_is_current_only(self):
        _, windows, lengths = event_features_and_windows(self.frame, 1, 10)
        np.testing.assert_array_equal(windows[:, 0], np.arange(5))
        np.testing.assert_array_equal(lengths, 1)

    def test_target_ids_labels_and_prior_training_history(self):
        frame, _, parts, _ = prepare_data(self.frame.iloc[:2], self.frame.iloc[2:],
                                          train_targets=1, val_targets=3)
        val = parts["validation"]
        np.testing.assert_array_equal(val["ids"], ["B", "C", "D"])
        np.testing.assert_array_equal(val["labels"], [1, 0, 1])
        self.assertEqual(frame.TransactionID.iloc[val["windows"][0, 0]], "A")
        self.assertEqual(parts["train"]["ids"].tolist(), ["E"])

    def test_training_statistics_are_not_fitted_to_validation(self):
        train, val = self.frame.iloc[:2].copy(), self.frame.iloc[2:].copy()
        _, before, _, processor = prepare_data(train, val)
        val["TransactionAmt"] = 1e20
        _, after, _, changed = prepare_data(train, val)
        np.testing.assert_array_equal(processor.mean, changed.mean)
        np.testing.assert_array_equal(processor.median, changed.median)
        np.testing.assert_array_equal(processor.scale, changed.scale)
        np.testing.assert_array_equal(before[:2], after[:2])
        self.assertAlmostEqual(processor.mean[0], np.log1p([100, 900]).mean())

    def test_future_and_label_invariance(self):
        frame = self.frame.copy()
        frame["TransactionAmt"] = frame.TransactionAmt.astype(float)
        frame.loc[4, "TransactionAmt"] = 1e20
        frame["isFraud"] = 1 - frame.isFraud
        raw, windows, lengths = event_features_and_windows(frame, 20, 10)
        np.testing.assert_allclose(self.raw[:4], raw[:4], equal_nan=True)
        np.testing.assert_array_equal(self.windows, windows)
        np.testing.assert_array_equal(self.lengths, lengths)

    def test_no_nonfinite_model_inputs(self):
        frame = self.frame.copy()
        frame["TransactionAmt"] = [np.nan, np.inf, -np.inf, -1, 0]
        raw, windows, _ = event_features_and_windows(frame, 20, 10)
        processor = Preprocessor.fit(raw[:2], 10)
        self.assertTrue(np.isfinite(materialize(processor.transform(raw), windows)).all())

    def test_invalid_times_and_split_overlap_rejected(self):
        frame = self.frame.copy()
        frame.loc[0, "TransactionDT"] = np.nan
        with self.assertRaises(ValueError):
            prepare_data(frame.iloc[:2], frame.iloc[2:])
        with self.assertRaises(ValueError):
            prepare_data(self.frame.iloc[:3], self.frame.iloc[3:])
        with self.assertRaises(ValueError):
            event_features_and_windows(self.frame, 0, 10)


class ModelTests(unittest.TestCase):
    def test_output_shapes_and_padding_invariance(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        x = torch.randn(3, 5, 7)
        lengths = torch.tensor([5, 2, 1])
        for kind in ["gru", "transformer"]:
            model = make_model(kind, 7, 5).eval()
            with torch.no_grad():
                output = forward(model, kind, x, lengths)
                self.assertEqual(tuple(output.shape), (3, 1))
                self.assertTrue(torch.isfinite(output).all())
                perturbed = x.clone()
                perturbed[1, 2:] = 1000
                perturbed[2, 1:] = -1000
                torch.testing.assert_close(output, forward(model, kind, perturbed, lengths))
        model = make_model("mlp", 4, 1)
        self.assertEqual(tuple(forward(model, "mlp", x[:, :1, :4], lengths).shape), (3, 1))


if __name__ == "__main__":
    unittest.main()
