"""Hand-calculated temporal graph checks, independent of the real dataset."""

import unittest
import numpy as np
import pandas as pd

from graph_models.temporal_graph_features import FEATURES, temporal_graph_features


def example():
    return pd.DataFrame({
        "TransactionID": ["T1", "T2", "T3", "T4"],
        "TransactionDT": [100, 200, 200, 300],
        "card1": ["A", "A", "A", "A"],
        "ProductCD": ["W", "W", "C", "C"],
        "card4": ["visa", "visa", "visa", "visa"],
    })


class TemporalGraphTests(unittest.TestCase):
    def test_session_hand_calculated_example(self):
        actual = temporal_graph_features(example())
        # Card degree, product degree, distinct products, pair count, novelty,
        # network degree. T2 and T3 both see only T1; T4 sees all three.
        expected = [[0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 0, 1],
                    [1, 0, 1, 0, 1, 1], [3, 1, 2, 1, 0, 3]]
        np.testing.assert_array_equal(actual[FEATURES[:6]], expected)
        np.testing.assert_array_equal(actual[FEATURES[6:]], 0)

    def test_distinct_groups_and_categories(self):
        frame = example()
        frame["TransactionDT"] = [100, 200, 300, 400]
        frame["card1"] = ["A", "A", "B", "A"]
        frame["card4"] = ["visa", "visa", "mastercard", "visa"]
        actual = temporal_graph_features(frame)
        np.testing.assert_array_equal(actual.graph_card_prior_degree, [0, 1, 0, 2])
        np.testing.assert_array_equal(actual.graph_product_prior_degree, [0, 1, 0, 1])
        np.testing.assert_array_equal(actual.graph_card4_prior_degree, [0, 1, 0, 2])
        self.assertEqual(actual.graph_card_product_prior_count.iloc[-1], 0)
        self.assertEqual(actual.graph_card_distinct_products.iloc[-1], 1)

    def test_missing_attributes_do_not_share_history(self):
        frame = example()
        frame["card1"] = [None, None, "A", "A"]
        frame["ProductCD"] = ["W", "W", None, "W"]
        frame["card4"] = [None, None, "visa", "visa"]
        actual = temporal_graph_features(frame)
        self.assertTrue(actual.graph_card_prior_degree.iloc[:2].isna().all())
        self.assertTrue(actual.graph_card4_prior_degree.iloc[:2].isna().all())
        self.assertEqual(actual.graph_product_prior_degree.iloc[1], 1)
        self.assertTrue(np.isnan(actual.graph_product_prior_degree.iloc[2]))
        self.assertEqual(actual.graph_card_prior_degree.iloc[3], 1)
        self.assertEqual(actual.graph_card_distinct_products.iloc[3], 0)
        self.assertEqual(actual.graph_card_product_prior_count.iloc[3], 0)
        self.assertEqual(actual.graph_new_product_for_card.iloc[3], 1)
        self.assertTrue(actual.graph_new_product_for_card.iloc[:3].isna().all())

    def test_nonfinite_keys_and_label_free_inputs(self):
        frame = example()
        frame["card1"] = [np.inf, -np.inf, np.nan, 1.0]
        actual = temporal_graph_features(frame)
        self.assertTrue(actual.graph_card_prior_degree.iloc[:3].isna().all())
        self.assertEqual(actual.graph_card_prior_degree.iloc[3], 0)
        self.assertFalse(np.isinf(actual.to_numpy()).any())
        frame["isFraud"] = [1, 0, 1, 0]
        frame["TransactionAmt"] = [np.inf, 1, 99999, np.nan]
        pd.testing.assert_frame_equal(actual, temporal_graph_features(frame))

    def test_future_and_same_timestamp_peer_invariance(self):
        frame = example()
        original = temporal_graph_features(frame)
        frame.loc[2:, "ProductCD"] = "OTHER"
        frame.loc[3, "card1"] = "B"
        pd.testing.assert_frame_equal(original.iloc[:2], temporal_graph_features(frame).iloc[:2])
        pd.testing.assert_frame_equal(original.iloc[:3], temporal_graph_features(example().iloc[:3]))

    def test_input_order_and_target_alignment(self):
        frame = example()
        frame.index = [8, 2, 6, 4]
        shuffled = frame.iloc[[3, 2, 0, 1]]
        expected = temporal_graph_features(frame)
        actual = temporal_graph_features(shuffled)
        pd.testing.assert_frame_equal(expected.sort_index(), actual.sort_index())
        self.assertEqual(actual.index.tolist(), shuffled.index.tolist())

    def test_training_to_online_validation_history(self):
        frame = example()
        frame["split"] = ["train", "train", "validation", "validation"]
        actual = temporal_graph_features(frame)
        self.assertEqual(actual.graph_card_prior_degree.iloc[2], 1)
        self.assertEqual(actual.graph_card_prior_degree.iloc[3], 3)

    def test_invalid_ids_and_times_rejected(self):
        for invalid_time in [np.nan, np.inf, -np.inf]:
            frame = example()
            frame["TransactionDT"] = frame.TransactionDT.astype(float)
            frame.loc[0, "TransactionDT"] = invalid_time
            with self.assertRaises(ValueError):
                temporal_graph_features(frame)
        for invalid_id in ["T2", None]:
            frame = example()
            frame.loc[0, "TransactionID"] = invalid_id
            with self.assertRaises(ValueError):
                temporal_graph_features(frame)


if __name__ == "__main__":
    unittest.main()
