"""Matched-row ablation and paired bootstrap correctness, without real data."""

import unittest
import numpy as np
import pandas as pd

from evaluation.session20_ablations import (
    align_evaluation, block_metrics, chronological_blocks, paired_ap_bootstrap,
)


def fixture():
    validation = pd.DataFrame({"TransactionID": [1, 2, 3, 4],
                               "TransactionDT": [10, 20, 30, 40], "isFraud": [0, 1, 0, 1]})
    reference = validation.iloc[[3, 0, 2, 1]].copy()
    reference["catboost"] = [.8, .1, .2, .9]
    return validation, reference


class Session20Tests(unittest.TestCase):
    def test_reference_ids_and_labels_align_despite_shuffled_rows(self):
        validation, reference = fixture()
        actual = align_evaluation(validation.iloc[::-1], reference)
        pd.testing.assert_frame_equal(actual, validation)

    def test_reference_mismatch_cannot_be_silently_ignored(self):
        validation, reference = fixture()
        for column, bad_value in [("TransactionID", 99), ("TransactionDT", 999), ("isFraud", 0)]:
            changed = reference.copy()
            changed.loc[3, column] = bad_value
            with self.assertRaises((ValueError, AssertionError)):
                align_evaluation(validation, changed)
        with self.assertRaises(ValueError):
            align_evaluation(validation, pd.concat([reference, reference.iloc[:1]]))

    def test_blocks_preserve_ties_and_cover_every_row_once(self):
        times = np.array([1, 2, 2, 2, 3, 4, 5, 6])
        blocks = chronological_blocks(times, 4)
        self.assertEqual([(s.start, s.stop) for s in blocks], [(0, 4), (4, 6), (6, 8)])
        np.testing.assert_array_equal(np.concatenate([np.arange(len(times))[s] for s in blocks]), np.arange(len(times)))
        for before, after in zip(blocks[:-1], blocks[1:]):
            self.assertLess(times[before][-1], times[after][0])

    def test_identical_scores_have_zero_paired_delta_and_reproducible_draws(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([.1, .2, .8, .9])
        first, summary = paired_ap_bootstrap(labels, scores, scores, repetitions=100, seed=7)
        repeated, _ = paired_ap_bootstrap(labels, scores, scores, repetitions=100, seed=7)
        pd.testing.assert_frame_equal(first, repeated)
        self.assertTrue(first.delta_ap.eq(0).all())
        self.assertEqual((summary["ci_lower"], summary["ci_upper"]), (0., 0.))
        self.assertEqual(summary["valid_draws"] + summary["skipped_single_class_draws"], 100)
        self.assertGreater(summary["skipped_single_class_draws"], 0)

    def test_bootstrap_uses_same_sample_indices_for_both_models(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        amount = np.array([.9, .8, .7, .6, .5, .4])
        full = np.array([.1, .8, .2, .9, .3, .7])
        draws, _ = paired_ap_bootstrap(labels, amount, full, repetitions=20, seed=42)
        # The first seeded sample is [0, 4, 3, 2, 2, 5]: positives are indices
        # 3 and 5. Full ranks them first/second; Amount ranks them fourth/sixth.
        first = draws.iloc[0]
        self.assertEqual(first.draw, 1)
        self.assertEqual(first.sampled_frauds, 2)
        self.assertAlmostEqual(first.all_five_ap, 1.0)
        self.assertAlmostEqual(first.amount_only_ap, (.25 + 1 / 3) / 2)
        self.assertAlmostEqual(first.delta_ap, 1 - (.25 + 1 / 3) / 2)

    def test_invalid_bootstrap_inputs_rejected(self):
        for labels, a, b in [([0, 0], [.1, .2], [.2, .3]),
                             ([0, 1], [.1], [.2, .3]),
                             ([0, 1], [.1, np.inf], [.2, .3])]:
            with self.assertRaises(ValueError):
                paired_ap_bootstrap(labels, a, b, repetitions=20)
        with self.assertRaises(ValueError):
            chronological_blocks([2, 1], 2)

    def test_zero_fraud_block_metrics_are_explicitly_unavailable(self):
        result = block_metrics(np.zeros(3), np.array([.1, .2, .3]), np.arange(3))
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(result["recall_at_100"])
        self.assertEqual(result["precision_at_100"], 0.)


if __name__ == "__main__":
    unittest.main()
