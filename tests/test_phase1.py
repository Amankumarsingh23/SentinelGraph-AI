"""Small hand-calculated regressions; no datasets or model training needed."""
import unittest
import numpy as np
import pandas as pd

from feature_engineering import FEATURES, historical_features
from evaluate_risk import alert_ranking, chronological_midpoint, metric_report, threshold_costs


def events(amounts, times=None, groups=None):
    n = len(amounts)
    return pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionAmt": amounts,
        "TransactionDT": times if times is not None else np.arange(n),
        "card1": groups if groups is not None else np.ones(n),
    })


class HistoricalTests(unittest.TestCase):
    def test_missing_amounts_and_first_event(self):
        f = historical_features(events([10, np.nan, 30]))
        np.testing.assert_allclose(f.card1_prior_count, [0, 1, 2])
        np.testing.assert_allclose(f.card1_prior_mean_amount, [np.nan, 10, 10], equal_nan=True)
        np.testing.assert_allclose(f.card1_amount_ratio, [np.nan, np.nan, 3], equal_nan=True)
        self.assertTrue(np.isnan(f.card1_time_since_previous.iloc[0]))

    def test_simultaneous_events(self):
        f = historical_features(events([10, 20, 30, 40], [1, 2, 2, 3]))
        np.testing.assert_allclose(f.card1_prior_count, [0, 1, 1, 3])
        np.testing.assert_allclose(f.card1_prior_mean_amount, [np.nan, 10, 10, 20], equal_nan=True)
        np.testing.assert_allclose(f.card1_time_since_previous, [np.nan, 1, 1, 1], equal_nan=True)
        first_batch = historical_features(events([10, 20], [1, 1]))
        self.assertTrue(first_batch.card1_prior_count.eq(0).all())
        self.assertTrue(first_batch.card1_prior_mean_amount.isna().all())

    def test_groups_and_earlier_training_history(self):
        frame = events([10, 100, 30, 200], [1, 1, 2, 3], [1, 2, 1, 2])
        frame['_split'] = ['train', 'train', 'validation', 'test']
        f = historical_features(frame)
        np.testing.assert_allclose(f.card1_prior_mean_amount, [np.nan, np.nan, 10, 100], equal_nan=True)
        np.testing.assert_allclose(f.card1_time_since_previous, [np.nan, np.nan, 1, 2], equal_nan=True)

    def test_future_and_labels_do_not_affect_history(self):
        frame = events([10, 20, 30, 1000])
        before = historical_features(frame.iloc[:3])
        frame['isFraud'] = [1, 0, 1, 1]
        pd.testing.assert_frame_equal(before, historical_features(frame).iloc[:3])
        frame.loc[3, 'TransactionAmt'] = -1000
        frame['isFraud'] = 0
        pd.testing.assert_frame_equal(before, historical_features(frame).iloc[:3])

    def test_input_order_and_same_time_peer_invariance(self):
        frame = events([10, 20, 30, 40], [1, 2, 2, 3])
        original = historical_features(frame)
        shuffled = frame.iloc[[3, 2, 0, 1]]
        pd.testing.assert_frame_equal(original, historical_features(shuffled).sort_index())
        frame.loc[2, 'TransactionAmt'] = 9999
        pd.testing.assert_series_equal(original.iloc[1], historical_features(frame).iloc[1])

    def test_missing_entities_nonfinite_amounts_and_zero(self):
        frame = events([0, np.inf, -np.inf, np.nan, 5, 7, 8], groups=[1, 1, 1, 1, 1, np.nan, np.inf])
        f = historical_features(frame)
        self.assertEqual(f.card1_prior_count.iloc[4], 4)
        self.assertEqual(f.card1_prior_mean_amount.iloc[4], 0)
        self.assertTrue(f.card1_amount_ratio.isna().all())
        self.assertTrue(f.iloc[-2:][FEATURES[:4]].isna().all().all())
        self.assertEqual(f.card1_missing.sum(), 2)
        self.assertFalse(np.isinf(f.to_numpy()).any())
        unknown = historical_features(events([1, 2], groups=[np.nan, np.nan]))
        self.assertTrue(unknown[FEATURES[:4]].isna().all().all())

    def test_unknown_history_and_invalid_time(self):
        f = historical_features(events([np.nan, 5]))
        self.assertTrue(f.card1_prior_mean_amount.isna().all())
        for time in [np.nan, np.inf, -np.inf]:
            with self.assertRaises(ValueError):
                historical_features(events([1], [time]))


class EvaluationTests(unittest.TestCase):
    def test_ranking_ties_and_permutation(self):
        p = np.array([.5, .5, .5, .4])
        raw = np.array([.7, .8, .8, .9])
        ids = np.array([9, 4, 2, 1])
        np.testing.assert_array_equal(ids[alert_ranking(p, raw, ids)], [2, 4, 9, 1])
        order = [3, 1, 0, 2]
        np.testing.assert_array_equal(ids[order][alert_ranking(p[order], raw[order], ids[order])], [2, 4, 9, 1])

    def test_precision_and_recall_share_top_k(self):
        ids = np.arange(101)
        labels = np.zeros(101, dtype=int)
        labels[[0, 100]] = 1
        report = metric_report(labels, np.full(101, .2), np.full(101, .1), ids)
        self.assertEqual(report['frauds_in_top_k'], 1)
        self.assertEqual(report['precision_at_k'], .01)
        self.assertEqual(report['recall_at_k'], .5)

    def test_cost_baselines_alerts_and_all_ties(self):
        result = threshold_costs([1, 0], [.2, .1], [50, 10])
        none = result.loc[result.policy == 'alert_none'].iloc[0]
        all_alerts = result.loc[result.policy == 'alert_all'].iloc[0]
        self.assertEqual((none.alerts, none.realized_cost_proxy, none.missed_frauds), (0, 50, 1))
        self.assertEqual((all_alerts.alerts, all_alerts.realized_cost_proxy, all_alerts.false_positives), (2, 200, 1))
        # At 0.15 and 0.20 only the fraud is alerted: zero realized cost.
        self.assertEqual(result.loc[result.is_minimum, 'threshold'].tolist(), [.15, .2])
        self.assertTrue(result.loc[result.is_minimum, 'alerts'].eq(1).all())

    def test_midpoint_keeps_time_ties_together(self):
        self.assertEqual(chronological_midpoint([1, 2, 2, 2, 3, 4]), 4)
        with self.assertRaises(ValueError):
            chronological_midpoint([1, 1])
        with self.assertRaises(ValueError):
            chronological_midpoint([2, 1])

    def test_invalid_cost_amount_rejected(self):
        for amount in [np.nan, np.inf, -1]:
            with self.assertRaises(ValueError):
                threshold_costs([1], [.1], [amount])


if __name__ == '__main__':
    unittest.main()
