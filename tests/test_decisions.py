"""Decision-cost hand calculations and calibration/selection separation."""

import unittest
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from decision_models.evaluate_decisions import (
    aligned_saved_scores, capacity_mask, capacity_table, cost_report,
    fit_calibrator, score_logits, select_capacity, three_way_partition, transfer_capacity,
)
from decision_models.explain_catboost import shap_attributions


def fixture():
    frame = pd.DataFrame({
        "TransactionID": np.arange(1, 13), "TransactionDT": np.arange(10, 130, 10),
        "TransactionAmt": [100.] * 12, "isFraud": [0, 1] * 6,
        "CatBoostScore": [.1, .8] * 6,
        "calibrated_probability": [.2, .6] * 6,
    })
    return frame


class DecisionTests(unittest.TestCase):
    def test_three_part_boundaries_keep_timestamp_groups_whole(self):
        frame = fixture()
        frame.loc[3, "TransactionDT"] = 30  # tie across first quarter
        frame.loc[6, "TransactionDT"] = 60  # tie across midpoint
        parts = three_way_partition(frame)
        self.assertEqual([len(part) for part in parts.values()], [4, 3, 5])
        self.assertLess(parts["calibration"].TransactionDT.max(), parts["policy_development"].TransactionDT.min())
        self.assertLess(parts["policy_development"].TransactionDT.max(), parts["evaluation"].TransactionDT.min())

    def test_calibrator_only_fits_designated_first_block(self):
        parts = three_way_partition(fixture())
        calibration = parts["calibration"]
        model = fit_calibrator(calibration)
        changed = fixture()
        changed.loc[3:, "isFraud"] = 1 - changed.loc[3:, "isFraud"]
        changed.loc[3:, "CatBoostScore"] = .99
        repeated = fit_calibrator(three_way_partition(changed)["calibration"])
        np.testing.assert_array_equal(model.coef_, repeated.coef_)
        np.testing.assert_array_equal(model.intercept_, repeated.intercept_)
        self.assertTrue(np.isfinite(score_logits([0, 1])).all())
        with self.assertRaises(ValueError):
            score_logits([1.1])

    def test_cost_formula_and_no_alert_all_alert_baselines(self):
        actual = cost_report([1, 0, 1], [100, 500, 300], np.array([True, True, False]))
        self.assertEqual(actual["realized_cost_proxy"], 500)
        self.assertEqual(actual["true_positives"], 1)
        self.assertEqual(actual["false_positives"], 1)
        self.assertEqual(actual["caught_fraud_amount"], 100)
        self.assertEqual(cost_report([1, 0], [100, 500], np.array([False, False]))["realized_cost_proxy"], 100)
        self.assertEqual(cost_report([1, 0], [100, 500], np.array([True, True]))["realized_cost_proxy"], 200)

    def test_cost_rejects_invalid_money_labels_and_mask(self):
        for invalid in [np.nan, np.inf, -1]:
            with self.assertRaises(ValueError):
                cost_report([1], [invalid], np.array([True]))
        with self.assertRaises(ValueError):
            cost_report([2], [10], np.array([True]))
        with self.assertRaises(ValueError):
            cost_report([1], [10], np.array([1]))

    def test_ties_use_raw_score_then_id_without_labels(self):
        frame = fixture().iloc[:3].copy()
        frame["TransactionID"] = [9, 4, 2]
        frame["calibrated_probability"] = .5
        frame["CatBoostScore"] = [.7, .8, .8]
        selected = capacity_mask(frame, 1)
        np.testing.assert_array_equal(frame.TransactionID[selected], [2])
        frame["isFraud"] = 1 - frame.isFraud
        np.testing.assert_array_equal(capacity_mask(frame, 1), selected)

    def test_candidate_cost_ties_prefer_fewer_alerts_and_transfer_fraction(self):
        frame = fixture().iloc[:4].copy()
        frame["isFraud"] = 1
        frame["TransactionAmt"] = 0
        table = capacity_table(frame, capacities=[0, 1, 2])
        self.assertTrue(table.is_minimum.all())
        self.assertEqual(select_capacity(table), 0)
        self.assertIn(4, table.capacity.tolist())
        self.assertEqual(transfer_capacity(50, 2500, 5000), 100)
        self.assertEqual(transfer_capacity(1, 2, 3), 2)

    def test_score_alignment_checks_labels_and_exact_ids(self):
        frame = fixture()
        saved = frame.iloc[::-1].copy()
        np.testing.assert_allclose(aligned_saved_scores(frame, saved, "CatBoostScore"), frame.CatBoostScore)
        saved.loc[0, "isFraud"] = 1
        with self.assertRaises(ValueError):
            aligned_saved_scores(frame, saved, "CatBoostScore")
        with self.assertRaises(ValueError):
            aligned_saved_scores(frame, frame.iloc[:-1], "CatBoostScore")

    def test_native_shap_reconstructs_fitted_raw_output_and_probability(self):
        X = pd.DataFrame({"amount": [1., 2., 3., 4., 5., 6.], "history": [0., 1., 0., 2., 1., 3.]})
        model = CatBoostClassifier(iterations=5, depth=2, verbose=False, random_seed=42,
                                   thread_count=2, allow_writing_files=False)
        model.fit(X, [0, 0, 0, 1, 1, 1])
        contributions, base, raw, rebuilt, probabilities = shap_attributions(model, X.iloc[:2])
        self.assertEqual(contributions.shape, (2, 2))
        np.testing.assert_allclose(raw, rebuilt, atol=1e-5)
        self.assertEqual(probabilities.shape, (2,))
        with self.assertRaises(ValueError):
            shap_attributions(model, X[["history", "amount"]])


if __name__ == "__main__":
    unittest.main()
