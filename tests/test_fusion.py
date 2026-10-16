"""Chronological holdout stacking: score alignment, labels and scaler scope."""

import unittest
import numpy as np
import pandas as pd

from fusion_models.train_fusion import (
    SCORE_COLUMNS, STACKS, fit_stacker, holdout_partition, merge_anomaly_scores,
)


def fixture():
    validation = pd.DataFrame({
        "TransactionID": np.arange(1, 7), "TransactionDT": np.arange(10, 70, 10),
        "isFraud": [0, 1, 0, 1, 0, 1],
    })
    predictions = {}
    for name in SCORE_COLUMNS:
        predictions[name] = pd.DataFrame({
            "TransactionID": np.arange(1, 7), "isFraud": validation.isFraud,
            "model": name, "anomaly_score": np.arange(1, 7) / 10,
        })
    return validation, predictions


class FusionTests(unittest.TestCase):
    def test_saved_scores_align_by_id_with_original_labels(self):
        validation, predictions = fixture()
        for name in predictions:
            predictions[name] = predictions[name].sample(frac=1, random_state=4)
        actual = merge_anomaly_scores(validation.iloc[::-1], predictions)
        np.testing.assert_array_equal(actual.TransactionID, np.arange(1, 7))
        np.testing.assert_array_equal(actual.isFraud, validation.isFraud)
        np.testing.assert_allclose(actual.AutoencoderScore, np.arange(1, 7) / 10)

    def test_duplicate_mismatched_or_unknown_ids_rejected(self):
        validation, predictions = fixture()
        for problem in ["duplicate", "missing", "unknown"]:
            changed = {name: table.copy() for name, table in predictions.items()}
            if problem == "duplicate":
                changed["autoencoder"] = pd.concat([changed["autoencoder"], changed["autoencoder"].iloc[:1]])
            elif problem == "missing":
                changed["autoencoder"] = changed["autoencoder"].iloc[:-1]
            else:
                changed["isolation_forest"].loc[0, "TransactionID"] = 100
            with self.assertRaises(ValueError):
                merge_anomaly_scores(validation, changed)

    def test_wrong_labels_nonfinite_scores_and_model_name_rejected(self):
        validation, predictions = fixture()
        for column, value in [("isFraud", 1), ("anomaly_score", np.inf), ("model", "wrong")]:
            changed = {name: table.copy() for name, table in predictions.items()}
            changed["autoencoder"].loc[0, column] = value
            with self.assertRaises(ValueError):
                merge_anomaly_scores(validation, changed)

    def test_midpoint_moves_complete_timestamp_batch_to_meta_training(self):
        validation, _ = fixture()
        validation["TransactionDT"] = [10, 20, 30, 30, 40, 50]
        meta, evaluation = holdout_partition(validation)
        self.assertEqual(meta.TransactionID.tolist(), [1, 2, 3, 4])
        self.assertEqual(evaluation.TransactionID.tolist(), [5, 6])
        self.assertLess(meta.TransactionDT.max(), evaluation.TransactionDT.min())

    def test_invalid_boundary_or_label_classes_rejected(self):
        validation, _ = fixture()
        validation["TransactionDT"] = 10
        with self.assertRaises(ValueError):
            holdout_partition(validation)
        validation, _ = fixture()
        validation.loc[:2, "isFraud"] = 0
        with self.assertRaises(ValueError):
            holdout_partition(validation)

    def test_stacker_scaling_fits_meta_rows_only_and_preserves_score_order(self):
        validation, predictions = fixture()
        scored = merge_anomaly_scores(validation, predictions)
        scored["CatBoostScore"] = [.1, .8, .2, .9, .3, .7]
        meta, evaluation = holdout_partition(scored)
        for columns in STACKS.values():
            model = fit_stacker(meta, columns)
            scaler = model.steps[0][1]
            np.testing.assert_allclose(scaler.mean_, meta[columns].mean().to_numpy())
            self.assertEqual(scaler.n_samples_seen_, len(meta))
            before = scaler.mean_.copy()
            changed = evaluation.copy()
            changed[columns] = 1000.
            scores = model.predict_proba(changed[columns])[:, 1]
            self.assertEqual(scores.shape, (len(evaluation),))
            self.assertTrue(np.isfinite(scores).all())
            np.testing.assert_array_equal(scaler.mean_, before)


if __name__ == "__main__":
    unittest.main()
