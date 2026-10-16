"""Hand-calculated PSI, label-free alignment and frozen serving-contract checks."""

import json
from pathlib import Path
import tempfile
import unittest
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from api.main import create_app
from api.model import FEATURE_COLUMNS, load_model, prepare_features
from monitoring.check_drift import align_features, load_features, run_monitoring, select_windows
from monitoring.drift_metrics import (
    bin_counts, calculate_psi, decode_edges, distribution_summary, encode_edges,
    reference_edges, smoothed_proportions,
)
from monitoring.verify_session23 import verify_outputs


def feature_fixture():
    raw = pd.DataFrame({"TransactionID": [3, 1, 2], "TransactionDT": [30, 10, 20],
                        "TransactionAmt": [300., 100., 200.], "isFraud": [1, 0, 1]})
    graph = raw[["TransactionID", "TransactionDT"]].iloc[::-1].copy()
    for feature in FEATURE_COLUMNS[1:]:
        graph[feature] = graph.TransactionID.astype(float)
    graph["isFraud"] = 999  # Labels must not be selected from either source.
    return raw, graph


class PsiTests(unittest.TestCase):
    def test_identical_distributions_have_exact_zero_psi_including_missingness(self):
        values = pd.Series([1., 2., 3., 4., 5., np.nan] * 20)
        edges = reference_edges(values)
        self.assertEqual(calculate_psi(values, values, edges), 0.)

    def test_two_bin_example_matches_hand_calculation(self):
        reference, current = [0] * 80 + [1] * 20, [0] * 60 + [1] * 40
        expected = (.6 - .8) * np.log(.6 / .8) + (.4 - .2) * np.log(.4 / .2)
        self.assertAlmostEqual(expected, .1961658506023453)
        self.assertAlmostEqual(calculate_psi(reference, current, [-np.inf, .5, np.inf], epsilon=1e-12), expected, places=10)

    def test_shifted_distribution_has_positive_psi(self):
        reference, current = [1.] * 50 + [10.] * 50, [1.] * 10 + [10.] * 90
        self.assertGreater(calculate_psi(reference, current, reference_edges(reference)), 0)

    def test_fixed_edges_retain_out_of_range_observations(self):
        reference = np.arange(20.)
        edges = reference_edges(reference)
        original = edges.copy()
        counts = bin_counts([-999, 999, np.nan, np.inf, -np.inf], edges)
        self.assertEqual(counts.sum(), 5)
        self.assertEqual(counts[0], 1)
        self.assertEqual(counts[-2], 1)
        self.assertEqual(counts[-1], 3)
        calculate_psi(reference, [999.] * 10, edges)
        np.testing.assert_array_equal(edges, original)

    def test_repeated_counts_and_constant_shifts_are_detectable(self):
        values = [0] * 90 + [1] * 10
        edges = reference_edges(values)
        self.assertTrue(np.all(edges[1:] > edges[:-1]))
        self.assertLess(len(edges), 12)
        self.assertEqual(bin_counts(values, edges).sum(), 100)
        for current in [[0.] * 20, [2.] * 20]:
            constant = [1.] * 20
            self.assertGreater(calculate_psi(constant, current, reference_edges(constant)), 0)

    def test_all_missing_and_missingness_shift_remain_finite(self):
        reference = [np.nan] * 100
        edges = reference_edges(reference)
        self.assertEqual(calculate_psi(reference, reference, edges), 0.)
        psi = calculate_psi(reference, [5.] * 100, edges)
        self.assertTrue(np.isfinite(psi))
        self.assertGreater(psi, 0)
        self.assertIsNone(distribution_summary(reference)["mean"])
        values = [1.] * 100
        shifted = [1.] * 30 + [np.nan] * 70
        self.assertGreater(calculate_psi(values, shifted, reference_edges(values)), 0)

    def test_smoothed_proportions_are_positive_and_sum_to_one(self):
        proportions = smoothed_proportions([1, 1, np.nan], [-np.inf, 0, np.inf])
        self.assertTrue((proportions > 0).all())
        self.assertAlmostEqual(proportions.sum(), 1.)
        np.testing.assert_allclose(proportions, (np.array([0, 2 / 3, 1 / 3]) + 1e-6) / (1 + 3e-6))

    def test_bin_serialization_is_standard_json_and_round_trips(self):
        edges = reference_edges([1, 2, 3, 4])
        encoded = json.dumps(encode_edges(edges), allow_nan=False)
        self.assertNotIn("Infinity", encoded)
        np.testing.assert_array_equal(decode_edges(json.loads(encoded)), edges)

    def test_invalid_windows_edges_smoothing_and_bin_counts_are_rejected(self):
        for edges in [[0, 1], [-np.inf, 1, 1, np.inf], [-np.inf, np.nan, np.inf]]:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                calculate_psi([1], [2], edges)
        for epsilon in [0, -1, np.nan, np.inf]:
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                calculate_psi([1], [2], [-np.inf, np.inf], epsilon)
        with self.assertRaises(ValueError):
            reference_edges([])
        with self.assertRaises(ValueError):
            reference_edges([1, 2], n_bins=1)


class MonitoringIntegrationTests(unittest.TestCase):
    def test_alignment_uses_exact_ids_and_original_timestamps(self):
        raw, graph = feature_fixture()
        aligned = align_features(raw, graph)
        self.assertEqual(aligned.TransactionID.tolist(), [1, 2, 3])
        np.testing.assert_array_equal(aligned[FEATURE_COLUMNS[1]], [1., 2., 3.])
        self.assertNotIn("isFraud", aligned)
        for bad in [graph.iloc[:-1], pd.concat([graph, graph.iloc[:1]]),
                    graph.assign(TransactionDT=999), graph.assign(TransactionID=99)]:
            with self.assertRaises(ValueError):
                align_features(raw, bad)

    def test_loader_selects_only_label_free_train_validation_columns(self):
        raw, graph = feature_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw.to_csv(directory / "train.csv", index=False)
            graph.to_csv(directory / "train_graph_features.csv", index=False)
            with patch("monitoring.check_drift.pd.read_csv", wraps=pd.read_csv) as reading:
                result = load_features("train", directory, directory)
                self.assertEqual(len(reading.call_args_list), 2)
                for call in reading.call_args_list:
                    self.assertNotIn("isFraud", call.kwargs["usecols"])
                self.assertNotIn("isFraud", result)
            with patch("monitoring.check_drift.pd.read_csv") as reading:
                with self.assertRaisesRegex(ValueError, "reserved test"):
                    load_features("test", directory, directory)
                reading.assert_not_called()

    def test_all_missing_feature_values_work_but_absent_columns_fail(self):
        raw, graph = feature_fixture()
        graph[FEATURE_COLUMNS[1]] = np.nan
        aligned = align_features(raw, graph)
        self.assertTrue(aligned[FEATURE_COLUMNS[1]].isna().all())
        self.assertIsNone(distribution_summary(aligned[FEATURE_COLUMNS[1]])["median"])
        with self.assertRaises(KeyError):
            align_features(raw, graph.drop(columns=FEATURE_COLUMNS[1]))

    def test_selection_is_latest_train_earliest_validation_without_overlap(self):
        raw, graph = feature_fixture()
        train = align_features(raw, graph)
        validation = train.assign(TransactionID=lambda x: x.TransactionID + 10, TransactionDT=lambda x: x.TransactionDT + 100)
        reference, current = select_windows(train, validation, 2, 2)
        self.assertEqual(reference.TransactionID.tolist(), [2, 3])
        self.assertEqual(current.TransactionID.tolist(), [11, 12])
        for wrong in [train, validation.assign(TransactionDT=0)]:
            with self.assertRaises(ValueError):
                select_windows(train, wrong, 2, 2)
        with self.assertRaises(ValueError):
            select_windows(train, validation, 100, 2)

    @pytest.mark.private
    def test_batch_adapter_matches_api_and_independent_frozen_vectors(self):
        frozen = json.loads((Path(__file__).parent / "fixtures/session22_golden.json").read_text())
        frame = pd.DataFrame([vector["raw_features"] for vector in frozen["vectors"]])
        model = load_model()
        scores = model.classifier.predict_proba(prepare_features(frame), thread_count=2)[:, 1]
        np.testing.assert_allclose(scores, [vector["expected_score"] for vector in frozen["vectors"]], rtol=1e-10, atol=1e-12)
        with TestClient(create_app()) as client:
            for vector, score in zip(frozen["vectors"], scores):
                raw = vector["raw_features"]
                response = client.post("/predict", json={"transaction_id": vector["transaction_id"],
                    "transaction_amount": raw["TransactionAmt"], "graph_card_prior_degree": raw["graph_card_prior_degree"],
                    "graph_product_prior_degree": raw["graph_product_prior_degree"],
                    "graph_pair_prior_count": raw["graph_card_product_prior_count"],
                    "graph_card_distinct_products": raw["graph_card_distinct_products"]})
                self.assertEqual(response.status_code, 200)
                self.assertAlmostEqual(response.json()["risk_score"], score, places=12)

    @pytest.mark.private
    def test_saved_reference_bins_can_be_reused_but_not_silently_changed(self):
        raw, graph = feature_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for split, offset in [("train", 0), ("validation", 100)]:
                raw.assign(TransactionID=raw.TransactionID + offset, TransactionDT=raw.TransactionDT + offset).to_csv(directory / f"{split}.csv", index=False)
                graph.assign(TransactionID=graph.TransactionID + offset, TransactionDT=graph.TransactionDT + offset).to_csv(directory / f"{split}_graph_features.csv", index=False)
            output = directory / "reports"
            run_monitoring(2, 2, output=output, data=directory, graph=directory)
            baseline = output / "reference_bins.json"
            before = baseline.read_bytes()
            run_monitoring(2, 2, output=output, reference_bins=baseline, data=directory, graph=directory)
            self.assertEqual(before, baseline.read_bytes())
            audit = verify_outputs(output)
            self.assertEqual(audit["scores_reproduced"], 4)
            detail_path = output / "bin_proportions.csv"
            detail_bytes = detail_path.read_bytes()
            details = pd.read_csv(detail_path)
            details.loc[0, "reference_count"] += 1
            details.to_csv(detail_path, index=False)
            with self.assertRaises(AssertionError):
                verify_outputs(output)
            detail_path.write_bytes(detail_bytes)
            record = json.loads(baseline.read_text())
            record["edges"]["TransactionAmt"][1] += .5
            baseline.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Saved cuts"):
                run_monitoring(2, 2, output=output, reference_bins=baseline, data=directory, graph=directory)
