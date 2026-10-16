"""Independent strict-time oracle compared with Session 13 and the serving path."""

import unittest
import pytest

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from api.main import create_app
from api.model import FEATURE_COLUMNS, load_model
from dashboard.evidence import align_evidence, make_evidence_record
from graph_models.temporal_graph_features import temporal_graph_features
from monitoring.check_drift import prepare_features


def calculate_prior_card_degree(events):
    """Independent O(n²) oracle: filter strictly earlier events, no mutable state."""
    frame = pd.DataFrame(events)
    output = {}
    for row in frame.itertuples(index=False):
        if pd.isna(row.card1):
            output[row.TransactionID] = np.nan
        else:
            earlier = frame.loc[frame.TransactionDT.lt(row.TransactionDT)]
            output[row.TransactionID] = int(earlier.card1.eq(row.card1).sum())
    return output


def example():
    return pd.DataFrame({"TransactionID": [1, 2, 3, 4], "TransactionDT": [100, 200, 200, 300],
                         "card1": [4001, 4001, 4001, 4001], "ProductCD": ["W", "W", "W", "W"],
                         "card4": ["visa", "visa", "visa", "visa"], "TransactionAmt": [100., 100., 100., 100.]})


class TemporalIntegrationTests(unittest.TestCase):
    def test_equal_time_events_match_independent_oracle_and_actual_generator(self):
        events = example()
        expected = calculate_prior_card_degree(events)
        self.assertEqual(expected, {1: 0, 2: 1, 3: 1, 4: 3})
        actual = temporal_graph_features(events)
        np.testing.assert_array_equal(actual.graph_card_prior_degree, [expected[tid] for tid in events.TransactionID])

    def test_input_order_and_id_tie_break_do_not_create_peer_history(self):
        events = example().iloc[[2, 0, 3, 1]].copy()
        events.TransactionID = [900, 300, 1, 700]
        expected = calculate_prior_card_degree(events)
        actual = temporal_graph_features(events)
        np.testing.assert_array_equal(actual.graph_card_prior_degree, [expected[tid] for tid in events.TransactionID])
        np.testing.assert_array_equal(actual.loc[events.TransactionDT.eq(200), "graph_card_prior_degree"], [1, 1])

    def test_future_labels_and_split_names_do_not_change_earlier_history(self):
        events = example()
        events["split"] = ["train", "train", "validation", "validation"]
        events["isFraud"] = [0, 1, 1, 0]
        baseline = temporal_graph_features(events)
        changed = events.copy()
        changed["isFraud"] = 1 - changed.isFraud
        changed.loc[3, ["card1", "ProductCD", "TransactionAmt"]] = [9999, "C", 999999.]
        pd.testing.assert_frame_equal(baseline.iloc[:3], temporal_graph_features(changed).iloc[:3])
        self.assertEqual(baseline.graph_card_prior_degree.iloc[2], 1)

    def test_missing_proxies_remain_unknown_in_oracle_and_actual_generator(self):
        events = example()
        events.card1 = [np.nan, np.nan, 4001., 4001.]
        expected = calculate_prior_card_degree(events)
        actual = temporal_graph_features(events)
        np.testing.assert_allclose(actual.graph_card_prior_degree, [expected[tid] for tid in events.TransactionID], equal_nan=True)

    @pytest.mark.private
    def test_actual_features_flow_through_api_monitor_adapter_and_dashboard(self):
        events = example()
        features = temporal_graph_features(events)
        graph = pd.concat([events[["TransactionID", "TransactionDT"]], features], axis=1)
        inputs = pd.concat([events[["TransactionAmt"]], features], axis=1)
        model = load_model()
        scores = model.classifier.predict_proba(prepare_features(inputs), thread_count=2)[:, 1]
        predictions = events[["TransactionID", "TransactionDT"]].assign(isFraud=0, AllFive=scores)
        table = align_evidence(predictions, events.assign(isFraud=0), graph)
        with TestClient(create_app()) as client:
            for position, row in events.iterrows():
                request = {"transaction_id": int(row.TransactionID), "transaction_amount": float(row.TransactionAmt),
                           "graph_card_prior_degree": float(features.graph_card_prior_degree.iloc[position]),
                           "graph_product_prior_degree": float(features.graph_product_prior_degree.iloc[position]),
                           "graph_pair_prior_count": float(features.graph_card_product_prior_count.iloc[position]),
                           "graph_card_distinct_products": float(features.graph_card_distinct_products.iloc[position])}
                response = client.post("/predict", json=request)
                self.assertEqual(response.status_code, 200)
                evidence = make_evidence_record(table, int(row.TransactionID))
                self.assertAlmostEqual(response.json()["risk_score"], scores[position], places=12)
                self.assertAlmostEqual(evidence["model_score"], scores[position], places=12)
                self.assertEqual(response.json()["decision"], "not_assigned")
        self.assertEqual(FEATURE_COLUMNS, model.classifier.feature_names_)
