"""Investigator evidence remains keyed, reproducible and distinct from outcomes."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import pytest
from unittest.mock import patch

import numpy as np
import pandas as pd

from api.model import FEATURE_COLUMNS
from dashboard.evidence import (
    PREDICTIONS, align_evidence, deterministic_summary, load_clusters,
    load_evidence_table, make_evidence_record,
)
from dashboard.shap_evidence import checked_shap, load_shap_export


def fixtures():
    raw = pd.DataFrame({"TransactionID": [1, 2], "TransactionDT": [10, 20], "isFraud": [0, 1],
                        "TransactionAmt": [10., 20.], "card1": [11, 22], "ProductCD": ["W", "C"],
                        "card4": ["visa", "mastercard"]})
    graph = raw[["TransactionID", "TransactionDT"]].copy()
    for name in FEATURE_COLUMNS[1:]:
        graph[name] = [4., 2.]
    predictions = raw[["TransactionID", "TransactionDT", "isFraud"]].copy()
    predictions["AllFive"] = [.1, .9]
    return predictions, raw, graph


class EvidenceTests(unittest.TestCase):
    def test_summary_is_grounded_and_assigns_no_fraud_decision(self):
        summary = deterministic_summary({"transaction_id": 123, "model_score": .25,
            "historical_features": {"graph_card_prior_degree": 4, "graph_pair_prior_count": 2}})
        for text in ["Transaction 123", "0.250000", "4", "2", "uncalibrated", "No fraud decision has been assigned"]:
            self.assertIn(text, summary)
        for unsupported in ["criminal network", "fraud ring", "is fraudulent"]:
            self.assertNotIn(unsupported, summary)

    def test_missing_evidence_is_unavailable_while_zero_is_preserved(self):
        evidence = {"transaction_id": 456, "model_score": .01,
                    "historical_features": {"graph_card_prior_degree": None, "graph_pair_prior_count": None}}
        summary = deterministic_summary(evidence)
        self.assertIn("prior matching card-product transactions: unavailable", summary)
        self.assertIn("Prior card-proxy observations: unavailable", summary)
        evidence["historical_features"]["graph_card_prior_degree"] = 0
        self.assertIn("Prior card-proxy observations: 0;", deterministic_summary(evidence))

    def test_shuffled_sources_align_by_id_and_rank_without_exposing_labels(self):
        predictions, raw, graph = fixtures()
        table = align_evidence(predictions, raw.iloc[::-1], graph.iloc[::-1])
        self.assertEqual(table.TransactionID.tolist(), [2, 1])
        evidence = make_evidence_record(table, 1)
        self.assertEqual(evidence["transaction_amount"], 10.)
        self.assertEqual(evidence["model_score"], .1)
        self.assertEqual(evidence["historical_features"]["graph_card_prior_degree"], 4.)
        self.assertEqual(evidence["decision"], "not_assigned")
        self.assertFalse(evidence["calibrated"])
        self.assertNotIn("isFraud", table)
        self.assertNotIn("isFraud", evidence)

    def test_duplicate_or_unmatched_ids_are_rejected_in_every_source(self):
        frames = fixtures()
        for index in range(3):
            bad = list(frames)
            bad[index] = pd.concat([bad[index], bad[index].iloc[:1]])
            with self.subTest(source=index), self.assertRaisesRegex(ValueError, "unique"):
                align_evidence(*bad)
        for index in [1, 2]:
            bad = list(frames)
            bad[index] = bad[index].iloc[:1]
            with self.assertRaisesRegex(ValueError, "validation and graph evidence"):
                align_evidence(*bad)

    def test_wrong_timestamps_and_retrospective_labels_are_rejected(self):
        predictions, raw, graph = fixtures()
        for bad in [(predictions.assign(TransactionDT=99), raw, graph),
                    (predictions, raw, graph.assign(TransactionDT=99))]:
            with self.assertRaisesRegex(ValueError, "timestamps"):
                align_evidence(*bad)
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            align_evidence(predictions.assign(isFraud=0), raw, graph)

    def test_invalid_saved_scores_and_unknown_selection_are_rejected(self):
        predictions, raw, graph = fixtures()
        for score in [np.nan, np.inf, -1, 2]:
            with self.assertRaises(ValueError):
                align_evidence(predictions.assign(AllFive=score), raw, graph)
        with self.assertRaises(ValueError):
            make_evidence_record(align_evidence(predictions, raw, graph), 999)

    def test_ties_are_ranked_by_id_and_missing_features_stay_unknown(self):
        predictions, raw, graph = fixtures()
        predictions.AllFive = .5
        graph["graph_card_prior_degree"] = np.nan
        table = align_evidence(predictions, raw, graph)
        self.assertEqual(table.TransactionID.tolist(), [1, 2])
        record = make_evidence_record(table, 1)
        self.assertIsNone(record["historical_features"]["graph_card_prior_degree"])
        json.dumps(record, allow_nan=False)


@pytest.mark.private
class SavedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = load_evidence_table()
        cls.payload, cls.lookup = load_shap_export()

    def test_all_saved_scores_match_session20_and_loader_never_opens_test(self):
        with patch("dashboard.evidence.pd.read_csv", wraps=pd.read_csv) as reading:
            table = load_evidence_table()
        paths = [str(call.args[0]) for call in reading.call_args_list]
        self.assertTrue(all("test.csv" not in path and "test_graph" not in path for path in paths))
        saved = pd.read_csv(PREDICTIONS).set_index("TransactionID")
        np.testing.assert_array_equal(table.model_score, saved.loc[table.TransactionID, "AllFive"])
        self.assertEqual(len(table), 5000)

    def test_exact_shap_record_is_accepted_and_missing_id_has_no_explanation(self):
        evidence = make_evidence_record(self.table, int(self.table.TransactionID.iloc[0]))
        row = checked_shap(evidence, self.payload, self.lookup)
        self.assertEqual(row["transaction_id"], evidence["transaction_id"])
        self.assertTrue(math.isclose(row["model_score"], evidence["model_score"], abs_tol=1e-10))
        self.assertIsNone(checked_shap(evidence, self.payload, {}))

    def test_wrong_model_contract_source_or_transaction_cannot_attach_shap(self):
        evidence = make_evidence_record(self.table, int(self.table.TransactionID.iloc[0]))
        for key in ["model_sha256", "feature_contract_version", "sources"]:
            bad = {**self.payload, key: "mismatch"}
            with self.subTest(key=key), self.assertRaises(ValueError):
                checked_shap(evidence, bad, self.lookup)
        original = self.lookup[evidence["transaction_id"]]
        for key, value in [("transaction_id", -1), ("model_sha256", "wrong"), ("input_sha256", "wrong"),
                           ("feature_contract_version", "wrong"), ("output_scale", "probability")]:
            bad = {evidence["transaction_id"]: {**original, key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                checked_shap(evidence, self.payload, bad)

    def test_broken_additivity_cannot_be_displayed(self):
        evidence = make_evidence_record(self.table, int(self.table.TransactionID.iloc[0]))
        bad = copy.deepcopy(self.lookup[evidence["transaction_id"]])
        bad["feature_contributions"]["TransactionAmt"] += 1
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            checked_shap(evidence, self.payload, {evidence["transaction_id"]: bad})

    def test_export_loader_rejects_wrong_model_and_duplicate_transaction_ids(self):
        for change in ["wrong_model", "duplicate_id"]:
            with tempfile.TemporaryDirectory() as temporary:
                payload = copy.deepcopy(self.payload)
                if change == "wrong_model":
                    payload["model_sha256"] = "wrong"
                else:
                    payload["records"].append(payload["records"][0])
                path = Path(temporary) / "bad_export.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_shap_export(path)

    def test_retrospective_components_remain_disjoint_and_label_free_members(self):
        components, members, edges, summary = load_clusters()
        self.assertEqual(len(components), 9)
        self.assertEqual(len(members), 87)
        self.assertEqual(summary["sample_rows"], 20000)
        self.assertFalse(members.TransactionID.isin(self.table.TransactionID).any())
        self.assertNotIn("isFraud", members)
        self.assertEqual(len(edges), 68)
