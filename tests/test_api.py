"""Real-artifact API checks, including independently frozen Session 20 scores."""

import json
from pathlib import Path
import tempfile
import unittest
import pytest
from unittest.mock import patch

from catboost import CatBoostClassifier
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from api.main import create_app
from api.model import (
    EXPECTED_MODEL_SHA256, FEATURE_COLUMNS, MODEL_PATH, feature_frame, load_model,
)
from api.schemas import TransactionRequest

FIXTURE = Path(__file__).parent / "fixtures/session22_golden.json"


def valid_payload():
    return {"transaction_id": 100001, "transaction_amount": 5000.0,
            "graph_card_prior_degree": 4.0, "graph_product_prior_degree": 120.0,
            "graph_pair_prior_count": 2.0, "graph_card_distinct_products": 3.0}


@pytest.mark.private
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = create_app()
        cls.client = cls.enterClassContext(TestClient(cls.application))
        # Separate binary load and explicit raw feature construction for parity.
        cls.direct_model = CatBoostClassifier()
        cls.direct_model.load_model(str(MODEL_PATH))

    def test_health_and_model_metadata(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_loaded"])
        self.assertEqual(response.json()["model_sha256"], EXPECTED_MODEL_SHA256)
        info = self.client.get("/model-info")
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()["feature_columns"], FEATURE_COLUMNS)
        self.assertEqual(info.json()["request_to_model_features"]["graph_pair_prior_count"],
                         "graph_card_product_prior_count")
        self.assertIn("no scaler or log transform", info.json()["preprocessing"])

    def test_prediction_matches_direct_raw_model_and_has_no_decision(self):
        response = self.client.post("/predict", json=valid_payload())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        direct = self.direct_model.predict_proba(pd.DataFrame(
            [[5000., 4., 120., 2., 3.]], columns=self.direct_model.feature_names_))[0, 1]
        self.assertAlmostEqual(body["risk_score"], direct, places=12)
        self.assertEqual(body["transaction_id"], 100001)
        self.assertEqual(body["model_name"], "AllFive")
        self.assertEqual(body["model_sha256"], EXPECTED_MODEL_SHA256)
        self.assertEqual(body["score_type"], "uncalibrated_model_score")
        self.assertEqual(body["decision"], "not_assigned")
        self.assertNotIn("isFraud", body)

    def test_frozen_session20_vectors_match_without_api_preprocessing_helper(self):
        frozen = json.loads(FIXTURE.read_text())
        self.assertEqual(frozen["model_sha256"], EXPECTED_MODEL_SHA256)
        for vector in frozen["vectors"]:
            raw = vector["raw_features"]
            payload = {"transaction_id": vector["transaction_id"], "transaction_amount": raw["TransactionAmt"],
                       "graph_card_prior_degree": raw["graph_card_prior_degree"],
                       "graph_product_prior_degree": raw["graph_product_prior_degree"],
                       "graph_pair_prior_count": raw["graph_card_product_prior_count"],
                       "graph_card_distinct_products": raw["graph_card_distinct_products"]}
            with self.subTest(transaction_id=vector["transaction_id"]):
                response = self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertAlmostEqual(response.json()["risk_score"], vector["expected_score"], places=12)

    def test_null_and_omitted_history_match_native_nan_predictions(self):
        payload = {"transaction_id": 7, "transaction_amount": 12.0}
        expected = self.direct_model.predict_proba(pd.DataFrame(
            [[12., np.nan, np.nan, np.nan, np.nan]], columns=self.direct_model.feature_names_))[0, 1]
        omitted = self.client.post("/predict", json=payload)
        payload.update({name: None for name in valid_payload() if name.startswith("graph_")})
        explicit = self.client.post("/predict", json=payload)
        self.assertEqual(omitted.status_code, 200)
        self.assertEqual(explicit.json(), omitted.json())
        self.assertAlmostEqual(omitted.json()["risk_score"], expected, places=12)

    def test_zero_history_is_preserved_and_amount_is_untransformed(self):
        payload = {key: 0.0 for key in valid_payload() if key.startswith("graph_")}
        payload.update(transaction_id=0, transaction_amount=5000.0)
        frame = feature_frame(TransactionRequest.model_validate(payload))
        np.testing.assert_array_equal(frame.to_numpy(), [[5000., 0., 0., 0., 0.]])
        missing = feature_frame(TransactionRequest(transaction_id=0, transaction_amount=5000.0))
        self.assertTrue(missing.iloc[0, 1:].isna().all())
        self.assertEqual(self.client.post("/predict", json=payload).status_code, 200)

    def test_reordered_fields_and_repeated_ids_do_not_change_scores(self):
        payload = valid_payload()
        first = self.client.post("/predict", json=payload).json()
        reordered = self.client.post("/predict", json=dict(reversed(list(payload.items())))).json()
        self.assertEqual(first, reordered)

    def test_required_fields_and_strict_types(self):
        for field in ["transaction_id", "transaction_amount"]:
            missing = valid_payload()
            missing.pop(field)
            with self.subTest(missing=field):
                self.assertEqual(self.client.post("/predict", json=missing).status_code, 422)
        for field, value in [("transaction_amount", "5000"), ("transaction_amount", "banana"),
                             ("transaction_amount", None), ("transaction_amount", True),
                             ("transaction_id", True), ("transaction_id", 1.5),
                             ("transaction_id", "1"), ("graph_pair_prior_count", "2")]:
            with self.subTest(field=field, value=value):
                self.assertEqual(self.client.post("/predict", json={**valid_payload(), field: value}).status_code, 422)

    def test_negative_inputs_and_unexpected_fields_are_rejected(self):
        for field in valid_payload():
            with self.subTest(negative=field):
                self.assertEqual(self.client.post("/predict", json={**valid_payload(), field: -1}).status_code, 422)
        for field in ["isFraud", "label", "TransactionAmt", "graph_card_product_prior_count", "unknown"]:
            with self.subTest(extra=field):
                self.assertEqual(self.client.post("/predict", json={**valid_payload(), field: 1}).status_code, 422)

    def test_nonfinite_values_return_422_with_valid_json_errors(self):
        for field in ["transaction_amount", "graph_card_prior_degree", "graph_product_prior_degree",
                      "graph_pair_prior_count", "graph_card_distinct_products"]:
            for value in [float("nan"), float("inf"), float("-inf")]:
                # Raw text tests nonstandard JSON NaN/Infinity as well as an
                # overflowing JSON numeric literal, without HTTPX prevalidation.
                body = json.dumps({**valid_payload(), field: value})
                with self.subTest(field=field, value=value):
                    response = self.client.post("/predict", content=body, headers={"Content-Type": "application/json"})
                    self.assertEqual(response.status_code, 422)
                    self.assertIn("detail", response.json())
                    self.assertNotIn("input", response.json()["detail"][0])
        response = self.client.post("/predict", content='{"transaction_id":1,"transaction_amount":1e999}',
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)

    def test_malformed_json_is_rejected(self):
        response = self.client.post("/predict", content="{broken", headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)

    def test_invalid_model_output_never_returns_a_score(self):
        classifier = self.application.state.serving_model.classifier
        for value in [np.nan, np.inf, -0.1, 1.1]:
            with patch.object(classifier, "predict_proba", return_value=np.array([[.5, value]])):
                self.assertEqual(self.client.post("/predict", json=valid_payload()).status_code, 500)

    def test_openapi_and_interactive_documentation(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        self.assertEqual(set(schema["paths"]), {"/health", "/model-info", "/predict"})
        request = schema["components"]["schemas"]["TransactionRequest"]
        self.assertEqual(set(request["required"]), {"transaction_id", "transaction_amount"})
        self.assertFalse(request["additionalProperties"])


@pytest.mark.private
class StartupTests(unittest.TestCase):
    def test_model_loads_once_per_lifespan_and_is_cleared_on_shutdown(self):
        application = create_app()
        with patch("api.main.load_model", wraps=load_model) as loading:
            with TestClient(application) as client:
                client.post("/predict", json=valid_payload()).raise_for_status()
                client.post("/predict", json=valid_payload()).raise_for_status()
                self.assertEqual(loading.call_count, 1)
            self.assertIsNone(application.state.serving_model)

    def test_missing_model_or_wrong_hash_prevents_startup(self):
        with tempfile.TemporaryDirectory() as temporary:
            application = create_app(Path(temporary) / "missing.cbm")
            with self.assertRaisesRegex(RuntimeError, "missing"), TestClient(application):
                pass
        with self.assertRaisesRegex(RuntimeError, "SHA-256"), TestClient(create_app(expected_sha256="0" * 64)):
            pass

    def test_feature_order_mismatch_prevents_startup(self):
        real = load_model()
        with patch("api.model.FEATURE_COLUMNS", list(reversed(FEATURE_COLUMNS))):
            with self.assertRaisesRegex(RuntimeError, "feature names/order"):
                load_model()
        self.assertEqual(real.classifier.feature_names_, FEATURE_COLUMNS)

    def test_uninitialized_app_is_not_reported_as_ready(self):
        # A context manager starts lifespan; deliberately omit it to check guards.
        client = TestClient(create_app())
        self.addCleanup(client.close)
        for method, path, kwargs in [("get", "/health", {}), ("get", "/model-info", {}),
                                     ("post", "/predict", {"json": valid_payload()})]:
            self.assertEqual(getattr(client, method)(path, **kwargs).status_code, 503)
