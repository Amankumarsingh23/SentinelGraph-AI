"""Verify the running loopback API against frozen Session 20 scores."""

import argparse
import json
import math
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    golden = json.loads((ROOT / "tests/fixtures/session22_golden.json").read_text())
    checks = []
    with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=20, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json()["model_loaded"] is not True or health.json()["model_sha256"] != golden["model_sha256"]:
            raise ValueError("Health does not identify the expected loaded model")
        info = client.get("/model-info")
        info.raise_for_status()
        metadata = info.json()
        if metadata["model_sha256"] != golden["model_sha256"] or metadata["score_type"] != "uncalibrated_model_score":
            raise ValueError("Unexpected serving model metadata")
        for path in ["/docs", "/openapi.json"]:
            client.get(path).raise_for_status()
        for vector in golden["vectors"]:
            raw = vector["raw_features"]
            payload = {"transaction_id": vector["transaction_id"], "transaction_amount": raw["TransactionAmt"],
                       "graph_card_prior_degree": raw["graph_card_prior_degree"],
                       "graph_product_prior_degree": raw["graph_product_prior_degree"],
                       "graph_pair_prior_count": raw["graph_card_product_prior_count"],
                       "graph_card_distinct_products": raw["graph_card_distinct_products"]}
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            body = response.json()
            score = body["risk_score"]
            if (not math.isclose(score, vector["expected_score"], rel_tol=1e-10, abs_tol=1e-12)
                    or body["transaction_id"] != vector["transaction_id"]
                    or body["model_sha256"] != golden["model_sha256"]
                    or body["decision"] != "not_assigned"
                    or body["score_type"] != "uncalibrated_model_score"):
                raise ValueError("API prediction differs from frozen Session 20 evidence")
            checks.append({"transaction_id": vector["transaction_id"], "risk_score": score,
                           "expected_score": vector["expected_score"], "status": "PASS"})
        sample = {"transaction_id": 100001, "transaction_amount": 5000.0,
                  "graph_card_prior_degree": 4.0, "graph_product_prior_degree": 120.0,
                  "graph_pair_prior_count": 2.0, "graph_card_distinct_products": 3.0}
        example = client.post("/predict", json=sample)
        example.raise_for_status()
        for invalid in [{**sample, "transaction_amount": -1}, {**sample, "isFraud": 1},
                        {**sample, "transaction_amount": "5000"}]:
            response = client.post("/predict", json=invalid)
            if response.status_code != 422:
                raise ValueError("Invalid request was not rejected with HTTP 422")
        missing = client.post("/predict", json={"transaction_id": 7, "transaction_amount": 12.0})
        missing.raise_for_status()
    report = {"status": "PASS", "base_url": f"http://127.0.0.1:{args.port}",
              "health": health.json(), "model_info": metadata, "golden_vectors": checks,
              "example_prediction": example.json(), "invalid_request_checks": 3,
              "missing_history": "PASS", "training_reexecuted": False, "test_access": False}
    output = ROOT / "outputs/api/session22"
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PASS: all three endpoints, docs/OpenAPI, {len(checks)} frozen vectors, invalid requests and missing history")
    print("Actual session example:", json.dumps(example.json()))
    print("Report:", output / "verification.json")


if __name__ == "__main__":
    main()
