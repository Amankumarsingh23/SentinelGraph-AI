# Session 22: local FastAPI model serving

The API serves the saved Session 20 `AllFive.cbm` through `GET /health`,
`GET /model-info` and `POST /predict`. It loads the model once during application
startup, checks its pinned SHA-256, binary class order and exact feature names,
and reuses it for requests. Startup fails if the artifact is missing or differs
from the serving contract. No datasets, training scripts, MLflow server or graph
state are needed to start the API.

## Run and verify

Run from the repository root in PowerShell with the existing `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-serving.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000/docs> for interactive documentation. The machine-readable
schema is at `/openapi.json`. Stop the server with Ctrl+C. For development,
`--reload` can replace `--workers 1`; the verified command above uses one worker
without automatic reload. If port 8000 is occupied, choose another port for both
the server and verification script.

In another terminal, while the server is running:

```powershell
.\.venv\Scripts\python.exe api/verify_session22.py
.\.venv\Scripts\python.exe -m pytest tests -q --junitxml=outputs/api/session22/test_results.xml
```

`verify_session22.py --port 8001` supports an alternate local port. It checks the
actual HTTP endpoints, docs/OpenAPI, six frozen Session 20 predictions, rejected
inputs and omitted history, then writes `outputs/api/session22/verification.json`.
It does not train models or read raw datasets or reserved test files.

Verified environment: Windows x64, Python 3.13.15, CatBoost 1.2.10, FastAPI
0.143.3, Pydantic 2.14.1, Uvicorn 0.54.0, HTTPX 0.28.1 and pytest 9.1.1.
FastAPI, Pydantic and Uvicorn were already installed. Adding HTTPX, pytest and
their three missing dependencies preserved **all 143 existing package versions**;
`pip check` passed. `requirements-serving.txt` extends the existing project
requirements with the five added pins. Installation reports and environment
comparisons are retained in ignored `outputs/api/session22/`.

## Feature contract and corrections to the session text

The actual Session 20 training code and `config.json` use **raw numerical
features with no log transform or scaler**. The session's `log1p()` explanation
does not match this trained artifact. Applying it would change predictions.
The pair-count column also has a different internal name. The API uses this
explicit mapping, in the exact order expected by the saved model:

| External request field | CatBoost column |
|---|---|
| `transaction_amount` | `TransactionAmt` |
| `graph_card_prior_degree` | `graph_card_prior_degree` |
| `graph_product_prior_degree` | `graph_product_prior_degree` |
| `graph_pair_prior_count` | `graph_card_product_prior_count` |
| `graph_card_distinct_products` | `graph_card_distinct_products` |

`transaction_id` is required metadata and never becomes a model input. It must
be a nonnegative integer. Amount is required, non-null, finite and nonnegative.
Historical fields are optional finite nonnegative numbers: omitted or JSON null
becomes NaN, using CatBoost's native numerical missing-value behavior; zero stays
zero. Missing history is not filled with zero. Counts remain numerical fields
as in the session schema; this interface does not enforce integer-valued counts
or verify their historical truth.

Strict Pydantic validation rejects numeric strings, booleans, negative values,
NaN/Infinity, malformed bodies and unexpected fields, including `isFraud`.
The API returns HTTP 422 with field locations and messages. Validation errors
do not echo raw transaction values; this also permits valid JSON errors when
the rejected input contains a nonfinite number. A model that has not completed
startup returns 503. Inference failures or invalid scores return 500.

The pinned serving model fingerprint is:

```text
4738766c01b72de4b113d922f5236a9f7d979286b2994ff854cce99e7ddd44ed
```

The same binary is retained in Session 21's registration. Every prediction
includes this hash and feature contract version `session22-v1`. Changing the
model requires an intentional update to the expected hash and contract, with
new verification. `/model-info` exposes the mapping, missing-value policy,
preprocessing, score meaning and development scope.

## Actual request and result

```powershell
$predictionBody = @{
    transaction_id = 100001
    transaction_amount = 5000.0
    graph_card_prior_degree = 4.0
    graph_product_prior_degree = 120.0
    graph_pair_prior_count = 2.0
    graph_card_distinct_products = 3.0
} | ConvertTo-Json

Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -Body $predictionBody
```

The verified response contains:

```json
{
  "transaction_id": 100001,
  "risk_score": 0.012703279408038082,
  "model_name": "AllFive",
  "model_sha256": "4738766c01b72de4b113d922f5236a9f7d979286b2994ff854cce99e7ddd44ed",
  "feature_contract_version": "session22-v1",
  "score_type": "uncalibrated_model_score",
  "decision": "not_assigned"
}
```

This is an uncalibrated model score, not an adjudicated fraud label or a
calibrated probability. No decision threshold or Session 19 calibrator is
applied. The AllFive model is a serving demonstration selected from development
ablations, not a finalized production model.

## Verification evidence

All **16 API tests passed** and the complete suite passed **97 tests**. Checks
cover successful startup and one load per lifespan, missing/wrong model hashes,
feature-order mismatch, strict inputs, nonfinite error responses, optional
history, zero preservation, direct-model parity, repeated/reordered requests,
score bounds, metadata, readiness and OpenAPI.

`tests/fixtures/session22_golden.json` freezes six feature vectors from different
positions in the original Session 20 development evaluation interval. Their
expected scores were copied from the existing `predictions.csv`, independently
of the API and its preprocessing helper. The fixture records the model hash,
prediction-file hash and original evaluation-input fingerprint. Both tests and
the live HTTP verifier match all six scores within numerical tolerance. This
adds an early golden-vector check; it does not replace broader Session 25
integration work or establish fresh-training reproducibility.

The HTTP verifier confirmed all three endpoints, docs/OpenAPI and frozen-score
parity on the running loopback server. Saved evidence:

- `outputs/api/session22/verification.json`: live requests and actual scores.
- `outputs/api/session22/test_results.xml`: full-suite test results.
- `outputs/api/session22/environment_changes.json`: unchanged prior packages.

The installed Starlette version reports a deprecation warning for its current
HTTPX TestClient adapter. Tests pass; no dependency replacement was needed.
Existing PyTorch/PyG deprecation warnings also remain in the full suite.

## Scope and references

This is a local internal demonstration accepting **precomputed historical
features**. It has no trusted online feature store, authentication, rate
limiting or financial decision policy. `card1` is an anonymized proxy, not a
verified customer identity. Repeated scoring requests do not update counters
or other transaction state; a future stateful feature service will need its
own idempotency and timestamp-consistency guarantees. No model was retrained
and no reserved test observations were read for this session.

Implementation references: FastAPI's [model-loading lifespan](https://fastapi.tiangolo.com/advanced/events/),
[testing guide](https://fastapi.tiangolo.com/tutorial/testing/) and Pydantic's
[strict-mode validation](https://docs.pydantic.dev/latest/concepts/strict_mode/).
