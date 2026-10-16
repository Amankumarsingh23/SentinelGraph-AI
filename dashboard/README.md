# Session 24: investigation dashboard and grounded evidence

The Streamlit dashboard combines saved Session 20 predictions, original
validation observations, Session 13 historical features and separately verified
AllFive SHAP explanations. It is a read-only local development demonstration.
The summary is deterministic Python, not an LLM; no paid API is used.

## Run

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe dashboard/export_shap.py
.\.venv\Scripts\python.exe dashboard/verify_session24.py
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open <http://127.0.0.1:8501>. Stop the server with Ctrl+C. Choose another
`--server.port` if this port is occupied. No FastAPI or MLflow server is needed;
the dashboard reads local saved artifacts and does not call a prediction API.

Verified on Windows x64 / Python 3.13.15 / Streamlit **1.66.0**. Installation
added six pinned packages and preserved **all 148 previously installed package
versions**. The pins extend the existing serving requirements. Installation and
environment-change reports are in ignored `outputs/dashboard/session24/`.
`pip check` passed.

## Views

**Transaction evidence:** a table ranked by saved uncalibrated AllFive score,
with TransactionID as the tie-breaker. Select any of 5,000 records to update its
score, amount, historical feature snapshot, grounded summary and matching SHAP
chart. The JSON evidence record and source fingerprints are available beside
the summary and can be downloaded. No retrospective fraud label appears in the
ranked table, selected evidence record or summary. No threshold or final fraud
decision is assigned.

The historical indicators describe earlier observed events, as saved by the
point-in-time Session 13 pipeline. They are not counts recomputed from later
history. Missing values are displayed as unavailable, never invented or filled
with zero. `card1` denotes an anonymized proxy, not a verified customer.

**Retrospective clusters:** a separate Session 16 training-sample research view.
The nine components cover only **87 of 20,000** sampled transactions. Select a
component to inspect observed card/address proxy links and membership IDs.
Retrospective aggregate fraud statistics are hidden unless explicitly enabled;
the view explains that those outcomes were evaluated after graph construction.
The largest component has 57 transactions, 27 card proxies and 13 address
proxies, with retrospective fraud share 7/57 (12.28%). These components are not
joined to Session 20 validation transactions or treated as inference-time evidence.

**Monitoring:** the saved Session 23 feature PSI, missingness rates and score PSI
for the same frozen model. These are offline diagnostics, not live service
observability or evidence that model accuracy declined.

## Evidence verification

`evidence.py` uses the repository's actual
`outputs/evaluation/session20/predictions.csv`, rather than the tutorial's
`ablation_predictions.csv`. It uses the actual internal pair-count column
`graph_card_product_prior_count`; `graph_pair_prior_count` is the API alias.

The loader checks unique/nonmissing IDs, complete prediction-to-source matches,
finite matching timestamps, finite scores in [0, 1], and retrospective label
agreement against the authoritative validation split. Labels are removed after
that check. It also reconstructs and matches the **original Session 20 ordered
evaluation-input fingerprint**, verifies the AllFive feature contract, checks
the pinned model SHA-256 and records hashes of each source file.

The interface fails with an explanatory message if core evidence cannot be
verified. Cached evidence is keyed by file paths, sizes and nanosecond modification
times so changed artifacts trigger reloading. Optional absent/mismatched SHAP
or retrospective artifacts do not get silently substituted.

Evidence includes the transaction ID/time, amount, observed anonymized proxies,
saved model score, model hash, feature contract version, historical feature
values, input fingerprint, sources and limitations. The summary uses only these
recorded facts and explicitly states that no fraud decision has been assigned.
It does not invent identities, links, causes or intent.

## SHAP export

Session 19 explained a different, Session 18 CatBoost model. Those explanations
are not used by this dashboard. `export_shap.py` loads the **Session 20 AllFive**
binary and the shared Session 22 raw-feature adapter, then computes native
CatBoost SHAP for all **5,000 evaluation records** outside the UI.

Each explanation records TransactionID, model SHA-256, feature contract,
input fingerprint, all five contributions, measured baseline, raw model output
and score. The export also retains source hashes and output scale
`raw_log_odds`. Before display, it must match the selected evidence's ID,
model hash, contract, source hashes and exact input fingerprint. Duplicate
export IDs are rejected. Baseline plus contributions must reconstruct the raw
output, whose sigmoid must reproduce the selected saved score. A missing or
invalid explanation is withheld while the matched transaction remains visible.

Observed verification errors:

- Maximum absolute raw-output reconstruction error: **1.3322676295501878e-14**.
- Maximum absolute difference from saved Session 20 scores: **9.974659986866641e-17**.

SHAP contributions explain the fitted model's raw output relative to its
baseline. They are not probability-point changes, causal effects or evidence
of fraudulent intent. Positive values increase raw model output.

## Tests and audit outputs

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_evidence.py tests/test_dashboard_ui.py -v
.\.venv\Scripts\python.exe -m pytest tests -q --junitxml=outputs/dashboard/session24/test_results.xml
```

All **18 dashboard tests passed**, and the full suite passed **130 tests**.
Evidence tests cover grounded/missing summaries, zero preservation, shuffled
sources, exact score agreement, duplicate/unmatched IDs, stale timestamps,
label mismatch, invalid scores, missing IDs, reserved-test exclusion, wrong
SHAP IDs/model/input/contract/scale, duplicate export IDs and broken additivity.
Five Streamlit AppTest checks execute transaction selection, graph-feature and
summary updates, missing evidence, invalid SHAP, separate retrospective outcomes
and monitoring views. These programmatically exercise the application script;
they are not browser screenshot tests. See Streamlit's official
[AppTest documentation](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).

The live loopback server returned HTTP 200 for both its page and health endpoint.
The separate audit reproduced all 5,000 saved model scores, validated all 5,000
keyed explanations and checked the nine components, 87 memberships and 68 edges.
The research and evaluation populations have **zero overlapping IDs**.

Generated evidence is ignored under `outputs/dashboard/session24/`:

- `shap_export.json`: exact keyed explanations and source metadata.
- `verification.json`: complete saved-evidence/SHAP/component audit.
- `test_results.xml`: full-suite results.
- `ui_http_verification.json`: live server page/health checks.
- `environment_before.json`, `environment_changes.json`,
  `installed_constraints.txt`, `install_report.json`: dependency-install evidence.

Existing third-party deprecation warnings remain in the full suite. No models
were retrained, source data/labels rewritten, graph state updated or reserved
test observations loaded. This dashboard has no authentication, production
hosting, trusted online feature construction or real delayed-label workflow.
It does not establish final model selection or independent final-test quality.
