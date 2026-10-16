# Session 25: development release and system integration

This workflow freezes and verifies the supported demonstration: precomputed
historical features → FastAPI score → saved dashboard evidence and keyed SHAP,
with offline drift monitoring. It does not create a live feature store or run
the reserved test. AllFive remains a **development demonstration candidate**,
not an approved final research winner or deployed financial decision system.

## Build and verify

Use the existing [dashboard environment](../dashboard/README.md); no new
dependencies are needed. From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe integration/build_release.py
.\.venv\Scripts\python.exe integration/verify_release.py
```

The build needs the existing Session 20 model/config/reports, processed train
and validation CSVs, Session 13 graph features, Session 21 retained registration
snapshot and local MLflow store, Session 22 golden fixture, Session 23 drift
reports and Session 24 SHAP/component artifacts. It uses the actual
`predictions.csv` and `graph_card_product_prior_count` names in this repository.
Missing or inconsistent prerequisites fail rather than getting regenerated.
Build and verification never retrain or rewrite source data, fitted models,
predictions, SHAP, graph or monitoring artifacts.

Every build creates a unique release ID with an immutable manifest under:

```text
outputs/integration/session25/
  release_manifest.json                     # latest-build convenience copy
  releases/<release-id>/release_manifest.json # preserved release identity
  verifications/<verification-id>/           # verification of the latest copy
    verification.json
    test_results.xml
    tests.log
    api_server.log
  verification.json                         # latest verification result
```

When verifying a retained manifest, results are stored next to that retained
manifest instead. For example:

```powershell
.\.venv\Scripts\python.exe integration/verify_release.py --manifest outputs/integration/session25/releases/<release-id>/release_manifest.json
```

Replace `<release-id>` with the printed directory name. Rebuilding explicitly
creates a new candidate and moves the convenience copy; it never changes an
older manifest. Verification does not rewrite hashes to accept a mismatch.
Investigate changes and, when intended, build a new release. Keep the immutable
manifest path when referring to a particular candidate.

`--integrity-only` checks frozen files/contracts/environment without the full
system audit and reports `PASS_INTEGRITY_ONLY`. It does not satisfy the combined
release checkpoint. Only the default complete workflow reports `PASS` with
the eight development checklist entries.

## Manifest identity and provenance

The manifest fingerprints 24 required artifact/protocol records, plus the
relevant Python sources, component guides, root README, tests, golden fixture
and dependency pins. It records the complete feature/request/response contract,
model SHA-256, candidate evaluation protocol, Python/platform/package versions
and build-time Git state. Source inventory changes, changed or missing bytes,
path traversal, incompatible contracts, changed installed packages and
unsupported production/final-test claims are rejected.

The release compares its model, development predictions, training config and
four train/validation data/graph files with the retained **Session 21** manifest.
The full verifier also retrieves the AllFive model, prediction CSV and config
from the exact finished MLflow run and checks their bytes. A later registration
does not silently become the provenance record for an existing release.
The SQLite database itself is not frozen as a static file; experiment history
can grow while the selected run's recorded artifacts must remain consistent.

These checks enforce a local development snapshot, not cryptographic release
signing or historical training provenance. Git currently has no commits, so
the commit is recorded as unavailable; no commit is invented or created.
Session 21 recorded existing artifacts after training. Release-build source
hashes identify today's implementation, not the missing original training-code
revision. Original Session 20 ordered evaluation-input fingerprints are checked
through the evidence loader.

## Combined verification

The default verifier performs these checks in one command:

1. Checks every frozen artifact, source inventory/hash, environment and protocol.
2. Retrieves the three earlier MLflow artifacts and compares them with the release.
3. Runs the complete pytest suite, saves its log/JUnit report, rejects skipped
   checks and confirms the API, temporal, dashboard, monitoring and release
   modules actually ran.
4. Audits all 5,000 saved dashboard scores and keyed SHAP records, including
   input/model/ID matching and raw-score additivity. It verifies the separate
   nine-component/87-transaction retrospective research population.
5. Audits all 20,000 monitoring scores, fixed training bins, counts, missingness,
   PSI and input fingerprints using the shared raw-feature adapter.
6. Starts the current release API in its own temporary loopback server, sends
   six frozen golden requests through real HTTP, and matches API scores with
   the independent fixture, dashboard evidence, SHAP and monitoring adapter.
   It rejects a supplied fraud label and checks model/version/no-policy identity.
7. Stops only its own API process and checks frozen files/environment again.

Existing API, MLflow UI and Streamlit processes are not stopped or reused as
the release API. The fresh server uses an available loopback port and writes
its log into the verification directory. Test and audit reports are the only
new outputs. Data/model artifacts remain read-only. Failed checks preserve a
FAILED report with the error, rather than a successful checklist. Full reports
retain the actual pytest count and per-component audit counts.

The eight checklist entries are artifact integrity, feature contract, golden
HTTP predictions, temporal safety, evidence integrity, research scope,
monitoring parity and candidate evaluation-protocol documentation. Passing
them verifies this **development workflow**. `final_test_ready` remains false.

## Temporal integration

`tests/test_temporal_integration.py` includes an independent strict-time oracle
that filters only observations with `history time < target time`; it does not
copy the generator's mutable-counter implementation. It compares the oracle
against the actual Session 13 `temporal_graph_features` function.

For one card proxy at times 100, 200, 200 and 300, both implementations must
produce **0, 1, 1, 3**. Additional checks cover shuffled input/ID order, unknown
proxies, future/label invariance and split-crossing history. A synthetic path
passes actual generated features through the monitoring adapter, frozen model,
FastAPI and dashboard evidence and verifies identical scores. It does not read
or simulate an actual reserved-test sample, distributed state store or live
late-event arrival stream.

Run only the new focused checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_release.py tests/test_temporal_integration.py -v
```

There are eight release tests and five temporal integration tests, in addition
to the existing suite. Failure cases include changed bytes of the same length,
missing files, escaping paths, invalid release scope, altered contracts, changed
MLflow model/input fingerprints and attempted test-optimization flags.

## Candidate final-evaluation protocol

[`evaluation_protocol.json`](evaluation_protocol.json) is included in every
manifest. It proposes the exact existing AllFive artifact and feature contract,
AP as primary metric, ROC-AUC, Precision/Recall@100, raw-score Brier, latency,
five chronological diagnostics and an illustrative no-alert cost baseline.
Ranking ties use score descending then TransactionID ascending.

The cost diagnostic retains **missed-fraud amount + 200 × false positives**,
full loss for missed fraud and zero residual cost for caught fraud. It is not
validated financial loss. The no-alert strategy is a comparison baseline;
the API decision policy remains null/not assigned. No threshold/calibrator is
selected or optimized using reserved outcomes.

For a future separately authorized final evaluation, permitted history would
include earlier unlabeled train/validation events and, under the frozen
online-style protocol, strictly earlier test events. Same-timestamp peers must
all read history before any of them update it. TransactionID is not a source
of earlier event time. Labels may only be joined after feature computation and
predictions are fixed. No reserved-interval feature/evaluation script is added
or executed by Session 25.

Final model selection and the planned test workflow remain pending. Before
one-time final testing, approve the final candidate using development evidence,
implement and independently verify its reserved-interval feature/scoring path,
and freeze the intended policy/scope. Report all results and limitations without
iterating on test outcomes. Earlier full-dataset EDA exposed aggregate test-label
statistics, so the project must not claim a pristine untouched test history.
This session itself never opens reserved test files. The amount-only ablation,
reused validation, anonymous proxies and unavailable training revision remain
explicit limitations in the release protocol.
