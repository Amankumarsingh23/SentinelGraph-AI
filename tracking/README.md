# Session 21: MLflow and reproducibility

The four completed Session 20 CatBoost ablations are registered in the local
`SentinelGraph_Session20_Ablations` experiment. Registration does not fit models.
It first reproduces saved predictions and checks the original ordered-input
fingerprints, then logs parameters, metrics, models, reports and provenance.

## Environment and commands

Verified on Windows x64, Python 3.13.15 and **MLflow 3.17.1**. The installation
used constraints for all 93 previously installed packages: none changed, and
50 packages were added. `pip check` passed. Installation evidence is retained
under ignored `outputs/tracking/`: `environment_before.json`,
`installed_constraints.txt`, `mlflow_install_report.json`, and
`environment_changes.json`.

`requirements-tracking.txt` pins the installed MLflow dependency closure and
includes the existing graph/behavioral/core requirements. For this environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tracking.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tracking/register_session20.py
.\.venv\Scripts\python.exe tracking/verify_session21.py
.\.venv\Scripts\python.exe tracking/serve_ui.py
```

Run from the repository root. The UI helper uses the same absolute SQLite URI
as registration, handles spaces in this workspace path, and binds one worker to
<http://127.0.0.1:5000>. Stop it with Ctrl+C; use `--port 5001` if necessary.
MLflow's optional job-execution backend is unavailable on Windows; experiment
tracking, the UI and artifact verification were checked successfully here.

This implementation uses **SQLite**, following MLflow's current
[backend-store guidance](https://mlflow.org/docs/latest/tracking/backend-stores/),
with local file artifacts. The session's example `outputs/tracking/mlruns`
FileStore command points to a different store; use the helper above instead.
Usage telemetry is disabled by default in the tracking helpers.

The complete suite passed **81 tests**, including seven tracking checks covering
missing models, changed bytes, metric disagreement, overlapping timestamps,
incomplete/duplicate run manifests, an unborn Git HEAD and paths with spaces.
The current environment was verified; a clean installation on a second machine
and a fresh training reproduction have not been performed.

## Required data and historical results

Registration requires these existing local inputs:

- `data/processed/train.csv` and `validation.csv` from the chronological split.
- `outputs/graph/train_graph_features.csv` and `validation_graph_features.csv`
  from Session 13, joined by TransactionID with timestamp checks.
- `outputs/fusion/session18/evaluation_predictions.csv`, fixing the same
  5,000 evaluation IDs as Session 20.
- All ten reports listed in `register_session20.REPORT_FILES`, plus the four
  `.cbm` models, under `outputs/evaluation/session20/`.

For a new authorized dataset checkout, follow the root [data setup](../README.md#data-and-setup)
and run `temporal_split.py` and `feature_engineering.py`; rebuild point-in-time
graph counters with:

```powershell
.\.venv\Scripts\python.exe graph_models/temporal_graph_features.py
```

The [Session 17 anomaly](../anomaly_models/README.md) and
[Session 18 fusion](../fusion_models/README.md) commands create the reference
evaluation file. The [Session 20 guide](../evaluation/README.md) supplies the
exact ablation command: 50,000 training targets, 400 trees, depth 6, learning
rate 0.05, seed 42, two threads, 1,000 bootstrap draws and five time blocks.
Those training commands belong to rebuilding the experiment; they are not run
by the Session 21 registration or verification scripts.

The preflight checks required files before initializing tracking, validates
unique IDs and strict training/evaluation chronology, recomputes reported
metrics, checks training-time ordered-input fingerprints, reloads all four
models and reproduces all 5,000 probabilities per model within numerical
tolerance. Raw data files are fingerprinted locally; copies are not uploaded.
Reserved test data is not read by this session.

## Registered results

| Model | Average Precision | Precision@100 | Recall@100 |
|---|---:|---:|---:|
| AmountOnly | 0.048227 | 0.10 | 0.062893 |
| AmountPlusActivity | 0.144062 | 0.16 | 0.100629 |
| AmountPlusRelationships | 0.146785 | 0.17 | 0.106918 |
| AllFive | 0.190858 | 0.21 | 0.132075 |

All four use the same 50,000 training rows and 5,000 development evaluation
rows (159 frauds). Logged training/inference durations are the original saved
Session 20 measurements, not registration timings. AP deltas, feature order,
preprocessing, row counts and bootstrap settings are retained as well.

These are reused development results. `card1` denotes an anonymized proxy,
not verified customers; graph relationships do not establish real identities.
No final-test evaluation, production model selection or deployment is implied.
DVC is optional in the session and has not been installed or configured.

## Store, manifest and verification

Generated tracking files are covered by the existing `outputs/` Git ignore:

```text
outputs/tracking/
  mlflow.db                         # experiment/run metadata
  artifacts/                        # MLflow-owned model/report/source copies
  experiment_manifest.json          # latest successful registration
  verification.json                 # latest explicit verification
  ui_verification.json              # local HTTP/API smoke-check evidence
  registrations/<registration-id>/  # retained registration snapshots
    experiment_manifest.json
    reports/
    provenance/
```

Each successful registration creates four new runs with a shared unique
registration ID. It preserves older runs and immutable registration snapshots;
repeating it intentionally adds history. The latest manifest contains the
exact four run IDs. Verification retrieves those IDs through MLflow, checks
FINISHED status, experiment membership, names, tags, parameters and metrics,
then downloads every listed artifact and verifies its SHA-256 and byte count.
Additional historical runs do not make verification fail.

The executed repeat-registration check retained both batches: **eight historical
runs**, with four selected by the latest manifest. All four latest runs passed
parameter/metric checks and **252 artifact hash checks** (63 per run). The local
UI returned HTTP 200, exposed all eight runs through its API, and listed the
logged model artifacts successfully. Evidence is in `verification.json` and
`ui_verification.json`.

To verify an older snapshot:

```powershell
.\.venv\Scripts\python.exe tracking/verify_session21.py --manifest outputs/tracking/registrations/<registration-id>/experiment_manifest.json
```

Replace `<registration-id>` with the saved directory name. Its verification
report is written beside that manifest. A failed registration is retained as
FAILED and does not replace the latest successful manifest.

Back up the SQLite database, artifacts and manifests together while tracking
writes are stopped. Local artifact URIs and manifest tracking URIs are absolute;
moving the workspace requires updating storage configuration and those paths.
Data must be obtained through the user's authorized source, not from Git.

## Provenance limits

Each run retains the original Session 20 config/reports, current Python and
package inventory, requirements snapshot, Git state, registration-time source
copies and SHA-256 hashes, current input-file hashes, and prediction preflight.
The historical config also retains its original training-time runtime versions.

This repository currently has **no Git commits**. The registration commit is
therefore recorded as null, with an explicit reason and dirty-worktree status.
No original training commit was recorded. Current source snapshots and whole
file hashes describe registration time; they cannot retroactively prove which
code or whole-file bytes were used during training. The original Session 20
ordered-input hashes are separately retained and matched during preflight.

Matching saved predictions establishes artifact reproducibility in this
environment. It does not establish fresh-training reproducibility across
machines or libraries. Future training integrations should capture code,
environment and data provenance during the actual fit.
