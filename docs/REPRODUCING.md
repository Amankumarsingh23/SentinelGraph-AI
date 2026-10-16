# Reproduction and private artifacts

## Public workflow

Use Python 3.13 and `requirements-public.txt` in a fresh environment. Run
`python -B scripts/check_public_snapshot.py` and `python -B -m pytest -q`.
No data, saved model, private fixture, MLflow store, Streamlit installation or
server is required. Dependencies are pinned; `constraints-public.txt` records the
tested fresh public closure. Historical requirements preserve the old Windows
stack and are not Linux locks. Linux hosted CI is configured but unverified
until it runs. Other Python versions and GPU operation are not verified.

Private tests are marked `private` and deselected by default. The private-only
Streamlit module is not imported in the public tier. A guard rejects Python file
access to project-private paths in public tests. This is a regression boundary,
not an OS security sandbox. Use pytest, not bare unittest discovery, to apply it.

## Existing authorized workspace

Do not recreate/update a preserved environment. Use its existing compatible
interpreter, or install `requirements-apps.txt` in a new separate environment.

```powershell
python -B -m pytest --run-private -q
```

This runs both tiers, including original golden assertions. Missing/mismatched
private artifacts fail rather than skip. This command does not retrain saved
models or open the reserved split. Toy tests fit small synthetic models in memory
and write temporary synthetic files.

| Checks | Private prerequisites |
|---|---|
| API/temporal model path | Exact pinned `outputs/evaluation/session20/AllFive.cbm` |
| Independent golden comparisons | Original `tests/fixtures/session22_golden.json` |
| Saved dashboard/SHAP/UI | Processed validation, Session 13 graph features, Session 20 config/predictions/model, Session 24 SHAP export |
| Retrospective clusters | Session 16 summary/components/memberships/edges |
| Monitoring/UI | Session 23 reports and referenced train/validation inputs |
| Full historical release provenance | Original manifests and retained Session 21 registration/MLflow store, beyond pytest alone |

Do not obtain these from an unapproved public bundle. Source alone cannot run
the frozen research-model application. Rebuilding a model does not authorize
changing its expected fingerprint.

## Authorized data and fresh research generation

Obtain IEEE-CIS data through your own authorized Kaggle account:
<https://www.kaggle.com/competitions/ieee-fraud-detection/data>.
Review the provider's access/sharing terms. Do not commit records, credentials,
processed rows or model outputs. The identity table is unused by this pipeline.

Generation is separate research work in a **new** authorized workspace, never
rerunning scripts over preserved Session 26 files. Publication setup does not
generate research artifacts or authorize reserved-test access.

Historical dependency order and actual entry points:

1. `temporal_split.py` reads the full original transaction source and writes all
   chronological splits, including the reserved interval. Its aggregate outputs
   are part of historical test exposure, not an untouched-test guarantee.
2. `feature_engineering.py` exposes the historical feature function used by tests.
   Its legacy CLI also reads the reserved split. Do not run that CLI when reserved
   access is prohibited. The development-only paths below require already
   authorized processed train/validation inputs.
3. `python graph_models/temporal_graph_features.py` builds Session 13 features
   using processed train/validation only.
4. `python -m deep_learning.run_behavioral` and
   `python graph_models/train_temporal_gnn.py` run independent sequence/GNN
   experiments; neither is required for current API inference.
5. `python graph_models/discover_candidate_clusters.py` creates retrospective
   training-sample component evidence.
6. `python anomaly_models/train_anomaly_models.py`, then
   `python fusion_models/train_fusion.py`, generate anomaly/fusion artifacts and
   define Session 18 evaluation IDs.
7. `python evaluation/session20_ablations.py` creates matched CatBoost ablations
   against those IDs and saves four fitted models.
8. `python tracking/register_session20.py` logs completed results to local MLflow.
   `python monitoring/check_drift.py` writes offline diagnostics;
   `python dashboard/export_shap.py` writes keyed SHAP evidence.

These entry points write artifacts and some train models: they are **not** public
setup/verification commands. Review their arguments/outputs before a separately
authorized fresh research run. Root baseline/EDA scripts can execute work at
import time; do not import them as setup checks. Full baseline experiments also
need LightGBM from the historical requirements, beyond the public test tier.

This is not a fully automated recreation of the frozen release. Original training
Git provenance was not captured; inputs are private and exact cross-platform
retraining is not guaranteed. The six-vector fixture preserves independent prior
predictions: regenerating expected scores from the current model would not recreate
that independent evidence. New research needs a new reviewed candidate, contract,
provenance and verification; do not relabel it Session 26.

## Preserving release history

Old manifests fingerprint source too, so README/test changes correctly invalidate
their comparison against today's tree. Artifact-byte preservation and current
source verification are separate. The ignored publication record compares original
artifact/output hashes, dataset/environment metadata and the Git index against
a pre-edit baseline. Metadata checks do not open reserved dataset contents.

Legacy release CLIs write manifests/reports. Do not run them against preserved
output locations or update old expected hashes. A future release needs a distinct
output location and review. Publication verification is not final research or
production approval.
