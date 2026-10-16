# Session 18: chronological holdout stacking

`train_fusion.py` combines newly trained CatBoost scores with the **frozen**
Session 17 autoencoder and Isolation Forest scores. It implements chronological
holdout stacking on the existing 10,000-row validation subset; it does not
generate walk-forward out-of-fold predictions.

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe fusion_models/train_fusion.py
```

Explicit defaults:

```powershell
.\.venv\Scripts\python.exe fusion_models/train_fusion.py --train-targets 50000 --iterations 200 --seed 42 --threads 2 --anomaly-dir outputs/anomaly/session17 --output outputs/fusion/session18
```

The existing `requirements-behavioral.txt` environment contains the required
dependencies. Session 13 graph feature CSVs and the saved Session 17 models,
scaler, targets, configurations and predictions must already exist. There is no
automatic anomaly retraining or alteration of the original split files.

## Actual input schema and provenance

The tutorial's combined `validation_predictions.csv` does not exist in this
implementation. Instead the script reads the actual Session 17 files:

- `isolation_forest_predictions.csv`: TransactionID, isFraud, model, anomaly_score.
- `autoencoder_predictions.csv`: the same schema.

It joins scores by **unique TransactionID**, validates identical ID sets, and
compares both saved label columns with the original validation labels. Missing,
duplicate, unknown IDs, nonfinite scores and wrong model names are rejected.
It names the aligned columns IsolationForestScore and AutoencoderScore.

The five current/historical CatBoost inputs are TransactionAmt,
graph_card_prior_degree, graph_product_prior_degree,
**graph_card_product_prior_count**, and graph_card_distinct_products. The fourth
is the existing repository name for the tutorial's graph_pair_prior_count.
Graph features are joined through the existing Session 17 loader, including
timestamp/ID alignment checks. They were generated over full strictly earlier
history before selecting model-training rows.

Before fitting fusion models, the script checks that recorded anomaly training
IDs and labels match the original training split and strictly precede all saved
score targets. It recomputes the scaler's mean/scale using only those 50,000
training rows, verifies its fitted row count, then reloads the forest and
autoencoder and reproduces all 10,000 anomaly score values. These operations
perform **no refit**. This checks the recorded local pipeline and its artifacts;
it cannot prove the provenance of arbitrary externally substituted model files.

Session 17 artifact hashes are recorded in the Session 18 configuration. The
original anomaly models, preprocessing and score files are left unchanged.

## Chronological partitions

| Role | Rows | Frauds | Prevalence | TransactionDT range |
|---|---:|---:|---:|---|
| Base-model training | 50,000 | 2,265 | 4.53% | 9,031,128–10,437,996 |
| Meta-model training | 5,000 | 216 | 4.32% | 10,438,003–10,588,784 |
| Evaluation | 5,000 | 159 | 3.18% | 10,588,835–10,712,169 |

CatBoost uses the latest 50,000 training transactions. Its predictions on the
10,000 saved validation IDs are out of sample relative to CatBoost fitting.
The anomaly models were also fitted before this validation period.

The existing `chronological_midpoint` helper places the midpoint **after** any
timestamp ties. Here the original midpoint already separates timestamps strictly,
so the desired 5,000/5,000 allocation is retained. On other inputs a timestamp
batch can enlarge meta-training; the actual counts and boundaries are recorded.
Both resulting portions must contain both binary label classes.

The meta-models fit on the earlier half only; their scalers fit on the earlier
half only. The later half is used for prediction and the final same-row table.
No base-model in-sample predictions become meta-training features.

## Models and score interpretation

CatBoost: 200 iterations, depth 6, learning rate 0.05, Logloss, seed 42 and two
CPU threads. It receives the five **raw numerical inputs**, with infinities
converted to NaN and CatBoost's native missing-value handling. It does not use
the old benchmark's ProductCD/card4/card6 categorical inputs. Its training data,
features and evaluation rows differ from older CatBoost experiments.

The anomaly models retain Session 17 preprocessing: zero replacement, clipping
negative values to zero, log1p and a training-fitted StandardScaler. Their raw
scores are used as meta-features, not naively averaged with CatBoost probabilities.

Three predefined stackers are fitted:

1. CatBoost + autoencoder.
2. CatBoost + Isolation Forest.
3. CatBoost + autoencoder + Isolation Forest.

Each is `StandardScaler -> LogisticRegression`, with C=1, lbfgs, max_iter=1000,
no class weighting and seed 42. Standardization accounts for distinct score
scales. Logistic regression learns a linear combination of standardized scores,
followed by sigmoid. Its fitted output has not been independently calibrated.
No hyperparameters, coefficients or feature subsets are tuned on evaluation AP.

All six model rankings are evaluated on exactly the same later 5,000 IDs using
the existing AP/Precision@100/Recall@100 function. Score ties use ascending
TransactionID, independently of labels. No binary risk threshold is selected.

## Observed same-row results

| Model | Evaluation AP | Delta AP vs CatBoost | Precision@100 | Recall@100 |
|---|---:|---:|---:|---:|
| CatBoost | **0.182366** | 0 | 0.22 | 0.138365 |
| Autoencoder | 0.100439 | -0.081928 | 0.17 | 0.106918 |
| Isolation Forest | 0.052136 | -0.130230 | 0.05 | 0.031447 |
| CatBoost + autoencoder | 0.181334 | -0.001032 | **0.29** | **0.182390** |
| CatBoost + Isolation Forest | 0.157665 | -0.024701 | 0.19 | 0.119497 |
| CatBoost + both anomalies | 0.170859 | -0.011507 | 0.27 | 0.169811 |

**None of these stackers improved AP over CatBoost** on this development
interval. The CatBoost+autoencoder difference is small; it is not evidence of a
general superiority/inferiority result without further temporal comparisons.

At a budget of 100 alerts, CatBoost+autoencoder identified **29 of 159 frauds**
versus CatBoost's **22 of 159**. The full stack identified 27. These different
top-100 outcomes show why AP and alert-budget metrics should be considered
together. No operational policy or deployment choice is made from this reused
development interval.

Full run time was **8.25 seconds**, including data loading, frozen anomaly audit,
training, prediction and persistence. CatBoost fit took 2.191 seconds and scoring
all 10,000 IDs took 0.0074 seconds. Meta fits took 0.042, 0.008 and 0.009 seconds;
each scored the 5,000 evaluation rows in about 0.0018 seconds. Session 17 model
training was not rerun; its training/inference time cells are blank, not zero.
All six comparisons completed without failures or skipped experiments.

## Complementarity checks

Pearson and Spearman base-score correlations are saved separately for
meta-training and evaluation. Evaluation values are:

| Score pair | Pearson | Spearman |
|---|---:|---:|
| CatBoost / autoencoder | 0.290368 | 0.121221 |
| CatBoost / Isolation Forest | 0.262671 | 0.071531 |
| Autoencoder / Isolation Forest | 0.466913 | 0.509606 |

The scores are not identical, but those correlations alone do not establish
useful incremental fraud information. The actual stackers did not improve AP.
Coefficients and meta-training scaler statistics are saved for auditing;
coefficient size is not a causal or per-transaction fraud explanation.

## Persistence and verification

Artifacts are under ignored `outputs/fusion/session18/`:

- `catboost.cbm` and three named stacker `.joblib` pipelines, including their
  fitted scalers. The fitted models are persisted for reproducibility.
- `base_scores.csv`: the 10,000 aligned base-score rows, labels, timestamps and
  explicit meta_train/evaluation roles. Do not treat the meta-training half's
  downstream stacker predictions as held-out results.
- `base_training_targets.csv`: CatBoost fit IDs, labels and timestamps.
- `evaluation_predictions.csv`: all six scores on the same 5,000 evaluation rows.
- `model_comparison.csv`, `stacker_coefficients.csv`, correlation CSVs.
- `config.json`: actual hyperparameters, partitions, input artifact hashes and
  development scope; `frozen_anomaly_audit.json`; `run_summary.json`.

All **59 unit tests passed**, including six new tests in `tests/test_fusion.py`:
shuffled score alignment, incorrect/missing/duplicate IDs, label disagreement,
nonfinite scores, incorrect model names, timestamp-boundary handling, invalid
class splits, meta-only scaler fitting and prediction shape.

A separate reload audit reproduced CatBoost scores for all 10,000 targets and
all three stackers' scores for the 5,000 evaluation targets. It verified the
scalers' means/row counts against only meta-training scores, the strict time
boundary, saved ID/label alignment, and all six reported metric rows. Its result
is saved as `artifact_verification.json`.

## Limits and next steps

This is a **development** chronological holdout-stacking experiment. The second
half was already included in Session 17's aggregate validation reports; earlier
sessions also reused validation. It is not an untouched final assessment.
The reserved test split is never loaded or modified by this script.

No walk-forward OOF stacking, confidence intervals, repeated seeds, calibration,
cost optimization, or prospective generalization claim is included. GRU,
Transformer and GNN scores are not joined into this experiment. They require
consistent target IDs and documented frozen fitting before future integration.
Next-stage fusion selection needs a designated tuning interval and a separate
chronological evaluation interval before any final test evaluation.

## Three code decisions to understand

1. `merge_anomaly_scores` prevents accidental row mismatches and label disagreement.
   Different CSV row orders are safe because identity is explicit.
2. `holdout_partition` preserves complete timestamp groups. Base training, meta
   training and evaluation are temporally distinct; masks alone would not enforce
   that separation.
3. `fit_stacker` accepts only the meta-training table, with StandardScaler inside
   its fitted pipeline. Fitting its scaler on all 10,000 scores would use future
   evaluation information even if the logistic labels were masked correctly.
