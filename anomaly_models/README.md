# Session 17: unsupervised anomaly detection

`train_anomaly_models.py` trains Isolation Forest and a reconstruction
autoencoder on real IEEE-CIS transaction/temporal graph features, then evaluates
their rankings and a seeded uniform-random ranking on identical validation IDs.

Run from the repository root in PowerShell using the existing PyTorch environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe anomaly_models/train_anomaly_models.py
```

No new dependencies are needed beyond `requirements-behavioral.txt` (or the
existing graph environment that includes it). Prerequisites are the established
train/validation splits and Session 13 graph-feature CSVs. If those graph files
need regeneration, run `graph_models/temporal_graph_features.py`; do not regenerate
the original chronological splits to tune results.

The full explicit experiment command is:

```powershell
.\.venv\Scripts\python.exe anomaly_models/train_anomaly_models.py --train-targets 50000 --val-targets 10000 --epochs 10 --batch-size 256 --hidden-size 16 --latent-size 3 --learning-rate 0.001 --trees 200 --seed 42 --threads 2 --output outputs/anomaly/session17
```

## Inputs and preprocessing

Five inputs are used, in this order:

1. TransactionAmt.
2. graph_card_prior_degree.
3. graph_product_prior_degree.
4. graph_card_product_prior_count.
5. graph_card_distinct_products.

The session notes call the fourth feature `graph_pair_prior_count`; the script
uses its existing saved repository name, `graph_card_product_prior_count`.
No feature CSVs are renamed or duplicated to resolve this naming difference.

Original transaction data and graph features are joined by **TransactionID**,
with unique-ID and complete-match checks, then sorted by timestamp and ID.
Matching graph timestamps are checked too. The raw isFraud label is retained for
evaluation only; graph-file labels are not loaded. The loader permits only train
and validation split names.

The latest **50,000 training rows** and earliest **10,000 validation rows** are
selected deterministically without labels. Their historical graph inputs were
already calculated over full earlier history in Session 13, with strictly
earlier timestamps and online updates from earlier validation inputs. Selection
does not recompute histories using only these 50,000 targets. card1 remains an
anonymized proxy, not a verified customer identifier.

Preprocessing follows the exercise: replace NaN and positive/negative infinities
with zero, clamp negative values to zero, apply `log1p`, then standardize.
`StandardScaler` fits on the selected training rows only and stays frozen for
validation. Missing values, observed zero and negative values are deliberately
not distinguished in this compact baseline. These choices can discard useful
information; there are no missing indicators or categorical embeddings here.

## Models and evaluation protocol

- Isolation Forest: 200 trees, max_samples 256, contamination `auto`, seed 42,
  two jobs. Fits `X_train` only, without labels. Higher anomaly score is
  `-forest.score_samples(X_val)`; binary outlier thresholds are not evaluated.
- Autoencoder: `5 -> 16 -> 3 -> 16 -> 5`, ReLU in each hidden 16-unit layer,
  linear latent/output layers, Adam with learning rate 0.001, MSE loss,
  batch size 256, ten fixed epochs, seed 42 and two CPU threads. Its fitting
  function accepts training inputs only, with no labels or validation inputs.
  Each validation score is the mean squared reconstruction error over five
  standardized features.
- Random reference: `numpy.random.default_rng(42).random(10000)`. These scores
  depend only on the seed and validation row count, not labels.

All selected training observations are included, including **2,265** with fraud
labels. Labels do not select legitimate-only training examples, set contamination,
fit models, choose checkpoints or determine transformations. This is an
unsupervised experiment, not normal-only novelty detection. Validation labels
are used only after fitting to evaluate the final fixed model endpoints.

Metrics reuse the existing project evaluation function: sklearn Average
Precision, Precision@100 and Recall@100. Both top-K metrics use descending score
and ascending TransactionID for ties, without consulting labels for ranking.
Anomaly scores are not calibrated fraud probabilities.

## Actual results

| Data | Rows | Frauds | Prevalence | TransactionDT range |
|---|---:|---:|---:|---|
| Training | 50,000 | 2,265 | 4.53% | 9,031,128–10,437,996 |
| Validation | 10,000 | 375 | 3.75% | 10,438,003–10,712,169 |

| Model | Validation AP | Precision@100 | Recall@100 | Training seconds | Inference seconds |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 0.065217 | 0.03 | 0.008000 | 0.593 | 0.127 |
| Autoencoder | **0.124228** | **0.26** | **0.069333** | 5.244 | 0.057 |
| Seeded random reference | 0.038077 | 0.04 | 0.010667 | 0 | <0.001 |

The random-ranking expected AP is approximately prevalence (0.0375); the shown
random score is one realized ranking, not that theoretical expectation.
The autoencoder's training MSE decreased from **0.717579 to 0.036827**. The full
run completed in **11.71 seconds**, including loading, joining, preprocessing
and persistence. Model times exclude common preprocessing. All experiments
completed successfully, without skipped models. Reload verification was outside
the measured training run.

Both anomaly scores achieved higher AP than the random reference on this sample.
Isolation Forest's Precision@100 was lower than the random realization, so AP
alone does not establish good performance at an operational alert budget.
The autoencoder ranked 26 of 375 frauds in its top 100 scores.

These results do not establish incremental value beyond CatBoost. Previous
supervised/sequence/GNN experiments use different target populations and inputs;
their AP values are not matched comparisons here. One seed, one short validation
interval, prior validation reuse and a small feature set limit conclusions.
Future fusion experiments need aligned IDs and a separate chronological fitting
and evaluation protocol for combination weights or a stacking model.

## Saved artifacts and verification

Artifacts are under ignored `outputs/anomaly/session17/`:

- `isolation_forest.joblib`: fitted Isolation Forest, also persisted for
  reproducibility rather than leaving only its scores.
- `autoencoder.pt`: final state dictionary and construction dimensions.
- `scaler.joblib` and `preprocessing.json`: fitted scaler, feature order,
  mean/variance/scale, fit row count and missing-value policy.
- `config.json`: actual hyperparameters, seeds, runtime versions and protocol.
- `train_targets.csv`, `validation_targets.csv`: selected IDs, times and labels.
- `isolation_forest_predictions.csv`, `autoencoder_predictions.csv`,
  `random_predictions.csv`: TransactionID, actual label, model and anomaly score.
- `autoencoder_training.json`, `dataset_statistics.json`,
  `model_comparison.csv`, `run_summary.json`.

All **53 unit tests passed**, including six new tests in
`tests/test_anomaly_models.py`. They cover shuffled-ID joins, missing/duplicate
matches, stale timestamps, missing/infinite/negative inputs, label-independent
feature extraction, frozen scaler statistics, test-split rejection, batched
reconstruction shape, the hand-calculated MSE of **0.625**, and Isolation Forest
score orientation.

A separate saved-artifact audit reloaded the forest, autoencoder and scaler,
verified the scaler's mean against only the 50,000 training rows, and reproduced
all three score vectors on all 10,000 validation IDs. Saved labels, AP and
top-100 metrics also matched. It wrote `artifact_verification.json`.

The reserved test split was never opened, and no original datasets were modified.

## Code to understand

1. `align_graph_features`: merges with `validate="one_to_one"`; row order is
   insufficient to guarantee that a graph statistic belongs to a transaction.
2. `raw_features` and scaler fitting: select only the five named inputs, apply
   deterministic transformations, learn scaling on training, then reuse it.
3. `fit_autoencoder`: the target is the input batch itself. `reconstruction_error`
   reduces across features to obtain one anomaly score per transaction; it never
   compares the reconstruction with isFraud.
