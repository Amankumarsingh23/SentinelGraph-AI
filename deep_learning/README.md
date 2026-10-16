# Session 11: real behavioral modeling

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-behavioral.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -u -m deep_learning.run_behavioral
.\.venv\Scripts\python.exe -m deep_learning.verify_behavioral
```

The dependency file records the installed Windows CPU environment. A fresh
installation was not tested. All runs use CPU, two intra-operation threads, one
inter-operation thread and no DataLoader worker processes. The initial machine
has an Intel i5-12500H, 16 logical processors, 16 GB RAM and about 2.5 GB free RAM.

The complete explicit experiment command is:

```powershell
.\.venv\Scripts\python.exe -u -m deep_learning.run_behavioral --train-targets 100000 --val-targets 20000 --max-length 20 --batch-size 128 --hidden-size 32 --epochs 10 --patience 3 --learning-rate 0.001 --threads 2 --seed 42 --output outputs/behavioral/session11
```

For a separate, smaller development run (still retains all earlier history):

```powershell
.\.venv\Scripts\python.exe -u -m deep_learning.run_behavioral --train-targets 10000 --val-targets 3000 --epochs 3 --output outputs/behavioral/small_run
```

Do not regenerate the established splits. The runner validates train/validation
split hashes when the Phase 1 backup is available, requires repaired feature
artifacts, and recomputes all five Phase 1 features over train plus validation.
The separately run Phase 1 unit suite checks the strict timestamp semantics.
The runner never opens raw full-dataset files, test.csv or test_features.csv.

## Files and responsibilities

- `behavioral_data.py`: preprocessing, timestamp-batched windows, truncation,
  deterministic target selection and padding.
- `behavioral_models.py`: a small MLP and reuse of `FraudGRU` from Session 9 and
  `FraudTransformer` from Session 10. The learning examples stay runnable.
- `run_behavioral.py`: Phase 1 artifact audit, training, validation AP early
  stopping, matched-row CatBoost and artifact persistence.
- `verify_behavioral.py`: verify saved window contracts and reload each checkpoint
  to reproduce validation scores and metrics without training.
- `../tests/test_behavioral.py`: miniature hand-calculated dataset tests,
  preprocessing checks, shape checks and padding invariance checks.
- `session11_report.md`: observed results and interpretation for the recorded run.

## Point-in-time contract

`card1` is an anonymized card-related grouping proxy, not a verified customer ID.
Training target selection takes the latest 100,000 training rows; validation
selection takes the earliest 20,000 validation rows, after sorting on
`(TransactionDT, TransactionID)`. Both choices are deterministic and independent
of labels. All 413,378 training events are retained before selecting targets.

For every event, the window contains up to 19 strictly earlier-time events from
the same group plus that event. All predictions for one timestamp are assembled
before any event at that timestamp is added to history. A later transaction can
use both earlier simultaneous peers, ordered by ID only to make truncation
deterministic. That order is not evidence of actual arrival order. Missing or
nonfinite card1 values have no history and are never pooled as one group.

Validation has online-updated history: earlier training events and earlier
validation inputs are available, including inputs from rows not selected as
training targets. Validation labels never update any feature. Neither earlier
nor current fraud labels are input features. There is no reserved-test access.

The maximum length includes the current event. A target with zero historical
events has length one. Windows are right padded with zeros **after** scaling.
Length-five ablations retain the last five real events of the same length-twenty
windows. All models use exactly the same target IDs and labels.

## Exact numerical features

Each event's features describe information available at that event's own time:

| Feature | Definition |
|---|---|
| log_amount | `log1p(TransactionAmt)` for a finite, nonnegative amount; otherwise missing |
| relative_days | `(TransactionDT - minimum training TransactionDT) / 86400`; relative dataset time, not calendar date |
| amount_missing | 1 if amount is missing, nonfinite or negative; otherwise 0 |
| group_missing | 1 if the proxy is missing/nonfinite; otherwise 0 |
| log_gap_seconds | `log1p(current time - latest strictly earlier group timestamp)`; missing if no history |
| has_prior_history | 1 if at least one strictly earlier event exists for this group |
| log_prior_count | `log1p(number of strictly earlier group events)`; zero for missing/no-history groups |

Median imputation and population mean/standard deviation are fitted on **all
training events only**, never validation. All-missing columns use median zero;
constant columns use scale one. The missingness/history indicators distinguish
imputed values from observed values. Negative amounts are treated as invalid.
Invalid timestamps, duplicate IDs, overlapping split IDs, nonbinary labels and
nonseparated train/validation boundaries are rejected. Invalid model scores fail
evaluation. Empty historical windows are valid; empty target sets are rejected.

## Models and comparisons

The current-only MLP uses just the first four features of the current transaction.
The history-summary MLP additionally sees the current event's gap/history/count
features, but no preceding event vectors. It controls for some historical
information that is already available without a recurrent/attention model.

GRU-5/20 use packed sequences, hidden dimension 32 and the final valid hidden
state. Transformer-5/20 use dimension 32, four heads, one encoder layer, a
32-unit feed-forward sublayer, dropout zero, learned positional embeddings and
key-padding masks. Their final valid event representation feeds a linear head.
A causal mask is unnecessary for this final-event classification: each window
ends at its target. It would be required for simultaneous per-event prediction.

The neural models use Adam (learning rate 0.001), unweighted BCEWithLogitsLoss,
batch size 128, gradient norm clipping at 1, seed 42, at most ten epochs and
patience three. An improvement must exceed 0.000001 AP; the best validation AP
checkpoint is restored. Fixed seeds and deterministic PyTorch operations improve
repeatability on the same software/hardware, without cross-platform guarantees.
No oversampling, positive weighting or calibration is applied. Scores are not
claimed to be calibrated financial risk probabilities.

Matched CatBoost uses the same training and validation target IDs, 200 iterations,
depth 6, learning rate 0.05, Logloss, seed 42 and two threads. Its features reuse
the established benchmark: TransactionAmt, TransactionDT, ProductCD, card4 and
card6. This is a fair **row-matched** comparison, but it has different categorical
inputs and a different capacity/training budget. It does not isolate architecture.
The saved legacy CatBoost AP is reported separately because it used all 88,581
validation rows. It must not be compared directly with selected-row AP.

AP is sklearn Average Precision, not trapezoidal PR-AUC. Precision@100 and
Recall@100 use descending score then ascending TransactionID, reusing the
existing `alert_ranking` function. Labels never determine tie-breaking. On tiny
development selections the K is min(100, selected validation rows).

## Artifacts and timing

All artifacts are under the ignored `outputs/behavioral/session11/` directory:

- `.pt` state dictionaries plus model construction parameters, `.cbm` CatBoost.
- `preprocessing.json`, `config.json`, `phase1_audit.json`, per-model training logs,
  `dataset_statistics.json`, `run_summary.json` and `legacy_benchmark.json`.
- Target ID/label/length CSVs, compact window indices and valid lengths (`.npz`),
  ordered `history_events.csv`, and scaled `event_features.npy` for reconstruction.
- One prediction CSV per model: TransactionID, actual label, model name and score.
- `model_comparison.csv` with AP, top-100 metrics, prevalence, training/inference
  times, row counts, actual hyperparameters and selected epoch.

Neural training runtime includes per-epoch validation and checkpoint copying;
inference runtime measures one separate pass of the restored checkpoint. Data
construction and serialization are excluded from model-specific times but
included in total runtime. CPU times depend on other laptop activity. Unexpected
neural model failures are recorded in `failures.json` and printed; a Phase 1 audit
failure stops training immediately. Use a new output directory for independent
runs; rerunning a directory replaces corresponding artifacts.

## What to understand for interviews

1. **Timestamp batches in `event_features_and_windows`:** make predictions from
   existing state first, then append the complete batch. This makes the strict
   past-only rule hold even when many transactions share a timestamp. Picking
   target rows happens after constructing histories, so subsampling does not erase
   available preceding observations.
2. **Lengths and masks:** GRU packing skips the padded suffix. Transformer key
   masks prevent valid queries from attending to padding; the classifier selects
   index `length - 1`, never the final padded array position. Position embeddings
   give attention the event order.
3. **Validation and alignment:** training-fitted preprocessing is frozen, best
   checkpoints are selected using validation AP, and every prediction is saved
   beside its TransactionID. The same selected rows and deterministic top-K
   ranking make the model comparison auditable.

## Limits on conclusions

Validation is used for early stopping and comparisons and was already used in
Phase 1. These are development results, not an independent final test. Proxy
groups can mix distinct entities; arbitrary ID order within earlier timestamp
batches is a representation limitation. Length-five and length-twenty inputs
still carry full earlier-history counts and gap features for each event.

A sequence gain may come from historical information, capacity or optimization,
not necessarily learning useful **order**. The history-summary control helps but
does not replace order-shuffling ablations. One seed and one validation interval
do not establish significance or generalization across time. No hyperparameter
search, calibrated risk, business-cost conclusion or deployment claim is made.
