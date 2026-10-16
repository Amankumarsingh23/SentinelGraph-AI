# Session 23: retrospective feature and prediction drift

`check_drift.py` compares the **last 10,000 training transactions** with the
**first 10,000 validation transactions**, sorted by `(TransactionDT, TransactionID)`.
It reads only IDs, timestamps, amount and the four historical graph counters.
It never parses fraud labels, reads reserved test data, retrains, updates graph
history or changes the serving model. This is an offline diagnostic; live API
latency/error instrumentation, delayed-label ingestion and deployment/rollback
automation remain outside this session's implementation.

## Run and verify

Prerequisites are the existing processed train/validation CSVs, Session 13 graph
feature CSVs and the pinned Session 20 `AllFive.cbm`. Dependencies already exist
in the [Session 22 environment](../api/README.md); no packages were added.
Run from the repository root:

```powershell
.\.venv\Scripts\python.exe monitoring/check_drift.py
.\.venv\Scripts\python.exe monitoring/verify_session23.py
.\.venv\Scripts\python.exe -m pytest tests/test_monitoring.py -v
.\.venv\Scripts\python.exe -m pytest tests -q --junitxml=outputs/monitoring/session23/test_results.xml
```

To explicitly reuse the saved reference bins:

```powershell
.\.venv\Scripts\python.exe monitoring/check_drift.py --reference-bins outputs/monitoring/session23/reference_bins.json
.\.venv\Scripts\python.exe monitoring/verify_session23.py
```

The command also supports `--reference-rows`, `--current-rows`, `--bins`,
`--epsilon` and `--output`. Verification accepts the same `--output` directory.
Requested window sizes must be available; shorter inputs fail instead of
silently reducing the sample. The loader rejects unsupported split names before
reading any file, missing/duplicate IDs, unequal ID populations, stale graph
timestamps and absent required columns. Entirely missing numerical columns are
supported and represented explicitly.

## Model and feature contract

Both API requests and monitoring batches now use `api.model.prepare_features`:
exactly five raw numerical columns in the saved order, infinities converted to
NaN, native CatBoost missing-value handling, **no clipping, scaling or log1p**.
API input validation still rejects nonfinite client inputs before preprocessing.
Offline reports count infinite and negative source values separately rather than
silently hiding them; missingness and PSI use a missing/nonfinite category.

The actual internal pair-count name is `graph_card_product_prior_count`.
The API's external `graph_pair_prior_count` alias is not a physical CSV column.
The monitor loads the same hash-pinned binary as the API and checks its feature
names and binary class order. It does not import any training pipeline.

## PSI and fixed bins

`drift_metrics.py` contains pure statistical helpers. Importing it or the monitor
does not load data, load a model, create directories or generate reports.

For each numerical feature and the frozen model's scores, the reference-only
cuts use nominal deciles with NumPy's linear quantile method. Duplicate cuts
are collapsed, so discrete features can have fewer than ten numerical bins.
Unbounded tail bins retain observations outside the reference range. A constant
reference gets cuts at its value and the next representable larger float so
new constants on either side are detectable. An entirely missing reference
uses one numerical bin and the missing/nonfinite category.

There is always an additional missing/nonfinite category. For raw proportions
`p`, smoothing is `(p + epsilon) / (1 + categories * epsilon)`, with default
`epsilon = 1e-6`. All categories remain positive and sum to one. The PSI is
`sum((current - reference) * log(current / reference))` over those normalized
proportions. Both periods use the **same reference cuts**. Histogram intervals
are left closed/right open, with the last right edge included, following
[NumPy histogram semantics](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html).
The quantile method is recorded explicitly; see [NumPy quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html).

`reference_bins.json` retains cuts, smoothing, nominal bin count, reference
window fingerprint, model hash and feature contract version. Infinity bounds
are JSON strings `"-inf"` and `"+inf"`, keeping the file valid standard JSON.
Reusing bins checks all those settings and independently reconstructs cuts from
the designated training reference. Current observations never determine cuts.
A changed reference/model/configuration or corrupt cuts cause a failure.

## Actual results

| Feature/distribution | PSI |
|---|---:|
| TransactionAmt | 0.009799 |
| graph_card_prior_degree | 0.002988 |
| graph_product_prior_degree | 9.550425 |
| graph_card_product_prior_count | 0.004763 |
| graph_card_distinct_products | 0.015816 |
| Uncalibrated model score | 0.305103 |

Each window contains exactly **10,000 rows**. Missingness, infinite-value counts
and negative-value counts are zero for all five observed feature columns in
both windows. Training-reference TransactionDT spans **10,182,162–10,437,996**;
current validation spans **10,438,003–10,712,169**. These are relative dataset
timestamps, not calendar dates or production label-arrival times.

| Score summary | Training reference | Validation current |
|---|---:|---:|
| Mean | 0.039252 | 0.025874 |
| Median | 0.023172 | 0.016232 |
| 95th percentile | 0.128047 | 0.092740 |
| 99th percentile | 0.298011 | 0.194898 |

The product-history counter has the largest feature PSI. Its top reference
decile contains 1,000 reference observations and 8,087 current observations;
seven reference deciles contain no current observations. Accumulated historical
counters grow through time, so a large value can reflect expected history
accumulation as well as population changes. Empty categories also make this
PSI sensitive to the chosen smoothing. Investigation would compare product mix,
counter definitions and feature-pipeline versions before inferring a fault.

The score reference is **in-sample**: these training rows were used to fit
AllFive. Comparing them with validation scores mixes sampling/generalization
effects with temporal changes. Validation is already reused development data.
These statistics do not prove concept drift, changed fraud prevalence or an AP
decline. No outcome metrics or universal PSI alert/retraining rule are applied.

## Artifacts and checks

Generated files are ignored under `outputs/monitoring/session23/`:

- `feature_drift.csv`: five PSI rows, missingness deltas, finite/missing/infinite
  and negative counts, descriptive statistics and actual numerical bin counts.
- `reference_bins.json`: reusable training-derived feature and score cuts.
- `prediction_drift.json`: score PSI, summaries, model fingerprint and scope.
- `bin_proportions.csv`: counts, smoothed proportions and individual PSI terms,
  including the missing/nonfinite category, for all six distributions.
- `window_features.csv`: exact reference/current IDs, times, inputs and scores;
  contains no labels.
- `run_summary.json`: selection, input/window hashes, runtime versions and limits.
- `verification.json`, `repeatability.json`, `test_results.xml`: audit evidence.

All **15 monitoring tests passed**, and the complete suite passed **112 tests**.
Tests cover the hand-calculated PSI example (approximately 0.196166), identical
and shifted distributions, fixed tails, ties, constant/all-missing columns,
missingness shifts, normalized smoothing, invalid inputs, JSON serialization,
ID/time alignment, label-free reads and reserved-test rejection. Integration
checks match monitoring scores with the API and six independently frozen
Session 20 predictions, reuse saved bins, and reject changed bins/report counts.
`pip check` passed; existing dependency deprecation warnings remain.

The separate artifact verifier reproduced **all 20,000 saved scores**, checked
training-only reference cuts, replayed bin counts/proportions and PSI from saved
observations, verified missingness and row counts, and rechecked source-file
hashes. A second execution reusing the saved baseline produced byte-identical
bins, drift reports and saved windows. No service/model update was performed.
