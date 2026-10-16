# Session 19: calibration, alert policy, cost and native CatBoost SHAP

Both coding checkpoints are implemented and executed. No base model was
retrained. Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe decision_models/evaluate_decisions.py
.\.venv\Scripts\python.exe decision_models/explain_catboost.py
```

The existing environment suffices; CatBoost's native SHAP support requires no
additional `shap` package. Prerequisites are the original validation split,
Session 13 graph features, Session 17 target IDs, and the saved Session 18
CatBoost model/configuration/training target metadata and evaluation predictions.

## Artifact alignment and frozen model

The tutorial's `meta_training_predictions.csv` and combined anomaly
`validation_predictions.csv` are not the filenames in this implementation.
Session 18's **`base_scores.csv` already contains all 10,000 CatBoost scores**,
including the earlier 5,000 rows. Session 17's `validation_targets.csv` identifies
their exact IDs, labels and times.

`load_frozen_inputs` loads `outputs/fusion/session18/catboost.cbm`, checks the
saved feature order, joins the exact targets to original validation features,
verifies labels/timestamps, and reproduces all existing CatBoost scores. It
checks that the recorded base training period precedes validation. The model's
SHA-256 hash is recorded in both the decision and SHAP reports.

If `base_scores.csv` is unavailable, the same helper regenerates the 10,000
scores from the frozen model and original inputs, **without fitting on validation**.
Missing required model/feature artifacts stop execution. In the actual run the
existing base score file was present and passed the reproduction check.

The exact five raw numerical inputs are TransactionAmt, graph_card_prior_degree,
graph_product_prior_degree, graph_card_product_prior_count and
graph_card_distinct_products. Infinities become NaN as in Session 18, and CatBoost
uses native missing-value handling. These inputs are not log-transformed or
standardized for the saved CatBoost. Reusing the anomaly model's scaler here
would explain a different input representation and is intentionally avoided.

## Three chronological blocks

The existing timestamp-boundary helper is reused twice. Equal-time events stay
together; actual counts can differ from nominal quarters when ties cross cuts.
For this run both original boundaries were already strict:

| Block | Rows | Frauds | Prevalence | TransactionDT range |
|---|---:|---:|---:|---|
| Fit calibration | 2,500 | 128 | 5.12% | 10,438,003–10,512,289 |
| Select alert fraction | 2,500 | 88 | 3.52% | 10,512,348–10,588,784 |
| Evaluate frozen policy | 5,000 | 159 | 3.18% | 10,588,835–10,712,169 |

The last block's IDs/labels/scores are checked against Session 18's saved
evaluation rows. It is the same already-examined development interval, not a
new independent final test. The reserved test files are never opened.

## Sigmoid calibration and observed results

`fit_calibrator` accepts only the first block. It fits a regularized sigmoid:
LogisticRegression on the logit of the frozen CatBoost probability, with C=1,
lbfgs, max_iter=1000, no class weights and seed 42. Probabilities are clipped to
`[1e-12, 1 - 1e-12]` before computing log-odds. Both calibration parameters are
learned from the first block; the other blocks are transform-only.

The fitted slope is **1.336989** and intercept **1.696390**. Its positive slope
preserves ranking here. The actual later evaluation results are:

| Score | AP | Brier score | Precision@100 | Recall@100 |
|---|---:|---:|---:|---:|
| Raw frozen CatBoost | 0.182366 | **0.028538** | 0.22 | 0.138365 |
| Sigmoid-calibrated CatBoost | 0.182366 | 0.029170 | 0.22 | 0.138365 |

**Calibration worsened Brier score** on this development interval; it did not
improve ranking. This result is retained, without choosing a different
calibration method or hyperparameters after viewing evaluation labels.
The calibration/evaluation blocks also have different observed prevalence;
no causal explanation for the deterioration is established by this one run.
Fixed probability-bin counts, mean predictions and observed rates are saved in
`reliability_bins.csv`; sparse bins do not establish reliable calibration.

Only CatBoost is calibrated. Session 18 stackers were fitted using the earlier
5,000 validation labels, so their outputs on those same rows are not suitable
held-out calibration data under this design.

## Alert capacity and cost proxy

The explicit assumptions are:

```text
realized cost proxy = 200 * false positives + sum(missed fraud amounts)
true positives and true negatives cost zero
```

The 200 is an illustrative false-positive cost in **assumed compatible
dataset-amount units**. Dataset amounts are not asserted to be INR or another
currency. This is neither measured business loss nor validated operational cost.
It assumes complete fraud-amount loss when missed and complete prevention when
caught, with no true-positive investigation cost or residual loss.

Candidate alert counts on the 2,500-row policy-development block are
**0, 25, 50, 100, 200, 500, 1,000**, plus **2,500 (alert all)**. Ranking is descending
calibrated probability, descending original CatBoost score, then ascending
TransactionID. Original scores break calibration ties; labels never break ties.

Each candidate's cost is calculated on policy-development labels only. All
minimum-cost ties are recorded; fewer alerts win a cost tie. **K=0** was the
minimum among these candidates: cost **17,527.240**. The next candidate, K=25,
cost **18,055.046**. This is a finite-grid selection result, not a claim that every
possible capacity or transaction-specific decision rule was optimized.

The selected fraction is K / development rows. Evaluation capacity preserves
that fraction with deterministic nearest half-up rounding. Here **0/2,500 ->
0/5,000**. For example, a selected 50/2,500 would become 100/5,000. This preserves
a fraction of rows, not daily review capacity or an equal-duration workload.

Frozen-policy evaluation:

| Policy | Alerts | Frauds caught | False positives | Missed fraud amount | Realized cost proxy |
|---|---:|---:|---:|---:|---:|
| Selected fraction / alert none | 0 | 0 | 0 | 28,762.449 | **28,762.449** |
| Alert all | 5,000 | 159 | 4,841 | 0 | 968,200.000 |

This does not recommend a production no-alert policy. It exposes the outcome of
the chosen cost assumptions, candidate grid and development population.

### Fixed 100-alert comparison from Session 18

This separate descriptive comparison uses a predeclared K=100 for both models;
it does not revise the selected policy or fit another calibrator.

| Model | Frauds caught | False positives | Caught fraud amount | Missed fraud amount | Realized cost proxy |
|---|---:|---:|---:|---:|---:|
| CatBoost | 22 | 78 | 6,362.391 | 22,400.058 | 38,000.058 |
| CatBoost + autoencoder | 29 | 71 | 6,412.627 | 22,349.822 | 36,549.822 |

Fusion lowers this fixed-budget proxy by **1,450.236**, comprising **1,400** less
false-positive cost and **50.236** more aggregate fraud amount caught. The two
top-100 sets need not be nested. Both fixed-budget costs exceed the no-alert
baseline under the same assumptions. Detecting more fraud labels alone does
not establish economic value.

## Native SHAP exercise

`explain_catboost.py` uses the actual frozen Session 18 model and exact original
feature representation. It selects the **five highest raw CatBoost scores** in
the later evaluation block, with ID tie-breaking and no label-based selection.

For each transaction, CatBoost returns one contribution per input feature and
the expected raw-score baseline. The script checks:

```text
baseline + sum(feature SHAP contributions) == raw CatBoost output
sigmoid(reconstructed raw output) == frozen CatBoost predict_proba output
```

Both checks passed. Maximum raw-score reconstruction error was
**4.996e-15**, and probability error **1.110e-15**. The shared SHAP baseline for
these rows is approximately **-3.584334 raw log-odds**.

For example, transaction **3406492** has raw probability **0.649044** and raw
log-odds **0.614838**. Its actual contributions are:

| Input feature | Value | SHAP contribution, log-odds |
|---|---:|---:|
| TransactionAmt | 450 | +2.118680 |
| graph_product_prior_degree | 28,304 | +1.639184 |
| graph_card_prior_degree | 1,144 | +0.225511 |
| graph_card_product_prior_count | 169 | +0.172326 |
| graph_card_distinct_products | 5 | +0.043472 |

These explain the fitted **raw CatBoost output**, not the calibrated probability,
fusion score, alert decision, causal effect or fraudulent intent. Correlated
historical counts can complicate interpretation. Values across this selected
five-row sample are not a global feature-importance study. Proxy identifiers do
not establish verified real-world identities.

## Files, reproducibility and verification

New source files are `decision_models/evaluate_decisions.py`,
`decision_models/explain_catboost.py` and `tests/test_decisions.py`; the root README
links this guide. All runtime outputs are in ignored `outputs/decisions/session19/`:

- `sigmoid_calibrator.joblib`: frozen fitted calibrator.
- `development_scores.csv`: all 10,000 IDs, timestamps, amounts, labels, scores
  and explicit calibration/policy-development/evaluation roles.
- `probability_metrics.csv`, `reliability_bins.csv`.
- `policy_development_costs.csv`: every prespecified candidate and all minima.
- `policy.json`, `evaluation_policy_costs.csv`, `evaluation_decisions.csv`.
- `fixed_top100_model_costs.csv`.
- `config.json`, `run_summary.json`, `artifact_verification.json`.
- `shap/explained_transactions.csv`, `shap/feature_attributions.csv`,
  `shap/verification.json`.

All **67 tests passed**, including eight new tests covering timestamp boundaries,
calibration scope, hand-calculated costs, invalid amounts/labels/masks,
label-independent ranking ties, zero/all candidates, fewer-alert cost ties,
fraction transfer, ID alignment and native SHAP additivity/probability recovery.

A separate artifact audit reloaded the calibrator, reproduced its probabilities,
reselected the policy using only its designated development block, and verified
saved evaluation decisions, costs, AP/Brier/top-100 metrics and saved SHAP sums.
It passed. The decision run took **1.85 seconds**, excluding subsequent artifact
audit. No base models, original datasets or earlier-session artifacts were changed.

Additional options:

```powershell
.\.venv\Scripts\python.exe decision_models/evaluate_decisions.py --review-cost 200 --output outputs/decisions/session19
.\.venv\Scripts\python.exe decision_models/explain_catboost.py --count 5 --output outputs/decisions/session19/shap
```

Alternative cost assumptions are separate experiments, not a reason to select a
favorable result on these same evaluation labels. The existing validation reuse
limits all conclusions; an independent chronological protocol is still needed
before a final test or deployment decision.
