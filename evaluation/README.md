# Session 20: controlled ablations and development reliability

`session20_ablations.py` implements all four requested CatBoost configurations,
matched-row evaluation, paired AP bootstrap and chronological block diagnostics.
Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe evaluation/session20_ablations.py
```

The complete explicit experiment command is:

```powershell
.\.venv\Scripts\python.exe evaluation/session20_ablations.py --train-targets 50000 --iterations 400 --bootstrap-repeats 1000 --blocks 5 --seed 42 --threads 2 --reference outputs/fusion/session18/evaluation_predictions.csv --output outputs/evaluation/session20
```

Dependencies already exist in `requirements-behavioral.txt`. Prerequisites are
the established train/validation splits, Session 13 graph-feature CSVs and
Session 18's reference evaluation file. No test file is loaded, no earlier model
artifact is replaced, and no original split or graph-feature file is rewritten.

## Controlled experiment

All models use the **latest 50,000 training targets**, including 2,265 frauds
(4.53%), spanning TransactionDT **9,031,128–10,437,996**. Graph features were
computed from the full history before these targets were selected, preserving
all earlier training observations and the documented online validation-history
assumption. The full training period has 413,378 rows.

Evaluation uses exactly the **5,000 Session 18 evaluation TransactionIDs**,
with **159 frauds / 3.18% prevalence**, spanning TransactionDT
**10,588,835–10,712,169**. Original validation data is the authoritative source
for labels/times; the saved reference is checked for unique IDs, complete matches
and matching labels/timestamps. Shuffling either file does not change alignment.

The reused Session 17 loader merges graph features by ID and checks original
timestamps. The tutorial's `graph_pair_prior_count` is the existing repository
column **`graph_card_product_prior_count`**, whose name is preserved.

| Configuration | Input columns |
|---|---|
| AmountOnly | TransactionAmt |
| AmountPlusActivity | TransactionAmt, graph_card_prior_degree, graph_product_prior_degree |
| AmountPlusRelationships | TransactionAmt, graph_card_product_prior_count, graph_card_distinct_products |
| AllFive | All five columns above |

For every model: **400 iterations, depth 6, learning rate 0.05, Logloss, seed 42,
two CPU threads**. Infinities are converted to NaN; CatBoost's native missing-value
handling is used consistently for every included column. Inputs are raw numerical
values, with no scaler or log transform. Validation is not supplied as an eval_set,
so it cannot select tree count through early stopping. Only the feature subset
changes between these fits.

This is a deliberately weak **amount-only** current-transaction baseline, not
the richer earlier current-only model with time and categorical attributes.
Session 18's full-feature CatBoost used **200** iterations. Its old predictions
define the evaluation population only and are not reused as a 400-tree ablation
result. No cross-session leaderboard is constructed.

## Actual matched-row results

| Model | AP | Delta AP vs AmountOnly | Precision@100 | Recall@100 | Training seconds |
|---|---:|---:|---:|---:|---:|
| AmountOnly | 0.048227 | 0 | 0.10 | 0.062893 | 3.490 |
| AmountPlusActivity | 0.144062 | +0.095834 | 0.16 | 0.100629 | 3.994 |
| AmountPlusRelationships | 0.146785 | +0.098558 | 0.17 | 0.106918 | 3.515 |
| AllFive | **0.190858** | **+0.142630** | **0.21** | **0.132075** | 3.823 |

Inference on the 5,000 rows took approximately 0.002–0.0045 seconds per model.
The full run took **23.97 seconds**, including loading, fitting, uncertainty
analysis and persistence. All four models completed; none failed or were skipped.
Saved-artifact verification was performed after the timed run.

The aggregate data support a substantial development-sample gain from these
historical features **relative to amount alone**. Activity and relationship
groups both help in their respective variants, and their combination performs
best here. These marginal differences are not independent, additive or causal
attributions to individual features; groups can interact.

## Paired bootstrap

The prespecified contrast is:

```text
Delta AP = AP(AllFive) - AP(AmountOnly)
```

For each seeded draw, the same 5,000 row positions are sampled with replacement
for labels and both frozen score vectors. AP is calculated on the two paired
resamples. Models are not refitted in the bootstrap.

Actual analysis:

- Point AP difference: **+0.142630**.
- 95% paired-row percentile interval: **[+0.105239, +0.191303]**.
- Requested and valid draws: **1,000 / 1,000**; skipped single-class draws: **0**.
- Mean resampled AP difference: **0.145917**; standard deviation **0.022482**.
- All 1,000 observed resampled differences were positive.

The interval excludes zero under this resampling scheme. The mean bootstrap
delta is not substituted for the actual point estimate. The fraction of positive
draws is descriptive and is **not** a hypothesis-test p-value.

Transaction-row resampling assumes rows can be resampled independently. Real
transactions can share proxy groups or temporal bursts, so this interval may
understate uncertainty. It conditions on these fitted models and evaluation
rows; it does not include training/seed variation, temporal-distribution changes,
repeated development-set use, or uncertainty against a richer baseline.
Temporal/grouped bootstraps and repeated chronological evaluations remain later
research work.

## Chronological stability

Five consecutive 1,000-row evaluation blocks were used. Boundaries advance past
equal timestamp groups; on this dataset the nominal cuts already separated times
strictly. All four models are reported in `chronological_blocks.csv`; the main
contrast is:

| Block | Frauds / rows | Prevalence | AmountOnly AP | AllFive AP | Delta AP |
|---|---|---:|---:|---:|---:|
| 1 | 16 / 1,000 | 1.6% | 0.043949 | 0.036584 | **-0.007365** |
| 2 | 27 / 1,000 | 2.7% | 0.088075 | 0.251204 | +0.163129 |
| 3 | 23 / 1,000 | 2.3% | 0.031636 | 0.194665 | +0.163029 |
| 4 | 32 / 1,000 | 3.2% | 0.078768 | 0.217561 | +0.138794 |
| 5 | 61 / 1,000 | 6.1% | 0.071918 | 0.288124 | +0.216206 |

The full model improves AP in **four of five blocks**, but is worse in the first.
This instability must accompany the stronger aggregate result. Block prevalence
also varies; comparisons within each paired block are more directly controlled
than raw AP differences across different blocks. These are diagnostics within
one reused interval, not five independent tests or proof of concept drift.
If another run has a block without fraud labels, AP/recall are saved as unavailable
rather than a fabricated zero AP. Large timestamp batches can reduce actual
block count; the configuration records that count.

## Artifacts and verification

Outputs are under ignored `outputs/evaluation/session20/`:

- Four fitted `.cbm` models, named by configuration.
- `training_targets.csv`, `evaluation_targets.csv`: exact IDs, times and labels.
- `predictions.csv`: aligned four-model scores beside every evaluation ID.
- `model_comparison.csv`: metrics, feature lists, counts, runtime and AP deltas.
- `bootstrap_draws.csv`, `bootstrap_summary.json`.
- `chronological_blocks.csv`: all four models in each actual block, with time
  boundaries, prevalence, fraud count and per-block AP differences.
- `config.json`: common hyperparameters, actual feature groups, runtime versions,
  reference-file hash, and fingerprints of ordered training/evaluation inputs.
- `run_summary.json`, `artifact_verification.json`.

All **74 unit tests passed**, including seven new tests in
`tests/test_session20.py`. New checks cover shuffled-ID alignment, invalid
references, timestamp-safe block coverage, exact zero deltas for identical
models, reproducible resamples and skipped-draw accounting, a hand-calculated
paired draw, invalid inputs and no-positive-label block handling.

An independent saved-artifact audit:

1. Rebuilt the selected training/evaluation inputs and checked their fingerprints.
2. Reloaded all four models, checked feature order, 400 trees and seed, then
   reproduced all 5,000 scores per model and every overall metric.
3. Recomputed the 95% interval from all saved deltas and checked that every saved
   difference equals the two AP values' difference.
4. Replayed the first ten seeded paired bootstrap draws from source predictions.
5. Recomputed all twenty model/block metric rows.

All checks passed. No model retraining was needed for the audit.

## What the experiment establishes

The tested historical counters improve this controlled CatBoost comparison over
amount alone on these development rows, with a positive paired-row interval and
nonuniform chronological behavior. It does not establish that a GNN is necessary,
that graph relationships are verified real-world identities, or that a full
heterogeneous production GNN has been validated.

We have not completed a fair all-model comparison on a common protocol, a richer
current-only control, final production model selection, or reserved-test
evaluation. No production model/policy is frozen by this session. Features,
preprocessing, architecture, model-selection procedure and decision policy must
be fixed before any one-time final test evaluation.

Three code ideas to explain in an interview: reference TransactionIDs fix the
evaluation population; identical training/hyperparameters isolate feature-group
changes; `paired_ap_bootstrap` resamples the **same** indices for both models to
estimate uncertainty in their difference rather than compare separate intervals.
