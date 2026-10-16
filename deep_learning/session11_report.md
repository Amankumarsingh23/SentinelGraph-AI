# Session 11: observed behavioral modeling results

All seven experiments completed successfully on the local CPU. Historical
sequence inputs improved Average Precision over the current-only neural model,
but the matched-row CatBoost benchmark achieved the highest AP. These are
validation development results; the reserved test split was not opened.

## Implementation and verification

New files:

- `deep_learning/behavioral_data.py`: train-only preprocessing, strictly historical
  card1 windows, target selection and right padding.
- `deep_learning/behavioral_models.py`: current-only/history-summary MLP and reuse
  of the existing educational GRU and Transformer classes.
- `deep_learning/run_behavioral.py`: artifact audit, reproducible training,
  validation AP early stopping, matched CatBoost, metrics and saved artifacts.
- `deep_learning/verify_behavioral.py`: reload checkpoints, reproduce every
  validation score, audit saved sequence timing/groups/IDs/labels and metrics.
- `tests/test_behavioral.py`: 13 focused unit tests.
- `requirements-behavioral.txt`: installed Windows CPU dependency pins.
- `deep_learning/README.md` and this report: protocol, feature definitions,
  assumptions, reproduction and observed findings.

The root README was updated to point to this implementation. The Session 8–10
learning scripts and Phase 1 algorithms were reused without changes.
No data, model artifacts or secrets were added to version control. No commit or
remote push was performed. Existing staged changes were left in place.

Verification completed:

- **25/25 unit tests passed**: 12 existing Phase 1 and 13 new behavioral tests.
- Recomputed all five repaired Phase 1 features on **501,959 train/validation
  rows** and matched the saved repaired artifacts, including missing values.
- Original train and validation SHA-256 hashes matched the Phase 1 backup.
- All saved windows passed current-event, strictly-past-time, group-isolation,
  padding, target ID and target label checks.
- All **seven saved models** reproduced all **20,000 validation scores** and the
  reported AP/top-100 metrics after reloading their persisted checkpoints.
- `pip check` reported no broken requirements. The output directory is ignored
  by Git. No test data was needed for any of these checks.

## Data and sequence statistics

| Quantity | Training | Validation |
|---|---:|---:|
| Available split rows | 413,378 | 88,581 |
| Selected targets | 100,000 latest | 20,000 earliest |
| Selected frauds | 3,945 | 701 |
| Selected fraud prevalence | 3.945% | 3.505% |
| Target start TransactionDT | 7,792,935 | 10,438,003 |
| Target end TransactionDT | 10,437,996 | 10,958,412 |
| Mean valid length, cap 20 | 18.67882 | 18.88375 |
| Length-one targets | 952 | 172 |
| Length-twenty targets | 88,333 | 17,990 |

All 413,378 training events were retained for history and preprocessing before
training targets were selected. A length of twenty means nineteen eligible
historical events plus the current transaction. Missing-group and first-event
targets remain valid current-only examples.

Windows use only strictly earlier timestamps in the same anonymized card1
group. Simultaneous transactions cannot see one another. Training history and
earlier validation inputs may inform later validation targets under an online
history assumption. Fraud labels never enter input features. The exact seven
event features and their training-fitted transformations are documented in
[the implementation guide](README.md#exact-numerical-features).

## Actual model results

Each row uses exactly the same 100,000 training target IDs and 20,000 validation
target IDs. Validation prevalence is **0.03505** for every model.

| Model | AP | Precision@100 | Recall@100 | Training seconds | Inference seconds | Best epoch / epochs run |
|---|---:|---:|---:|---:|---:|---:|
| Current-only MLP | 0.054241 | 0.14 | 0.019971 | 5.47 | 0.102 | 2 / 5 |
| History-summary MLP | 0.063701 | 0.13 | 0.018545 | 7.15 | 0.100 | 4 / 7 |
| GRU, 5 events | 0.119656 | 0.32 | 0.045649 | 25.50 | 0.414 | 7 / 10 |
| GRU, 20 events | 0.156226 | 0.43 | 0.061341 | 63.90 | 0.652 | 3 / 6 |
| Transformer, 5 events | 0.109059 | 0.34 | 0.048502 | 19.12 | 0.194 | 2 / 5 |
| Transformer, 20 events | 0.157569 | 0.42 | 0.059914 | 26.90 | 0.333 | 2 / 5 |
| Matched-row CatBoost | **0.194312** | **0.44** | **0.062767** | 11.94 | 0.022 | 200 fixed iterations |

AP and recall are fractions, not percentages. Precision@100 of 0.44 means
44 frauds among the highest-ranked 100 transactions; recall is 44 / 701.
Ties are resolved by ascending TransactionID independently of labels.

Full run time, including Phase 1 audit, data construction and persistence, was
**184.58 seconds**. Neural training times include per-epoch validation; final
inference time is measured separately. All experiments ran; none failed or
were skipped. Reloading/artifact verification was performed after this timed run.

Actual settings: CPU, two threads, seed 42, batch size 128, Adam with learning
rate 0.001, unweighted BCEWithLogitsLoss, gradient norm clipping at 1, a maximum
of ten epochs, validation AP selection and patience three. Both MLPs use one
32-unit ReLU hidden layer. GRUs use hidden size 32. Transformers use dimension
32, four attention heads, one encoder layer, feed-forward size 32, dropout zero,
learned positional embeddings and padding masks. The final valid event is
classified; no within-window causal mask is needed for this target definition.

Matched CatBoost uses 200 iterations, depth 6, learning rate 0.05, Logloss,
seed 42 and two threads. It uses TransactionAmt, TransactionDT, ProductCD,
card4 and card6, matching the existing tabular benchmark's feature specification.

The saved legacy CatBoost result was **AP 0.149212 on all 88,581 validation
transactions**. Its rows differ from this run; it is retained as context and
is not a directly matched comparison. The newly trained CatBoost result above
is evaluated on exactly the selected neural validation IDs.

## Ablations and interpretation

- Adding summary history to the MLP increased AP by **0.009461**.
- Increasing GRU window size from 5 to 20 increased AP by **0.036569**.
- Increasing Transformer window size from 5 to 20 increased AP by **0.048509**.
- GRU-20 exceeded current-only MLP AP by **0.101985**.
- Transformer-20 exceeded current-only MLP AP by **0.103328**.
- Matched CatBoost exceeded Transformer-20 AP by **0.036744**.

Thus historical sequence inputs helped these small neural models on this
validation interval. They did **not** beat the matched current-transaction
CatBoost benchmark. The Transformer/GRU difference is small and should not be
treated as statistically established. Additional categorical inputs and distinct
model capacities mean the CatBoost comparison is row-matched, not feature-matched.

These experiments do not isolate the value of transaction **order** itself.
That requires an order-shuffling control. The history-summary MLP controls for
some historical information, while sequence models can also exploit multiple
past amounts, their distributions and other historical patterns.

## Limitations and leakage considerations

- card1 groups can mix multiple underlying entities. They are not verified
  individual customer histories.
- Earlier simultaneous events are ordered by ID for deterministic truncation;
  this order is not an observed arrival order. Current-time peers are excluded.
- Length-5 inputs retain event-level summary features computed from all available
  earlier history. This is an ablation of explicit event-window size, not of the
  complete historical information budget.
- The online protocol assumes earlier validation inputs are available when later
  targets are scored. A frozen-history batch protocol would differ.
- Validation drives early stopping and model comparison, and was previously used
  in Phase 1. Earlier project EDA also exposed aggregate test labels. This run
  itself never opens the test split and makes no independent final-test claim.
- One seed, one selected interval, compact features and limited architectures
  cannot establish robustness, significance or optimal hyperparameters.
- Unweighted BCE handles the observed class mix without added weighting, but
  the scores have not been calibrated. No financial-risk or production claim
  follows from fitting these models.

## Reproduction and saved evidence

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-behavioral.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -u -m deep_learning.run_behavioral --train-targets 100000 --val-targets 20000 --max-length 20 --batch-size 128 --hidden-size 32 --epochs 10 --patience 3 --learning-rate 0.001 --threads 2 --seed 42 --output outputs/behavioral/session11
.\.venv\Scripts\python.exe -m deep_learning.verify_behavioral --output outputs/behavioral/session11
```

The dependency snapshot records the working local environment; a fresh install
was not tested. Runtimes vary with laptop activity and library/hardware versions.

Evidence is stored under ignored `outputs/behavioral/session11/`: model state
dictionaries, CatBoost binary, preprocessing metadata, exact configurations,
epoch logs, per-model prediction CSVs, target IDs/labels, window indices, scaled
event features, `model_comparison.csv`, `dataset_statistics.json`,
`phase1_audit.json`, `run_summary.json` and `artifact_verification.json`.

## Three important pieces of code to explain

1. `behavioral_data.event_features_and_windows`: read history for the whole
   timestamp batch, then update it. This prevents simultaneous-event leakage.
   Construct history before selecting training targets so subsampling preserves
   the real available context.
2. `behavioral_models.forward`: packed lengths skip GRU padding; Transformer
   masks hide padding and its classifier uses the final **valid** position.
   Learned positional embeddings supply order to self-attention.
3. `run_behavioral.train_neural` and `record`: optimize raw logits with BCE,
   restore the best validation AP checkpoint, then save scores alongside exact
   TransactionIDs. This separates optimization from ranking evaluation and
   prevents accidental row misalignment across models.
