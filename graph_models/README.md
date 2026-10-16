# Session 12: graph fundamentals with NetworkX

Both coding exercises are implemented and executed successfully with the local
NetworkX 3.6.1 installation. Its version is pinned in the main requirements and
lock files; the behavioral requirements reuse that lock.

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe graph_models/graph_basics.py
.\.venv\Scripts\python.exe graph_models/build_transaction_graph.py
```

The real graph uses the earliest 2,000 training transactions, sorted by
`(TransactionDT, TransactionID)`. To choose another sample size:

```powershell
.\.venv\Scripts\python.exe graph_models/build_transaction_graph.py --limit 1000
```

## Toy account-device graph

`graph_basics.py` creates four synthetic accounts and two synthetic devices in
an undirected graph, with typed node attributes and account-device edges.

Observed outputs match the session checkpoints:

- Six nodes, four edges.
- Device D1 degree: 3; device D2 degree: 1.
- Two connected components: `{A, B, C, D1}` and `{E, D2}`.
- Shortest path: `A -> D1 -> C`.
- D1 degree centrality: `3 / (6 - 1) = 0.6`.

`nx.connected_components` finds groups connected by paths; `nx.shortest_path`
finds the fewest-edge path between A and C. Here it identifies a shared device.
Neither connectivity nor centrality establishes fraud or estimates its probability.

## Real transaction-card1 graph

`build_transaction_graph.py` reads only TransactionID, TransactionDT,
TransactionAmt and card1 from `data/processed/train.csv`. It loads no fraud labels,
identity table, validation data or test data. It writes no dataset or model artifacts.

Transactions have tuple identifiers such as `("transaction", 1001)`, node type
`transaction`, bipartite partition 0, transaction time and amount. Anonymized
card1 groups have identifiers such as `("card1_proxy", "1001")`, node type
`card1_proxy` and partition 1. Tuples prevent collisions between different entity
types even when their recorded IDs match. Edges carry relation `has_card1_proxy`.

A transaction connects to its recorded card1 value when it is present and finite.
Transactions without a usable value remain isolated nodes; they are not combined
under a shared missing group. Nonfinite amounts are represented as missing node
attributes. Duplicate/missing transaction IDs and nonfinite times are rejected.

Observed results with the default 2,000-row sample:

| Quantity | Result |
|---|---:|
| TransactionDT range | 86,400–141,447 |
| Total nodes | 2,695 |
| Transaction nodes | 2,000 |
| card1 proxy nodes | 695 |
| Edges | 2,000 |
| Bipartite | True |
| Transactions missing a usable card1 | 0 |
| Connected components | 695 |
| Largest component size | 243 |

The highest-degree proxies are 7919 (242), 9500 (51), 15885 (44), 17188 (30)
and 2803 (27). A proxy's degree counts its associated transactions **in this
selected sample**, not unique people or devices. The graph contains disjoint
stars: a transaction has at most one proxy neighbor. It does not reveal links
between different proxies or establish fraud rings.

card1 is an anonymized card-related attribute, not a verified individual
customer identifier. This graph is for descriptive exploration: its completed
sample includes relationships that occur after earlier sample transactions.
Its final degrees/components must not be used directly as historical fraud
features. The Session 13 implementation below uses strictly earlier-time history
and excludes simultaneous events from each other's historical features.

## Session 13: temporal heterogeneous graph features

Implemented in `temporal_graph_features.py`. Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe graph_models/temporal_graph_features.py
```

The complete training and validation periods are processed chronologically. All
training history remains available for validation; earlier validation **inputs**
also update history for subsequent validation transactions. Each timestamp has
two stages: read the existing counters for every event, then update counters and
sets for every event. This implements the historical graph `G_<t` without
materializing a new NetworkX snapshot at every event. The function streams sorted
timestamp batches using `itertools.groupby`; input order is restored in its
returned feature frame.

Feature definitions:

| Feature | Meaning before the current timestamp |
|---|---|
| graph_card_prior_degree | Count of transactions with the same observed card1 proxy |
| graph_product_prior_degree | Count of transactions with the same observed ProductCD |
| graph_card_distinct_products | Number of distinct observed products associated with this card1 |
| graph_card_product_prior_count | Count of transactions with both this card1 and ProductCD |
| graph_new_product_for_card | 1 when this product has never appeared for this card1; otherwise 0 |
| graph_card4_prior_degree | Count of transactions with the same observed card4 category |
| graph_card_missing | 1 when card1 is unavailable/nonfinite |
| graph_product_missing | 1 when ProductCD is unavailable/nonfinite |
| graph_card4_missing | 1 when card4 is unavailable/nonfinite |

An observed but previously unseen key has degree zero. A first observed
card-product pair has novelty one. Missing/nonfinite keys do not create shared
missing-value nodes: dependent features stay NaN, accompanied by flags. A known
card's degree still counts an event whose product is missing, but missing
products never enter distinct-product sets or pair counts. Other available
attributes continue to update their independent counts.

`TransactionAmt` and `isFraud` are not read by the feature calculation function.
The CLI retains the original seven selected columns and evaluation labels beside
the features. For downstream modeling, explicitly select input columns:

```python
from graph_models.temporal_graph_features import FEATURES

# Join to another input table by TransactionID, with validate="one_to_one".
X = feature_table[FEATURES]  # excludes isFraud, TransactionID and other metadata
y = feature_table["isFraud"]
```

No model is trained in Session 13. Downstream preprocessing must handle the
remaining NaNs with training-fitted transformations or a suitable model's native
missing-value support. These raw counters need no fitted preprocessing statistics.

The script loads only `train.csv` and `validation.csv`. It rejects invalid times,
duplicate/missing IDs, nonbinary evaluation labels, and overlapping temporal
split boundaries. Original data files and the reserved test split are untouched.
Outputs are stored in the existing ignored directory:

```text
outputs/graph/train_graph_features.csv
outputs/graph/validation_graph_features.csv
outputs/graph/temporal_graph_summary.json
```

### Observed Session 13 results

The script completed successfully in **7.65 seconds** on the local machine:

| Quantity | Training | Validation |
|---|---:|---:|
| Feature rows / unique TransactionIDs | 413,378 | 88,581 |
| New card-product observations | 19,072 | 1,132 |
| Unavailable card4 degree values | 829 | 4 |

All **33 unit tests passed**, including eight new tests in
`tests/test_temporal_graph.py`. They cover the hand calculation, timestamp ties,
separate categories/groups, missing/nonfinite attributes, future/label/amount
invariance, original-row alignment, online history and invalid IDs/times.

The exact session example produces:

| ID | Prior card degree | Prior product degree | Prior distinct products | Prior pair count | Product new for card? |
|---|---:|---:|---:|---:|---:|
| T1 (100, A, W) | 0 | 0 | 0 | 0 | 1 |
| T2 (200, A, W) | 1 | 1 | 1 | 1 | 0 |
| T3 (200, A, C) | 1 | 0 | 1 | 0 | 1 |
| T4 (300, A, C) | 3 | 1 | 2 | 1 | 0 |

A separate real-data audit checked all **501,959 saved rows** against independent
Pandas timestamp-batch cumulative counts. Distinct-product counts were checked
using product first-observation times and strict-time binary searches. Novelty,
missing flags and saved IDs/labels/current attributes also matched. No infinite
features were present. That audit wrote
`outputs/graph/temporal_graph_verification.json`; it did not open the test split.

These are graph-derived features, not a GNN or evidence of improved fraud
detection. card1 remains an anonymized proxy; ProductCD and card4 represent
recorded categories, not verified merchants or device identifiers. Common
categories can form large, weakly informative neighborhoods. Future model
comparisons still require matching target IDs and temporal graph sampling.

## Session 14: first GCN with PyTorch Geometric

Implemented in `gcn_basics.py`, using exactly the session's five-node synthetic
chain, two numerical features per node and labels `[0, 0, 1, 1, 0]`. Only nodes
0, 1 and 2 contribute labels to the training loss.

PyTorch Geometric **2.8.1** was installed and successfully imported with the
existing **PyTorch 2.14.1+cpu / Python 3.13.15** environment. Installation follows
the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html).
The example ran with the basic package installation, without additional compiled
PyG extensions. Actual installed versions are recorded in `requirements-graph.txt`,
which reuses the existing behavioral and base dependency pins.

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-graph.txt
.\.venv\Scripts\python.exe -c "import torch_geometric; print(torch_geometric.__version__)"
.\.venv\Scripts\python.exe graph_models/gcn_basics.py
```

The direct PyG installation was performed successfully; a fresh installation of
the entire combined environment was not tested. `pip check` reported no broken
requirements, and every new graph dependency pin matched the installed version.

The architecture is `GCNConv(2,8) -> ReLU -> GCNConv(8,8) -> Linear(8,1)`.
`encode()` exposes the eight-dimensional node embeddings. `GCNConv` supplies
self-loops and degree-normalized aggregation. Both directions of each chain edge
are provided in `edge_index`. `squeeze(-1)` produces one logit per node.

Training uses Adam, learning rate 0.01, unweighted `BCEWithLogitsLoss`, seed 42,
200 epochs and two CPU threads. The mask is applied to **both** logits and labels;
sigmoid is applied only when producing the final probabilities.

Observed output:

```text
Node feature shape: (5, 2)
Edge-index shape: (2, 8)
Label shape: (5,)
Training nodes: [0, 1, 2]
Epoch 1, Training loss: 0.7082
Epoch 50, Training loss: 0.3671
Epoch 100, Training loss: 0.0209
Epoch 150, Training loss: 0.0055
Epoch 200, Training loss: 0.0027
Node embedding shape: (5, 8)
Predicted probabilities: tensor([7.9291e-06, 4.2695e-03, 9.9615e-01, 9.9917e-01, 9.7388e-01])
Final prediction shape: (5,)
```

Additional checks confirmed a valid undirected graph and **identical loss and
parameter gradients when only unmasked labels were flipped**. The environment
emits an upstream `torch.jit.script` deprecation FutureWarning on PyG import;
import, training and inference all complete successfully.

This is a transductive learning demonstration: every node and edge remains
available for message passing, including nodes 3 and 4. Training masks do not
enforce temporal safety. Node 4 receives a high score despite its synthetic
label being zero, illustrating why fitting three training nodes does not establish
generalization. No fraud AP or real-world risk estimate is reported.

No IEEE-CIS files or reserved test data are read or modified. Real temporal GNN
training remains a later task requiring time-constrained graph neighborhoods.

## Session 15: GraphSAGE, GAT and real temporal neighborhoods

Both requested scripts are implemented. Run from the repository root:

```powershell
.\.venv\Scripts\python.exe graph_models/sage_gat_basics.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe graph_models/train_temporal_gnn.py
```

The existing `requirements-graph.txt` environment already provides both layers;
no additional packages or sampling extensions were required.

### Layer exercise

`sage_gat_basics.py` reuses Session 14's exact synthetic node features and chain
edges. It prints input `(5, 2)`, GraphSAGE `(5, 8)` and GAT `(5, 16)`. These are
untrained layer outputs. GAT concatenates two eight-dimensional attention heads.
See the official [SAGEConv](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.SAGEConv.html)
and [GATConv](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.GATConv.html)
references for the layer definitions and head-concatenation behavior.

### Real-data graph construction

`train_temporal_gnn.py` reuses `deep_learning.behavioral_data.prepare_data` rather
than maintaining a second history-selection implementation. Timestamp-batched
window construction excludes same-time peers, providing the same strict-time
behavior as a sorted-history `bisect_left` query. All **413,378 training events**
remain available before the most recent **6,000 training targets** are selected.
Validation targets are the earliest **1,500 validation rows**. Both models use
identical target IDs, labels, features and neighborhoods.

The reused sequence pipeline also fits imputation/scaling on all training events
only. The graph adapter uses just its standardized log-amount column; the other
sequence features do not enter these models. Each graph's node features are:

1. `log1p(TransactionAmt)`, median-imputed and standardized using training-only
   statistics. Missing, nonfinite and negative amounts are treated as unavailable.
2. `log1p((target TransactionDT - event TransactionDT) / 86400)`, the event's age
   in relative dataset days. This fixed transform needs no fitted statistics;
   the current transaction's age is zero.

Each independent graph places the **current transaction at node 0** and up to
nineteen strictly earlier same-card1 events at nodes 1 onward. All explicit edges
point **from historical nodes to node 0**, never the reverse. The builder checks
strictly earlier times, same nonmissing proxy, finite features and target ID/label
alignment. Missing groups produce a valid singleton graph with no explicit edges.
The reused pipeline also rejects invalid timestamps, overlapping split IDs,
duplicate IDs, invalid labels and nonseparated train/validation boundaries.

Neighborhood selection keeps the most recent eligible events, deterministically;
this is recency truncation, not uniform random neighbor sampling. Older events
remain available as context candidates. Earlier training events and earlier
validation **inputs** can inform later targets. No labels enter node features.
Earlier timestamp ties are ordered by ID only for deterministic truncation;
this is not evidence of actual arrival order. The test split is never opened.

PyG combines these per-target graphs as disconnected components. In
`TemporalGNN.forward`, `batch.ptr[:-1]` selects the first (current) node of each
component for classification. Reused historical events are copied into separate
graphs; message passing cannot cross from one target graph to another.

### Architectures and training

- GraphSAGE: one `SAGEConv(2, 32, aggr="mean")`, ReLU, `Linear(32, 1)`. The layer
  combines the target's root transformation with the historical-neighbor mean.
- GAT: one `GATConv(2, 16, heads=2, concat=True)`, ReLU, `Linear(32, 1)`. Self-loops
  are enabled so the target can attend to its own available input as well as its
  historical neighbors. Dropout is zero.

Both use Adam, learning rate 0.001, batch size 64, three epochs, seed 42, two CPU
threads and unweighted BCEWithLogitsLoss. Sigmoid is used for inference only.
Validation AP is logged every epoch, but the reported checkpoint is the **fixed
final epoch**, with no early stopping or best-epoch selection. The models each
have one aggregation layer; they do not implement a multi-hop heterogeneous GNN.

All defaults can be reproduced explicitly:

```powershell
.\.venv\Scripts\python.exe graph_models/train_temporal_gnn.py --train-targets 6000 --val-targets 1500 --max-events 20 --batch-size 64 --epochs 3 --hidden-size 32 --learning-rate 0.001 --seed 42 --threads 2 --output outputs/graph/session15
```

### Observed results

| Dataset statistic | Training | Validation |
|---|---:|---:|
| Selected targets | 6,000 | 1,500 |
| Fraud labels | 257 | 72 |
| Fraud prevalence | 4.2833% | 4.8% |
| Mean nodes per graph | 18.9235 | 18.9393 |
| Singleton graphs | 48 | 11 |
| Maximum nodes | 20 | 20 |

Both models trained and produced validation AP successfully:

| Model | Epoch 1 AP | Epoch 2 AP | Final epoch AP | Precision@100 | Recall@100 | Training seconds | Inference seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| GraphSAGE | 0.047776 | 0.063979 | **0.056522** | 0.08 | 0.111111 | 1.579 | 0.086 |
| GAT | 0.040476 | 0.048706 | **0.051072** | 0.04 | 0.055556 | 2.097 | 0.109 |

GraphSAGE training loss decreased from 0.46658 to 0.21428; GAT decreased from
0.52594 to 0.18613. Total run time including data preparation and persistence was
**14.74 seconds**. Model training times include per-epoch validation; inference
times measure a separate final prediction pass. Both experiments completed,
with no failures or skipped models.

AP is sklearn Average Precision. Top-100 metrics reuse the existing label-free
ranking function: descending score, then ascending TransactionID for score ties.
With validation prevalence 0.048, these APs are modest initial results. The
GraphSAGE score at epoch 2 was higher, but it is not substituted for the declared
three-epoch endpoint. A single small run does not establish one architecture's
superiority or useful out-of-time generalization.

The Session 11 CatBoost and sequence-model scores use different target sets.
They cannot be treated as matched comparisons with this experiment. This session
does not establish whether graph aggregation improves on those models; that
requires matched-row baselines, richer features and further temporal evaluation.
card1 remains an anonymized proxy, not a verified person or physical card ID.
These graphs contain no verified account/device/merchant relationships, and
scores are not calibrated fraud probabilities.

### Verification and persistence

All **39 unit tests passed**, including six new tests in
`tests/test_temporal_gnn.py`. These cover current-node position, directionality,
same-time exclusion, retained earlier training history, missing groups, finite
inputs, batched target selection, model shapes/backpropagation, label-free input
features, frozen training scaling and rejection of invalid history.

Under ignored `outputs/graph/session15/`, the run saves model state dictionaries,
exact configuration, preprocessing statistics, target IDs/labels, neighborhood
indices and lengths, dataset statistics, per-epoch logs, per-model predictions,
`model_comparison.csv` and `run_summary.json`. A separate reload check rebuilt
both target selections, compared all saved neighborhood indices, and verified
that both checkpoints reproduced all 1,500 validation scores, IDs, labels and
AP/top-100 metrics. Its outcome is saved in `artifact_verification.json`.

The installed libraries emit the previously observed upstream PyTorch/PyG
deprecation warnings. The layer exercise, tests and real-data training complete
successfully. The new scripts do not modify existing datasets or train on the
reserved test split.

## Session 16: candidate proxy-linked cluster discovery

`discover_candidate_clusters.py` implements the requested retrospective
connected-component exercise. Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe graph_models/discover_candidate_clusters.py
```

Explicit defaults, configurable without editing the script:

```powershell
.\.venv\Scripts\python.exe graph_models/discover_candidate_clusters.py --limit 20000 --min-cards 2 --max-cards 10 --output outputs/graph/session16
```

The script reads only TransactionID, TransactionDT, card1, addr1 and isFraud from
the training split. It takes the first 20,000 rows after sorting by timestamp
and TransactionID. Validation and test files are not opened; input datasets are
not modified. Existing NetworkX/Pandas dependencies suffice.

### Discovery and evidence

`discover_components()` reads **attributes only** and also works when the input
has no label column. It excludes missing/nonfinite proxy values, reusing the
existing observed-key handling. Canonical typed tuple nodes keep card and address
namespaces separate even when their recorded values match.

For each address proxy, `nunique()` counts distinct card proxies in complete
observed pairs across the selected sample. Only addresses with **2–10 distinct
cards, inclusive**, create graph edges. Repeated transactions do not inflate that
distinct-card filter. An undirected edge represents one observed card-address
pair; repeated transactions supply its transaction count, first-seen time and
last-seen time.

NetworkX connected components receive deterministic IDs based on sorted proxy
keys. Components are ranked by transaction count descending, card-proxy count
descending, unique-pair count descending, then component ID ascending. Fraud
labels never determine eligibility, connectivity, IDs, ranking or tie-breaking.
The 2–10 heuristic is fixed for this exercise and was not tuned against labels.

Only transactions whose **own observed card-address pair** survives filtering
receive membership. Other transactions sharing an included card through an
excluded/missing address are not silently included. This keeps counts and edge
evidence consistent; memberships deliberately cover only part of the sample.

`evaluate_components()` subsequently joins isFraud by unique TransactionID and
computes fraud counts, rates and enrichment. The enrichment denominator is the
fraud prevalence among **all 20,000 selected transactions**, including excluded
rows. With a zero-fraud sample, enrichment is unavailable (NaN), not infinite.
Discovery and report saving also handle a filter that produces no components.

### Actual results

The script completed successfully in **2.45 seconds**:

| Statistic | Result |
|---|---:|
| Selected training transactions | 20,000 |
| TransactionDT range | 86,400–517,208 |
| Sample fraud labels / prevalence | 562 / 2.81% |
| Transactions missing a usable pair | 1,676 |
| Observed address proxies with complete pairs | 117 |
| Address proxies below minimum distinct-card count | 41 |
| Address proxies above maximum distinct-card count | 55 |
| Eligible address proxies | 21 |
| Included transactions | 87 (0.435% of sample) |
| Excluded transactions | 19,913 |
| Graph nodes / unique pair edges | 72 / 68 |
| Connected components | 9 |
| Included fraud labels / retrospective prevalence | 7 / 8.046% |

The highest-ranked component by structure is **C0001**:

| Statistic | Result |
|---|---:|
| Transactions | 57 |
| Card proxies | 27 |
| Address proxies | 13 |
| Unique card-address pair edges | 44 |
| First / last observed TransactionDT | 431,993 / 516,929 |
| Retrospective fraud labels | 7 |
| Retrospective fraud rate | 12.281% |
| Enrichment versus full selected sample | 4.370x |

The other eight components contain 10, 6, 3, 3, 2, 2, 2 and 2 transactions,
respectively, with zero recorded fraud labels in this sample. All seven included
frauds occur in C0001. This is a descriptive finding in an already labeled
sample, not proof of coordinated activity or a prospective detection result.

Coverage is narrow: only **7 of the sample's 562 fraud labels** are represented;
the other **555** are outside the filtered graph. Enrichment must be interpreted
alongside this loss of coverage. Filtering common attributes can remove useful
relationships as well as reduce giant components. No thresholds were changed
after inspecting these outcomes.

### Saved reports and verification

All reports are under the existing ignored `outputs/graph/session16/` directory:

- `component_summary.csv`: structural rank, component ID, transaction/card/address
  counts, unique pair count, observation times and retrospective fraud metrics.
- `component_memberships.csv`: only included TransactionIDs, times, original
  proxies, canonical keys, component IDs and labels joined after discovery.
- `edge_evidence.csv`: component, proxy pair, number of supporting transactions,
  first and last observed times. Filter memberships by both keys to recover the
  exact supporting TransactionIDs and labels for any edge.
- `address_proxy_degrees.csv`: distinct-card count and eligibility for every
  observed address proxy with a usable pair.
- `summary.json`: coverage, filter settings, graph size, ranking/evaluation rules,
  sample prevalence, runtime and explicit retrospective-only scope.

All **47 unit tests passed**, including eight new tests in
`tests/test_candidate_clusters.py`. They cover hand-calculated components and
enrichment, label-independent discovery/ranking, distinct-card filtering,
exclusion of a transaction's ineligible own pair, missing/nonfinite keys, typed
node IDs, edge evidence, row-order invariance, empty graphs, zero-fraud baselines
and invalid inputs.

A separate saved-file audit checked the exact 87 eligible IDs and their labels,
all nine components' connectivity and counts, all 68 edges' transaction/time
provenance, and every enrichment value. It passed and wrote
`artifact_verification.json` alongside the reports.

### Interpretation limits

card1 and addr1 are anonymized recorded attributes, not verified people, cards,
households or physical locations. These are candidate proxy-linked clusters,
not confirmed fraud rings. Components and the hub filter use the **completed
sample graph**; they must not become historical prediction features for earlier
events in that sample. A prospective system needs chronological construction,
allowed current-event relationships and separate evaluation data.

The exercise makes no AP, calibrated fraud-probability or model-improvement
claim. Its evidence supports investigation of recorded links and times; it does
not establish coordination, causality or future generalization.
