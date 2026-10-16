# Historical aspirational research plan

> Preserved text transcription of the original specification, not an implemented
> feature list or current setup guide. Target-state resume bullets and proposed
> platform capabilities below remain historical aspirations. See the root README
> for implemented scope. Original DOCX is preserved unchanged outside this repo.
> Paragraph text is retained; DOCX layout/formatting is not reproduced.

SentinelGraph AI

Adaptive Financial Fraud & Risk Intelligence Platform

Project Type: Flagship Data Science / Machine Learning / Deep Learning / AI ProjectTarget Development Horizon: ~4 MonthsPrimary Profile: Data Science / Machine Learning EngineerSecondary Profiles: AI Engineer, Applied Scientist, ML Platform EngineerCore Theme: Multi-modal fraud intelligence using tabular ML, behavioral sequence modeling, graph neural networks, anomaly detection, explainability, adaptive learning, and AI-assisted investigation.



1. One-Line Project Definition

SentinelGraph AI is an adaptive financial fraud intelligence platform that combines transaction-level machine learning, behavioral sequence models, graph neural networks, anomaly detection, and explainable AI to detect individual fraud, coordinated fraud rings, and previously unseen suspicious behavior in near real time.



2. 30-Second Explanation

Traditional fraud detection often evaluates transactions individually.

However, fraud frequently appears through a combination of:

abnormal transaction characteristics,

unusual changes in a user’s behavior,

relationships between accounts, cards, devices, addresses, and merchants,

and new attack patterns that may not have appeared in labelled training data.

SentinelGraph AI combines four complementary risk engines:

Tabular ML for transaction-level risk,

Deep sequence models for behavioral abnormalities over time,

Graph Neural Networks for relational fraud and fraud-ring discovery,

Anomaly detection for previously unseen patterns.

These signals are fused into a calibrated risk score and passed through an explainable decision layer. An AI investigation assistant then converts model evidence into analyst-friendly case summaries without allowing the LLM itself to make the fraud decision.



3. Why Am I Building This?

I want this project to demonstrate much more than the ability to train a classifier.

The project should answer:

Can I take a difficult real-world machine-learning problem from raw data to modeling, experimentation, deployment, monitoring, explainability, and business decision-making?

Fraud detection is an excellent problem because it naturally includes:

extreme class imbalance,

temporal behavior,

graph relationships,

adversarial behavior,

data drift,

cost-sensitive decisions,

uncertainty,

explainability requirements,

high-throughput inference,

and human review.

This allows one project to bring together most of the important areas I want to learn deeply before placements.



4. Why This Project Is Resume-Worthy

The objective is not to write:

Built a fraud classifier using XGBoost.

The target project should support discussions around:

supervised learning,

deep learning,

transformers,

graph neural networks,

unsupervised learning,

feature engineering,

probability calibration,

temporal leakage,

imbalanced datasets,

model ensembles,

anomaly detection,

model explainability,

active learning,

concept drift,

MLOps,

real-time inference,

streaming data,

decision optimization,

and responsible GenAI integration.

It should be capable of generating a 30–45 minute technical interview discussion from one project.



5. Core Problem Statement

Given a new financial transaction, estimate whether it represents suspicious or fraudulent behavior while minimizing both:

Fraud Loss

allowing fraudulent transactions,

and

Customer Friction

incorrectly blocking legitimate transactions.

The system must additionally identify:

coordinated fraud networks,

accounts linked to known suspicious entities,

sudden behavioral changes,

previously unseen anomalies,

model degradation over time,

and explanations for high-risk decisions.



6. Why Ordinary Classification Is Not Enough

Consider three situations.

Case A — Transaction-level anomaly

A user normally spends:

₹300₹600₹900₹1,200

Suddenly:

₹75,000

The amount itself may contain useful signal.



Case B — Behavioral anomaly

The transaction amount may look normal:

₹4,500

But the previous 15 minutes show:

login from a new device,

password reset,

card added,

multiple failed transactions,

geographic change,

unusually rapid purchase sequence.

The suspicious information lies in the sequence.



Case C — Relational fraud

A new account makes an apparently normal transaction.

But:

its device is shared with 5 suspicious accounts,

its address is connected to multiple chargebacks,

its card appears across several identities,

and its merchant cluster has unusually high fraud rates.

The suspicious information lies in the graph.

SentinelGraph therefore models all three.



7. Main Research Question

The central research question for the project will be:

How much additional fraud-detection value is obtained by combining transaction-level, behavioral, relational, and anomaly signals compared with traditional tabular ML alone?

This should drive the experiments throughout the project.



8. Secondary Research Questions

RQ1

How much does feature engineering improve traditional fraud models?

RQ2

Does sequence modeling provide incremental predictive value over tabular transaction features?

RQ3

Can graph models identify suspicious entities that appear normal when evaluated individually?

RQ4

Which graph architecture performs best for relational fraud detection?

Compare:

GCN

GraphSAGE

GAT

Heterogeneous GNN

RQ5

Can unsupervised anomaly models discover fraud patterns excluded from supervised training?

RQ6

Does combining multiple models outperform the strongest single model?

RQ7

Does probability calibration materially improve risk-based decision thresholds?

RQ8

How much can cost-sensitive threshold optimization reduce financial loss compared with optimizing F1?

RQ9

How rapidly does model performance deteriorate under temporal/concept drift?

RQ10

Can active learning reduce the number of manually labelled samples required?

RQ11

Can model explanations meaningfully help an analyst investigate suspicious activity?

RQ12

What is the trade-off between model complexity, prediction quality, latency, and operating cost?



9. Target Users

SentinelGraph should be designed conceptually for three users.

Fraud Analyst

Needs:

suspicious case queue,

risk score,

evidence,

related accounts,

transaction history,

model explanation,

fraud-ring visualization.



Risk / Data Science Team

Needs:

model performance,

feature distributions,

drift,

calibration,

experiment tracking,

threshold simulations,

retraining decisions.



Risk Operations Manager

Needs:

total fraud exposure,

intervention rate,

false-positive cost,

fraud prevented,

high-risk clusters,

model health.



10. High-Level System Architecture

                        FINANCIAL TRANSACTION                                  │                                  ▼                         Event / Ingestion Layer                                  │                                  ▼                         Feature Engineering                                  │          ┌───────────────────────┼───────────────────────┐          │                       │                       │          ▼                       ▼                       ▼     Tabular Features      Behavioral History      Entity Relationships          │                       │                       │          ▼                       ▼                       ▼      ML Risk Model        Sequence DL Model         Graph Model   LightGBM/CatBoost       LSTM/Transformer       GraphSAGE/GAT          │                       │                       │          └───────────────┬───────┴───────────────┐                          │                       │                          ▼                       ▼                  Anomaly Detection          Rule Signals                  Isolation Forest /                    Autoencoder                          │                          ▼                    Risk Fusion Layer                          │          ┌───────────────┼────────────────┐          ▼               ▼                ▼      Risk Score      Uncertainty       Explanation          │          ▼                  Decision Engine          │               │               │        ALLOW          REVIEW          BLOCK                          │                          ▼                  Investigation System                          │                          ▼                  AI Analyst Assistant



11. Data Strategy

The project should ideally use multiple complementary datasets rather than forcing one dataset to support every research question.

Primary Dataset — Transaction Fraud

Use a large transaction fraud dataset such as IEEE-CIS Fraud Detection.

Important categories available include:

transaction amount,

cards,

addresses,

email domains,

devices,

identity information,

temporal data,

categorical transaction properties,

fraud labels.

Primary use:

tabular ML,

feature engineering,

class imbalance,

sequence generation,

temporal validation,

calibration,

anomaly detection,

ensemble experiments.



Graph Dataset

Use a dataset suitable for relational fraud such as a transaction/network dataset containing explicit entity relationships.

Primary use:

graph construction,

node classification,

relational fraud,

fraud-ring analysis,

GraphSAGE/GAT experiments,

community detection.

Where necessary, construct entity relationships from transaction datasets using shared:

cards,

devices,

addresses,

email domains,

accounts,

merchants.



12. Dataset Documentation

For every dataset, document:

Origin

Where did it come from?

Size

rows,

columns,

file size,

number of entities.

Target

What exactly constitutes fraud?

Feature Groups

Categorize features into:

transaction,

identity,

behavioral,

relational,

temporal,

categorical.

Missingness

Identify:

percentage missing,

structural missingness,

possible information encoded by missing values.

Class Distribution

Example:

Legitimate: 96–99%Fraud:      1–4%

Temporal Coverage

Determine whether timestamp information represents:

real timestamps,

relative timestamps,

ordered event times.

Dataset Limitations

Examples:

anonymized features,

synthetic relationships,

limited geographic information,

historical fraud patterns,

potential domain shift.



13. Data Leakage Checklist

Fraud projects are extremely susceptible to leakage.

Before modeling, explicitly check:

Does a feature become available only after fraud is confirmed?

Does the test set contain future information?

Are entity aggregates calculated using future transactions?

Are graph edges constructed using future interactions?

Are fraud labels leaking through engineered variables?

Are duplicate entities split between train/test incorrectly?

Are target-encoded features computed globally?

The project must maintain point-in-time correctness.

For a transaction at time T, its features may only depend on information available at or before T.



14. Train / Validation / Test Strategy

Avoid random shuffling as the primary evaluation method.

Use temporal splits.

Example:

EARLIER PERIOD                LATER PERIODTRAIN ─────────────── VALIDATION ───── TEST

Potential structure:

70% earliest events → Training

next 15% → Validation

final 15% → Test

Then perform rolling experiments.

Train A → Validate BTrain A+B → Validate CTrain A+B+C → Test D

This allows realistic measurement of model decay.



15. Phase I — Exploratory Data Analysis

EDA should answer questions rather than simply produce plots.

Class Distribution

Investigate:

fraud percentage,

class imbalance severity.



Transaction Amount

Compare:

fraud vs legitimate transaction amounts,

log-transformed distributions,

high-value outliers.



Time

Investigate:

transaction frequency by hour,

weekday,

session,

fraud rate over time.



Device Patterns

Investigate:

number of users/device,

fraud rate for shared devices,

new-device risk.



Card Patterns

Investigate:

users/card,

transaction velocity/card,

historical fraud/card.



Identity Patterns

Investigate:

email domains,

identity completeness,

address reuse.



Missingness

Create missing-value profiles.

Interesting question:

Does missingness itself predict fraud?



Entity Frequency

Investigate:

rare devices,

rare merchants,

rare cards,

high-degree entities.



Temporal Behaviour

Calculate:

transaction gaps,

transactions/hour,

velocity changes,

burstiness.



16. Feature Engineering — Transaction Features

Base features may include:

transaction amount,

product/category,

device,

card characteristics,

location attributes,

email attributes,

identity attributes.

Transformations:

log(amount)

categorical encoding,

frequency encoding,

target-independent count encoding,

missing-value indicators.



17. Feature Engineering — Behavioral Features

This is one of the most important layers.

For each entity/user/card/device, calculate rolling features such as:

transactions_last_10_minutestransactions_last_1_hourtransactions_last_24_hourstotal_amount_last_1_hourtotal_amount_last_24_hoursmean_amount_last_7_daysmedian_amount_last_30_daysamount_vs_user_meanunique_merchants_last_24_hoursunique_devices_last_7_daysunique_cards_last_7_daysfailed_attempts_recenttime_since_previous_transactiontransaction_velocitymerchant_noveltydevice_noveltyaddress_novelty



18. Novelty Features

Examples:

is_new_deviceis_new_merchantis_new_addressis_new_email_domainis_new_card

Relative features:

amount / historical_mean_amountamount / historical_median_amountcurrent_velocity / historical_velocity

These allow the model to measure deviation from personal normal behavior.



19. Graph Features Before GNNs

Before training graph neural networks, create classical graph-derived features.

Examples:

device_degreecard_degreeaccount_degreenumber_of_shared_devicesnumber_of_shared_addressesfraud_neighbor_countfraud_neighbor_ratiotwo_hop_fraud_exposureconnected_component_sizelocal_clustering_coefficient

This creates a useful baseline:

How much does classical graph feature engineering achieve before deep GNNs?



20. Baseline Model Ladder

Every advanced model should beat a meaningful baseline.

Baseline 0

Majority classifier.

Purpose:

Demonstrate why accuracy is meaningless.



Baseline 1

Logistic Regression.

Purpose:

interpretable linear baseline,

understand feature scaling,

establish calibration baseline.



Baseline 2

Random Forest.



Baseline 3

XGBoost / LightGBM.



Baseline 4

CatBoost.

Particularly useful where categorical features remain important.

The strongest boosted-tree model becomes the tabular benchmark.



21. Class Imbalance Strategy

Evaluate multiple approaches.

Class Weighting

class_weightscale_pos_weight

Threshold Adjustment

Train normally but choose a cost-sensitive threshold.

Under-sampling

Experiment carefully.

Over-sampling

Potentially investigate SMOTE but document why naive synthetic oversampling may distort fraud distributions.

Focal Loss

For neural models.

Do not blindly use oversampling because a tutorial recommends it.

Compare empirically.



22. Evaluation Metrics

Avoid Using Accuracy as the Main Metric

A system predicting every transaction as legitimate may achieve extremely high accuracy.

Primary metrics should include:

Precision

Among flagged transactions:

How many were actually fraudulent?

Recall

Among fraudulent transactions:

How many did we detect?

F1

Useful but not sufficient.

ROC-AUC

Secondary metric.

PR-AUC

Much more relevant under severe imbalance.

Precision@K

If analysts can review only K cases.

Recall@K

How much fraud is captured inside review capacity?



23. Business Metrics

The final system should also evaluate business consequences.

Define:

Fraud Loss=Sum(amount of undetected fraud)

and:

Customer Friction Cost=False Positives × intervention cost

Then:

Expected Cost=Fraud Loss+Customer Friction Cost

Possible action costs:

ALLOWREVIEWSTEP-UP AUTHENTICATIONBLOCK

Each can have different costs.



24. Threshold Optimization

A classifier produces probability:

P(fraud) = 0.72

But the threshold does not automatically need to be 0.5.

Search for:

threshold*=argmin Expected Business Cost

Compare:

F1-maximizing threshold,

recall-maximizing threshold,

cost-minimizing threshold.

This becomes an important project finding.



25. Probability Calibration

Risk systems need meaningful probabilities.

Evaluate:

Reliability Diagram

If the model predicts 0.8 probability for 100 transactions, approximately 80 should ideally correspond to the positive outcome.

Metrics:

Brier Score,

Expected Calibration Error.

Methods:

Platt Scaling,

Isotonic Regression,

Temperature Scaling for deep models.

Compare calibrated vs uncalibrated models.



26. Phase II — Behavioral Sequence Modeling

The next model examines a user’s history rather than treating each transaction independently.

For user u:

Transaction 1Transaction 2Transaction 3...Transaction N

Create fixed or variable-length sequences.

Each event may contain:

amountmerchantdevicetime-gapcategorylocationrisk features



27. Sequence Model Baseline — LSTM / GRU

First train:

LSTM,

GRU.

Purpose:

establish sequential-learning baseline,

learn recurrent networks,

understand sequence padding/masking,

evaluate whether behavior history improves predictions.



28. Transformer Behavioral Model

Later build a Transformer-based model.

Transaction representation:

Transaction Embedding=amount representation+merchant embedding+device embedding+time-gap embedding+categorical embeddings+positional/time encoding

Feed sequence into:

Transformer Encoder

Output:

Behavioral Fraud Probability



29. Temporal Representation Questions

Experiment with:

chronological positional encoding,

relative time gaps,

time-of-day embeddings,

day-of-week embeddings.

Important question:

Should transaction order or actual elapsed time matter more?



30. Self-Supervised Transaction Pretraining

Optional but important advanced extension.

Fraud labels are limited, while transaction data is abundant.

Pretrain transaction representations.

Masked Feature Modeling

Example:

amount     = ₹4,500merchant   = [MASK]device     = D17hour       = 02

Predict masked attribute.



Contrastive Learning

Encourage:

similar behavioral windows→ similar embeddings

and:

different behaviors→ distant embeddings

Then fine-tune these representations for fraud detection.

Research question:

Does self-supervised pretraining improve performance with fewer fraud labels?



31. Phase III — Graph Construction

Build a heterogeneous graph.

Node Types

Possible nodes:

User

Transaction

Card

Device

Address

Email

Merchant



Edge Types

User --MADE--> TransactionTransaction --USED_CARD--> CardTransaction --FROM_DEVICE--> DeviceTransaction --AT_MERCHANT--> MerchantUser --HAS_EMAIL--> EmailUser --HAS_ADDRESS--> Address



32. Why Graphs Matter

Consider:

User A ---- Device X ---- User B                       \                        User C

If User A and B are known suspicious accounts, User C inherits relational risk even before it develops a long fraudulent history.

This information is difficult to capture with independent transaction rows.



33. Graph Model Ladder

Do not jump immediately to the most complicated model.

Compare:

GCN

Basic message passing baseline.

GraphSAGE

Important because it learns neighborhood aggregation and can generalize to unseen nodes.

GAT

Adds attention over neighbors.

Heterogeneous GNN

Different relation/node types receive separate transformations.

Example:

User → Device

should not necessarily have the same meaning as:

User → Address



34. Graph Representation Learning

Each node receives embedding:

h_userh_deviceh_cardh_merchant

Message-passing layers propagate relational information.

Example:

User embedding=own features+information from connected devices+information from cards+information from merchants

Final node embedding feeds a fraud classifier.



35. Fraud-Ring Discovery

Prediction at the individual level is not enough.

Use graph algorithms:

connected components,

Louvain,

Leiden,

community detection.

Then calculate cluster risk.

Example:

Fraud Ring #23Accounts      37Devices        8Cards         21Addresses      4Merchants      6Average Risk   0.87



36. Graph Investigation Interface

Visualize suspicious networks.

Example:

         User A          /   \     Device X  Card 1       /   \   User B  User C            |         Merchant Z

Clicking a node should show:

risk,

historical fraud,

transaction count,

linked entities,

model explanation.

This becomes one of the project’s strongest demos.



37. Temporal Graph Extension

Relationships evolve over time.

Example:

10:01 Account Created10:05 Device Added10:08 Card Added10:10 Transaction10:13 Same Device Used by New Account

Advanced extension:

temporal graph features,

time-aware message passing,

temporal graph neural networks.

Research question:

Does knowing when suspicious relationships appear improve detection?



38. Phase IV — Unsupervised Anomaly Detection

Supervised models primarily learn fraud patterns present in training labels.

Fraud changes.

Therefore build another engine specifically for unknown abnormalities.



39. Classical Anomaly Models

Compare:

Isolation Forest,

Local Outlier Factor,

One-Class SVM.



40. Autoencoder

Train on primarily legitimate behavior.

Input  ↓Encoder  ↓Latent Representation  ↓Decoder  ↓Reconstruction

For normal behavior:

reconstruction error ≈ low

For abnormal behavior:

reconstruction error ↑

Use reconstruction error as anomaly score.



41. Variational Autoencoder

Advanced experiment:

learn probabilistic latent space,

investigate whether unusual events appear in low-density regions.



42. Unknown-Fraud Experiment

Create an explicit experiment.

Temporally or artificially withhold one fraud category/pattern during supervised training.

Train supervised model.

Then ask:

Can anomaly detection identify the withheld pattern?

This makes the anomaly component scientifically meaningful.



43. Multi-Model Risk Fusion

Eventually SentinelGraph produces:

Tabular risk scoreSequence risk scoreGraph risk scoreAnomaly score

Example:

Tabular      0.63Sequence     0.84Graph        0.92Anomaly      0.73



44. Ensemble Strategies

Compare:

Simple Average

Weighted Average

Weights based on validation performance.

Logistic Meta-Model

Inputs:

[xgb_score, transformer_score, graph_score, anomaly_score]

Output:

final_fraud_probability

Stacking

Train a meta-model with out-of-fold predictions.

Research question:

Which modalities provide independent information?



45. Ablation Study

This should become one of the headline experiments.

Compare:

A. Tabular onlyB. Tabular + behavioral featuresC. Tabular + sequence modelD. Tabular + graphE. Tabular + anomalyF. Tabular + sequence + graphG. Full ensemble

Metrics:

PR-AUC,

Recall@K,

expected financial loss,

inference latency.

This tells us exactly what each component contributes.



46. Explainability Layer

The system must explain risk at multiple levels.



47. Tabular Explainability

Use SHAP.

Possible explanation:

Risk increased because:+ New device                   +0.18+ Transaction amount 8× norm  +0.16+ 11 transactions / 20 min    +0.13+ New merchant                 +0.08



48. Sequence Explainability

Show which historical events most influenced behavioral risk.

Example:

Most influential events:T-1 New device loginT-2 ₹48,000 transactionT-3 password resetT-4 unusual geographical change

Investigate:

attention visualization,

Integrated Gradients,

perturbation-based explanation.



49. Graph Explainability

Show:

influential neighbors,

suspicious shared entities,

important graph paths.

Example:

Graph risk elevated because:Device D43:connected to 7 high-risk accountsAddress A18:connected to 3 confirmed fraudulent accountsMerchant M2:high fraud-neighborhood density



50. Confidence / Uncertainty

Not every prediction should be equally trusted.

Track:

prediction probability,

model disagreement,

calibration,

ensemble variance.

Potential policy:

Risk < 0.20→ AllowRisk 0.20–0.65→ Allow / monitorRisk 0.65–0.85→ Additional authenticationRisk > 0.85→ Analyst review / block

Thresholds should eventually be learned from cost simulations rather than arbitrarily fixed.



51. Decision Engine

Fraud prediction should become a decision.

Potential actions:

ALLOWALLOW + MONITORSTEP-UP AUTHENTICATIONMANUAL REVIEWBLOCK

The action should depend on:

predicted fraud probability,

transaction value,

model confidence,

business cost,

analyst capacity.



52. Analyst Capacity Optimization

Suppose analysts can review only:

1,000 transactions/day

Then we cannot review every suspicious transaction.

Rank cases by:

expected preventable loss

instead of risk probability alone.

Example:

priority=risk × transaction_amount × confidence

Later create better policies.



53. Concept Drift

Fraud patterns evolve.

Monitor:

transaction amount distribution,

categorical distributions,

fraud rate,

model score distribution,

embedding distribution,

model precision,

recall,

calibration.



54. Drift Metrics

Potential techniques:

Population Stability Index,

KS Test,

Jensen-Shannon divergence,

KL divergence where appropriate,

embedding-distance monitoring.

Differentiate:

Feature Drift

P(X) changes.

Label Drift

P(Y) changes.

Concept Drift

P(Y|X) changes.

The last one is the most important and hardest.



55. Retraining Policy

Do not automatically retrain simply because drift is detected.

Possible pipeline:

Drift Detected      ↓Performance Validation      ↓Candidate Retraining      ↓Offline Evaluation      ↓Champion vs Challenger      ↓Deploy only if better



56. Champion–Challenger Framework

Production model:

Champion

Candidate:

Challenger

Compare:

PR-AUC,

expected cost,

calibration,

latency,

stability.

Deploy challenger only when predefined criteria are met.



57. Active Learning

Fraud labels require analyst effort.

Use uncertainty sampling.

Cases near model uncertainty or with large model disagreement are prioritized for labeling.

Example:

Tabular:    0.82Graph:      0.20Sequence:   0.76

Strong disagreement.

This case may contain useful information.

Active-learning loop:

Model ↓Uncertain / informative cases ↓Human label ↓Training dataset ↓Model retraining

Measure:

performance vs number of labelled examples.



58. AI Investigation Assistant

This is where GenAI enters.

It should not make the fraud decision.

The ML system creates structured evidence.

Example:

risk_score = 0.91transaction_amount = 84000user_historical_median = 7200device_status = NEWtransactions_last_15m = 12graph_neighbors:- device shared with 5 suspicious accounts- address linked to 2 confirmed fraud casesSHAP:- device novelty- amount deviation- transaction velocity

The LLM receives this evidence.



59. AI Assistant Responsibilities

It can:

summarize why a transaction was flagged,

explain graph relationships,

summarize recent behavioral changes,

create analyst case notes,

compare model outputs,

answer natural-language investigation queries.

Examples:

Why was transaction T293 flagged?

Show the strongest evidence linking this user to known fraud.

Summarize this account’s behavior over the last 24 hours.

Which model contributed most to this decision?



60. AI Guardrails

The LLM must not independently determine:

Fraud / Not Fraud

Final risk remains based on:

trained models,

risk fusion,

deterministic decision rules.

The LLM is:

EXPLANATION + INVESTIGATION

not:

DECISION AUTHORITY

Additional guardrails:

structured inputs,

evidence citations,

no unsupported claims,

deterministic numeric values,

prompt injection resistance where relevant.



61. MLOps Architecture

Raw Data   ↓Data Validation   ↓Feature Pipeline   ↓Training Dataset   ↓Experiment Tracking   ↓Model Training   ↓Evaluation   ↓Model Registry   ↓Deployment   ↓Inference   ↓Prediction Logs   ↓Monitoring   ↓Retraining



62. MLflow

Use MLflow for:

parameters,

metrics,

artifacts,

experiment comparison,

model registry.

For every experiment log:

modelfeaturesdataset versionhyperparametersPR-AUCprecisionrecallcalibrationexpected losstraining time



63. Data / Model Versioning

Use DVC or an equivalent reproducibility system for:

dataset versions,

processed datasets,

feature versions,

trained model artifacts.

Goal:

Every model result should be reproducible from a Git commit + data version + configuration.



64. Configuration-Driven Experiments

Avoid hard-coded experiments.

Example:

model:  type: lightgbmfeatures:  behavioral: true  graph: falsetraining:  temporal_split: trueimbalance:  strategy: weightedevaluation:  primary_metric: pr_auc

This improves reproducibility.



65. Real-Time Architecture

Long-term target:

Transaction Event       ↓Kafka / Redpanda       ↓Stream Processor       ↓Online Feature Calculation       ↓Feature Cache / Store       ↓Inference Service       ↓Risk Fusion       ↓Decision Engine       ↓Case / Alert System



66. Backend

Recommended:

FastAPI

Endpoints might include:

POST /predictGET /transaction/{id}GET /user/{id}/riskGET /graph/{entity_id}GET /casesPOST /cases/{id}/reviewGET /model/health



67. Storage

Potential architecture:

PostgreSQL

For:

transaction metadata,

users,

cases,

analyst decisions,

model versions.

Redis

For:

low-latency features,

recent behavior,

counters,

short-lived state.

Graph Storage

Start without unnecessary infrastructure.

Graph data may initially be generated using:

NetworkX,

PyTorch Geometric structures.

Introduce Neo4j only if interactive graph querying becomes genuinely useful.



68. Streaming

Potential use of Kafka/Redpanda:

Topics:

transactions.rawtransactions.featuresfraud.predictionsfraud.alertsanalyst.feedback

Important concepts to learn:

partitions,

consumer groups,

ordering,

replay,

offset management,

delivery semantics.



69. Model Serving

Possible architecture:

API Gateway    ↓Feature Retrieval    ↓Tabular ModelSequence ModelGraph Model    ↓Risk Fusion

Investigate latency implications.

Not every model must necessarily execute synchronously.

Potential design:

Fast Path

Tabular + behavioral risk.

Deep Analysis Path

Graph + anomaly + AI investigation.

This creates a realistic latency architecture.



70. Model Optimization

Advanced experiments:

batch inference,

ONNX export,

quantization,

model distillation,

optimized PyTorch compilation,

caching embeddings.

Measure before and after.



71. System Performance Metrics

Track:

p50 inference latencyp95 inference latencyp99 inference latencytransactions/secfeature retrieval latencygraph inference latencymemory consumptionmodel size



72. Model Monitoring Dashboard

Dashboard should show:

Detection Metrics

PR-AUC,

recall,

precision,

false-positive rate.

Business Metrics

estimated fraud prevented,

missed fraud value,

review volume,

customer-friction estimate.

Model Health

drift,

calibration,

confidence.

Infrastructure

inference latency,

throughput,

error rate.



73. Analyst Dashboard

Core sections:

Fraud Overview

transactions processed,

high-risk alerts,

estimated exposure,

analyst queue.

Transaction Explorer

User Risk Profile

Fraud Graph

Model Explanation

Fraud-Ring Explorer

Drift Monitoring

Experiment Dashboard

AI Investigation Panel

UI exists to demonstrate the science.

It should not consume most project-development time.



74. Experiment Matrix

Maintain this throughout development.

ID

Experiment

Model

Feature Set

Primary Metric

Result

Key Learning

E001

Baseline

Logistic Regression

Raw

PR-AUC

TBD



E002

Tree Baseline

LightGBM

Raw

PR-AUC

TBD



E003

Engineered

LightGBM

Behavioral

PR-AUC

TBD



E004

Sequence

LSTM

Transaction History

PR-AUC

TBD



E005

Transformer

Transformer

Transaction History

PR-AUC

TBD



E006

Graph

GraphSAGE

Entity Graph

PR-AUC

TBD



E007

Graph

GAT

Entity Graph

PR-AUC

TBD



E008

Anomaly

Autoencoder

Behavioral

Recall Unknown Fraud

TBD



E009

Ensemble

Stacking

All

PR-AUC

TBD



E010

Cost Sensitive

Ensemble

All

Expected Loss

TBD



Continue expanding this table.



75. Failure Analysis

Maintain a dedicated failure-analysis document.

For every model inspect:

False Positives

Questions:

Why did legitimate behavior look suspicious?

Was this a new-device effect?

High-value legitimate customer?

Rare merchant?

Missing historical information?



False Negatives

Questions:

Did fraud imitate normal behavior?

Was relational evidence missing?

Did the fraud type not exist in training?

Did feature aggregation hide short-term behavior?



76. Error Cohorts

Create cohorts:

new usershigh-value transactionsnew devicesrare merchantscross-border transactionslow-history accountsshared-device accountslarge graph components

Calculate performance separately.

This may reveal hidden model weaknesses.



77. Fairness / Responsible ML Exploration

Even if dataset attributes are anonymized, document the broader issue.

Fraud models can disproportionately inconvenience groups of legitimate customers.

Discuss:

proxy variables,

geographic bias,

historical-label bias,

feedback loops.

Where demographic attributes are available in a safe dataset, evaluate group-level performance disparities.

Do not fabricate fairness claims if the dataset cannot support them.



78. Four-Month Development Roadmap

MONTH 1 — Data Science Foundation

Goals

Build a scientifically strong tabular fraud system.

Tasks

repository architecture,

data ingestion,

dataset documentation,

EDA,

temporal splitting,

leakage audit,

preprocessing,

behavioral feature engineering,

Logistic Regression,

Random Forest,

LightGBM,

XGBoost/CatBoost,

imbalance experiments,

threshold analysis,

calibration,

MLflow integration,

baseline dashboard.

Deliverable

A complete classical fraud-detection pipeline with reproducible experiments.



79. Month 2 — Deep Learning + Graph Intelligence

Sequence Modeling

user sequences,

LSTM,

GRU,

Transformer,

temporal embeddings.

Graph

graph construction,

graph statistics,

classical graph features,

GCN,

GraphSAGE,

GAT,

graph visualizer.

Deliverable

Demonstrate whether sequence and graph information improve beyond tabular ML.



80. Month 3 — Advanced Intelligence

Build:

anomaly detection,

autoencoders,

unknown-fraud experiments,

model fusion,

calibration,

fraud-ring discovery,

active learning,

explainability,

self-supervised pretraining if useful.

Deliverable

Multi-model adaptive fraud intelligence engine.



81. Month 4 — Production + Research Polish

Build:

streaming architecture,

FastAPI inference,

Redis feature state,

Kafka/Redpanda,

model registry,

model monitoring,

concept drift,

retraining workflow,

performance benchmarking,

AI investigation assistant,

complete dashboards,

documentation,

deployment,

interview preparation.

Final Deliverable

A production-style, research-backed fraud intelligence platform.



82. Learning Syllabus

This project is also a structured ML course for me.

Statistics

probability

conditional probability

Bayes theorem

imbalanced distributions

calibration

statistical tests

confidence intervals

drift tests

Machine Learning

Logistic Regression

Random Forest

XGBoost

LightGBM

CatBoost

feature engineering

hyperparameter optimization

class imbalance

ensemble learning

Deep Learning

neural-network fundamentals

embeddings

LSTM

GRU

attention

Transformers

autoencoders

self-supervised learning

Graph ML

graph fundamentals

node/edge features

message passing

GCN

GraphSAGE

GAT

heterogeneous graphs

community detection

graph explainability

temporal graphs

Advanced ML

anomaly detection

probability calibration

active learning

cost-sensitive ML

representation learning

concept drift

uncertainty

MLOps

MLflow

DVC

model registry

reproducibility

model serving

monitoring

champion/challenger

retraining pipelines

Engineering

FastAPI

PostgreSQL

Redis

Docker

Kafka / Redpanda

stream processing

API design

load testing



83. Target Technology Stack

Languages

PythonSQL

Core Data

NumPyPandasPolarsSciPy

Classical ML

scikit-learnXGBoostLightGBMCatBoost

Deep Learning

PyTorch

Graph Learning

PyTorch GeometricNetworkX

Explainability

SHAPIntegrated Gradients / CaptumGraph explanation methods

MLOps

MLflowDVC

API

FastAPI

Databases

PostgreSQLRedis

Streaming

Kafka or Redpanda

Infrastructure

DockerDocker Compose

Potential later:

Kubernetes

Monitoring

Evidently/custom monitoringPrometheusGrafana



84. Proposed Repository Structure

sentinelgraph-ai/│├── README.md├── ROADMAP.md├── ARCHITECTURE.md├── MODEL_CARD.md├── DATA_CARD.md│├── docs/│   ├── problem.md│   ├── dataset.md│   ├── leakage.md│   ├── feature-engineering.md│   ├── experiments.md│   ├── graph-design.md│   ├── modeling.md│   ├── failure-analysis.md│   ├── explainability.md│   ├── monitoring.md│   ├── benchmarks.md│   └── design-decisions/│├── data/│   ├── raw/│   ├── interim/│   └── processed/│├── notebooks/│   ├── 01_eda.ipynb│   ├── 02_baselines.ipynb│   ├── 03_features.ipynb│   ├── 04_sequence.ipynb│   └── 05_graph.ipynb│├── src/│   ├── data/│   ├── features/│   ├── models/│   │   ├── tabular/│   │   ├── sequence/│   │   ├── graph/│   │   ├── anomaly/│   │   └── ensemble/│   ││   ├── evaluation/│   ├── explainability/│   ├── monitoring/│   ├── serving/│   └── pipelines/│├── api/│├── dashboard/│├── configs/│├── scripts/│├── tests/│   ├── unit/│   ├── integration/│   └── model/│├── docker/│└── .github/    └── workflows/



85. Architecture Decision Records

Maintain ADRs.

Examples:

ADR-001 — Why temporal split instead of random splitADR-002 — Why PR-AUC is the primary metricADR-003 — Why LightGBM is the initial tabular benchmarkADR-004 — Why GraphSAGE before heterogeneous GNNADR-005 — Why the LLM cannot make fraud decisionsADR-006 — Why graph storage does/doesn't require Neo4jADR-007 — Why Kafka is introducedADR-008 — Online vs offline feature computationADR-009 — Risk threshold strategyADR-010 — Champion/challenger deployment

These documents become extremely useful before interviews.



86. README Structure

The final README should contain:

SentinelGraph AI

What it is

Why fraud detection requires more than classification

Demo

Architecture

Dataset

Research Questions

Models

Experiments

Key Results

Ablation Study

Fraud Graph

Explainability

Production Architecture

Monitoring

Benchmarks

Local Setup

Limitations

Roadmap



87. Project Demo Story

The final demo should tell a story.

Step 1

Incoming transaction:

₹84,250

At first glance it may look normal.

Step 2

Tabular model:

Risk = 0.68

Step 3

Behavioral model discovers:

new device,

9× historical amount,

11 recent transactions.

Risk = 0.85

Step 4

Graph model discovers:

device shared with suspicious accounts,

linked address in fraud cluster.

Risk = 0.94

Step 5

Ensemble:

Final Risk = 0.91

Step 6

Decision:

MANUAL REVIEW

Step 7

AI analyst explains why.

Step 8

Open graph and visualize suspicious network.

This is a much stronger demo than displaying an ROC curve.



88. Metrics Dashboard to Maintain

Keep an evolving table.

Dataset size:Fraud ratio:Best tabular PR-AUC:Best sequence PR-AUC:Best GNN PR-AUC:Full ensemble PR-AUC:Recall @ 1% FPR:Precision @ analyst capacity:Expected fraud loss reduction:Calibration error:p95 inference latency:Throughput:Graph size:Number of experiments:Number of labelled examples saved by active learning:

Do not invent these numbers.

Populate them throughout development.



89. Target Resume Bullet Bank

These are the target-state resume bullets that project development should make defensible by interview season.

Bullet 1 — Overall

Developed an adaptive fraud intelligence platform combining gradient-boosted models, Transformer-based behavioral modeling, graph neural networks, and unsupervised anomaly detection for multi-signal transaction risk assessment.

Bullet 2 — Graph ML

Constructed heterogeneous user–transaction–card–device graphs and trained GraphSAGE/GAT models to identify relational fraud patterns and coordinated fraud clusters beyond transaction-level signals.

Bullet 3 — DS Rigor

Designed temporal, leakage-safe feature pipelines for highly imbalanced financial data, benchmarking models using PR-AUC, Recall@K, calibrated probabilities, and cost-sensitive expected-loss metrics.

Bullet 4 — Advanced ML

Built ensemble risk fusion, explainability, anomaly detection, active-learning feedback loops, and concept-drift monitoring to improve adaptability to evolving fraud patterns.

Bullet 5 — MLOps / Production

Productionized model inference using FastAPI, streaming feature pipelines, Redis/PostgreSQL, MLflow/DVC lifecycle management, and an evidence-grounded AI investigation assistant for analyst case review.

These should progressively acquire genuine metrics.



90. Metric-Enhanced Bullet Template

Eventually convert:

improved fraud detection

into:

Improved PR-AUC from X → Y

or:

Increased fraud recall by X% at the same false-positive rate

or:

Reduced simulated expected fraud loss by X%

or:

Served predictions at p95 latency of X ms

Real measured numbers are significantly stronger than invented scale claims.



91. Interview Questions — Data Science Fundamentals

Prepare to answer:

Why isn’t accuracy useful here?

Why PR-AUC instead of ROC-AUC?

What causes class imbalance?

How did you handle imbalance?

Why didn’t you simply use SMOTE?

How did you choose a classification threshold?

What is probability calibration?

What is Brier score?

How did you perform feature engineering?

How did you detect leakage?

Why temporal split?

What are false-positive and false-negative costs?

Why LightGBM?

XGBoost vs LightGBM vs CatBoost?

What are your strongest features?

How did you deal with missing values?

Why not maximize F1?

What is Precision@K?

How did you tune hyperparameters?

How did you evaluate robustness?



92. Interview Questions — Deep Learning

Why use an LSTM?

Why Transformer?

How are transactions represented?

Why embeddings?

What is attention?

How do you represent time gaps?

How do you handle long transaction histories?

How do you pad sequences?

What loss function did you use?

Why focal loss?

What advantages did Transformer provide?

Did the Transformer actually beat LightGBM?

Why might tree models outperform DL on tabular fraud data?

How did you avoid overfitting?

What did self-supervised learning contribute?



93. Interview Questions — Graph ML

Why model fraud as a graph?

What are the nodes?

What are the edges?

What is message passing?

Explain GraphSAGE.

Why GraphSAGE over GCN?

What does graph attention do?

How do you handle unseen nodes?

What is oversmoothing?

What is neighborhood sampling?

How large is your graph?

How did you construct labels?

How do you prevent temporal leakage in graph construction?

What is a heterogeneous graph?

Why not simply use graph-derived features with LightGBM?

Did GNNs outperform classical graph features?

How did you identify fraud rings?

How do you explain a graph prediction?

How would you scale graph inference?

How would a new account with no neighbors be handled?



94. Interview Questions — Anomaly Detection

Why anomaly detection if supervised fraud labels exist?

What is Isolation Forest?

How does an autoencoder detect anomalies?

How do you choose anomaly threshold?

How do you evaluate anomaly detection without labels?

What happens if legitimate behavior changes?

How did you test unknown fraud detection?

Autoencoder vs VAE?

How does anomaly detection complement supervised models?

What are its biggest limitations?



95. Interview Questions — Production / MLOps

How is your model deployed?

What features are computed online?

What features are computed offline?

Why Redis?

Why Kafka?

How do you monitor drift?

What happens when drift occurs?

When do you retrain?

How do you roll back a bad model?

What is champion/challenger deployment?

How do you version datasets?

How do you reproduce experiments?

How does MLflow help?

What is your p95 latency?

What happens at 10K transactions/sec?

How would you scale inference?

Where would caching help?

How do you monitor model failures?

What happens when the graph model is unavailable?

Can the system degrade gracefully?



96. Interview Questions — AI Layer

Why add an LLM?

Why not use the LLM to classify fraud directly?

What data does the LLM see?

How do you prevent hallucination?

How do you ground responses?

Can analyst explanations cite model evidence?

How do you protect sensitive financial information?

What happens if the LLM service is unavailable?

How do you evaluate investigation summaries?

What does the AI layer contribute that SHAP doesn’t?



97. Behavioral Interview Story Bank

The project should eventually produce stories around:

A model that initially looked strong but leaked future data.

A deep model that failed to outperform a simpler baseline.

A graph architecture that became too expensive.

A misleading evaluation metric.

A false-positive pattern you discovered.

An experiment that contradicted your original assumption.

A production latency bottleneck.

A drift event.

A design decision where you removed unnecessary technology.

These stories are often more valuable than success stories.



98. Things We Should Explicitly Avoid

Do not turn SentinelGraph into:

a dashboard with a fraud model hidden underneath.

Do not blindly add:

Kubernetes,

Spark,

Kafka,

Neo4j,

Transformers,

GNNs

unless the previous stage creates a real reason.

Do not claim:

production users,

banking partnerships,

money saved,

fraud prevented in the real world,

unless such evidence genuinely exists.

Simulated business impact should be labelled simulated.



99. Project Success Criteria

By interview season, SentinelGraph should be successful if I can confidently demonstrate:

Data Science

strong EDA,

thoughtful feature engineering,

leakage-safe evaluation,

proper metrics,

calibrated probabilities,

cost-sensitive decisions.

ML

boosted-tree baselines,

model comparison,

ensembles.

DL

sequence model,

Transformer.

Graph AI

relational fraud graph,

at least two GNN architectures,

graph explainability,

fraud-ring discovery.

Unsupervised ML

anomaly detector,

unknown-pattern experiment.

Advanced

active learning,

concept drift.

MLOps

experiment tracking,

model registry,

versioning,

monitoring.

Engineering

inference API,

persistent storage,

streaming architecture,

Dockerized system.

AI

evidence-grounded investigation assistant.

Research

ablation studies,

failure analysis,

documented experiments.

Communication

architecture diagrams,

README,

technical documentation,

reproducible demo.



100. Final Project Identity

SentinelGraph should ultimately communicate:

I don’t view machine learning as model.fit(). I understand the full lifecycle: defining a decision problem, understanding messy data, preventing leakage, building baselines, developing advanced models, comparing them scientifically, measuring uncertainty and cost, explaining their decisions, deploying them, monitoring degradation, and integrating them into a real decision workflow.

That is the purpose of this project.



101. Final Portfolio Position

Within the Data Science resume:

SentinelGraph AI

should own:

Machine Learning + Deep Learning + Graph AI + Anomaly Detection + MLOps + AI

DemandOS

will own:

Time Series + Forecasting + Uncertainty + Optimization

CausalLift

will own:

Statistics + Experimentation + Causal Inference + Uplift Modeling

Together, the three projects should make the Data Science portfolio broad without becoming repetitive.



102. Development Philosophy

The goal over the next four months is not to accumulate features.

The progression should always remain:

Understand the problem        ↓Understand the data        ↓Build a simple baseline        ↓Measure it correctly        ↓Understand its failures        ↓Introduce a more advanced model        ↓Prove whether it actually helps        ↓Deploy it        ↓Measure it in operation        ↓Understand its limitations

If we follow that progression, SentinelGraph will become both a strong resume project and a serious four-month ML learning journey.
