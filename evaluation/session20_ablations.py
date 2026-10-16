"""Controlled CatBoost feature ablations and paired development uncertainty.

All models share training targets, evaluation IDs and hyperparameters. Bootstrap
resamples rows, not temporal/entity blocks; these reused validation rows are not
an independent final test. Reserved test data is never read.
"""

import argparse
import hashlib
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import catboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

from anomaly_models.train_anomaly_models import FEATURES, load_split
from decision_models.evaluate_decisions import aligned_saved_scores, file_hash
from deep_learning.run_behavioral import metrics, write_json

ALL_FEATURES = FEATURES.copy()
GROUPS = {
    "AmountOnly": ["TransactionAmt"],
    "AmountPlusActivity": ["TransactionAmt", "graph_card_prior_degree", "graph_product_prior_degree"],
    "AmountPlusRelationships": ["TransactionAmt", "graph_card_product_prior_count", "graph_card_distinct_products"],
    "AllFive": ALL_FEATURES.copy(),
}


def align_evaluation(validation, reference):
    """Use reference IDs as the population and raw validation as label/time truth."""
    if reference.empty or not reference.TransactionID.is_unique or reference.TransactionID.isna().any():
        raise ValueError("Reference evaluation IDs must be nonempty, unique and nonmissing")
    if not validation.TransactionID.is_unique or not reference.TransactionID.isin(validation.TransactionID).all():
        raise ValueError("Reference IDs must match unique original validation transactions")
    evaluation = validation.set_index("TransactionID").loc[reference.TransactionID].reset_index()
    evaluation = evaluation.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    # The old CatBoost score is checked only for valid alignment, not used as an
    # ablation prediction. Session 18 used 200 trees; this study fits 400 trees.
    aligned_saved_scores(evaluation, reference, "catboost")
    if not np.isfinite(evaluation.TransactionDT.to_numpy(dtype=float)).all():
        raise ValueError("Evaluation timestamps must be finite")
    if not evaluation.isFraud.isin([0, 1]).all() or evaluation.isFraud.nunique() != 2:
        raise ValueError("Evaluation requires both binary label classes")
    return evaluation


def chronological_blocks(times, n_blocks=5):
    """Return consecutive positional slices, advancing cuts beyond timestamp ties."""
    times = np.asarray(times)
    if n_blocks < 1 or not len(times) or not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError("Need finite sorted times and a positive block count")
    boundaries = [0]
    for cut in np.linspace(0, len(times), n_blocks + 1, dtype=int)[1:-1]:
        while 0 < cut < len(times) and times[cut] == times[cut - 1]:
            cut += 1
        if boundaries[-1] < cut < len(times):
            boundaries.append(int(cut))
    boundaries.append(len(times))
    return [slice(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]


def block_metrics(labels, scores, ids):
    """A block with no fraud has unavailable AP/recall; don't invent a zero AP."""
    if not np.asarray(labels).sum():
        return {"average_precision": None, "precision_at_100": 0.0, "recall_at_100": None,
                "validation_prevalence": 0.0, "val_examples": len(labels)}
    return metrics(labels, scores, ids)


def paired_ap_bootstrap(labels, amount_scores, full_scores, repetitions=1000, seed=42):
    """Sample identical row indices for both models on each draw; scores stay frozen.

    Replicates missing either class are excluded and counted. Percentiles describe
    this row-resampling scheme, not future-period reliability or a hypothesis-test
    p-value. No bootstrap replicate retrains a model.
    """
    labels, amount_scores, full_scores = map(np.asarray, (labels, amount_scores, full_scores))
    if not (labels.ndim == amount_scores.ndim == full_scores.ndim == 1):
        raise ValueError("Bootstrap inputs must be one-dimensional")
    if not (len(labels) == len(amount_scores) == len(full_scores)) or len(labels) < 2:
        raise ValueError("Bootstrap scores and labels must be aligned and nonempty")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Bootstrap evaluation needs both binary label classes")
    if not np.isfinite(amount_scores).all() or not np.isfinite(full_scores).all() or repetitions < 2:
        raise ValueError("Need finite scores and at least two bootstrap draws")
    rng = np.random.default_rng(seed)
    records, skipped = [], 0
    for draw in range(repetitions):
        sampled = rng.integers(0, len(labels), size=len(labels))
        y = labels[sampled]
        if len(np.unique(y)) != 2:
            skipped += 1
            continue
        ap_amount = float(average_precision_score(y, amount_scores[sampled]))
        ap_full = float(average_precision_score(y, full_scores[sampled]))
        records.append({"draw": draw + 1, "sampled_frauds": int(y.sum()),
                        "amount_only_ap": ap_amount, "all_five_ap": ap_full,
                        "delta_ap": ap_full - ap_amount})
    if len(records) < 2:
        raise ValueError("Insufficient valid bootstrap replicates containing both classes")
    draws = pd.DataFrame(records)
    low, high = np.quantile(draws.delta_ap, [.025, .975])
    point = float(average_precision_score(labels, full_scores) - average_precision_score(labels, amount_scores))
    summary = {
        "comparison": "AllFive AP minus AmountOnly AP", "point_delta_ap": point,
        "interval_method": "paired transaction-row bootstrap, 95% percentile interval",
        "ci_lower": float(low), "ci_upper": float(high),
        "interval_includes_zero": bool(low <= 0 <= high),
        "mean_bootstrap_delta": float(draws.delta_ap.mean()),
        "bootstrap_delta_std": float(draws.delta_ap.std(ddof=1)),
        "bootstrap_fraction_positive": float(draws.delta_ap.gt(0).mean()),
        "requested_draws": repetitions, "valid_draws": len(draws),
        "skipped_single_class_draws": skipped, "seed": seed,
        "limitations": "Fixed models, row independence assumption, one seed and previously examined development interval; not a p-value or final-test claim",
    }
    return draws, summary


def input_hash(frame, columns):
    """Local pandas-version-specific fingerprint of ordered rows and input values."""
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-targets", type=int, default=50_000)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/fusion/session18/evaluation_predictions.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation/session20")
    args = parser.parse_args()
    if min(args.train_targets, args.iterations, args.blocks, args.threads) < 1 or args.bootstrap_repeats < 2:
        parser.error("Limits/counts must be positive, with at least two bootstrap draws")
    start = time.perf_counter()
    full_train, validation = load_split("train"), load_split("validation")
    train = full_train.tail(args.train_targets).copy()
    evaluation = align_evaluation(validation, pd.read_csv(args.reference))
    if train.empty or not train.isFraud.isin([0, 1]).all() or train.isFraud.nunique() != 2:
        raise ValueError("Training selection needs both binary label classes")
    if train.TransactionDT.max() >= evaluation.TransactionDT.min() or train.TransactionID.isin(evaluation.TransactionID).any():
        raise ValueError("Training targets must strictly precede and be disjoint from evaluation")
    # Identical handling for each input feature across every model family variant.
    X_train = train[ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
    X_eval = evaluation[ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
    args.output.mkdir(parents=True, exist_ok=True)
    parameters = {"iterations": args.iterations, "depth": 6, "learning_rate": 0.05,
                  "loss_function": "Logloss", "random_seed": args.seed,
                  "thread_count": args.threads, "verbose": False, "allow_writing_files": False}
    scores = evaluation[["TransactionID", "TransactionDT", "isFraud"]].copy()
    train[["TransactionID", "TransactionDT", "isFraud"]].to_csv(args.output / "training_targets.csv", index=False)
    scores.to_csv(args.output / "evaluation_targets.csv", index=False)
    comparison = []
    for name, columns in GROUPS.items():
        model = CatBoostClassifier(**parameters)
        timer = time.perf_counter()
        # No eval_set/early stopping: validation does not select tree count.
        model.fit(X_train[columns], train.isFraud)
        fit_seconds = time.perf_counter() - timer
        timer = time.perf_counter()
        predictions = model.predict_proba(X_eval[columns])[:, 1]
        inference_seconds = time.perf_counter() - timer
        scores[name] = predictions
        model.save_model(str(args.output / f"{name}.cbm"))
        report = {"model": name, "features": ",".join(columns), "training_rows": len(train),
                  **metrics(evaluation.isFraud.to_numpy(), predictions, evaluation.TransactionID.to_numpy()),
                  "training_seconds": fit_seconds, "inference_seconds": inference_seconds}
        comparison.append(report)
        print(f"{name}: AP={report['average_precision']:.6f}, P@100={report['precision_at_100']:.3f}", flush=True)
        scores.to_csv(args.output / "predictions.csv", index=False)
        pd.DataFrame(comparison).to_csv(args.output / "model_comparison.csv", index=False)
    comparison = pd.DataFrame(comparison)
    baseline_ap = comparison.loc[comparison.model == "AmountOnly", "average_precision"].item()
    comparison["delta_ap_vs_amount_only"] = comparison.average_precision - baseline_ap
    comparison.to_csv(args.output / "model_comparison.csv", index=False)
    slices = chronological_blocks(evaluation.TransactionDT.to_numpy(), args.blocks)
    block_results = []
    for index, block_slice in enumerate(slices, 1):
        part = scores.iloc[block_slice]
        baseline = block_metrics(part.isFraud.to_numpy(), part.AmountOnly.to_numpy(), part.TransactionID.to_numpy())
        for name in GROUPS:
            report = block_metrics(part.isFraud.to_numpy(), part[name].to_numpy(), part.TransactionID.to_numpy())
            block_results.append({"block": index, "model": name, "frauds": int(part.isFraud.sum()),
                                  "start_time": float(part.TransactionDT.min()), "end_time": float(part.TransactionDT.max()),
                                  "delta_ap_vs_amount_only": report["average_precision"] - baseline["average_precision"]
                                  if report["average_precision"] is not None else None, **report})
    pd.DataFrame(block_results).to_csv(args.output / "chronological_blocks.csv", index=False)
    print(f"Paired bootstrap: {args.bootstrap_repeats} draws...", flush=True)
    draws, bootstrap = paired_ap_bootstrap(scores.isFraud.to_numpy(), scores.AmountOnly.to_numpy(),
                                          scores.AllFive.to_numpy(), args.bootstrap_repeats, args.seed)
    draws.to_csv(args.output / "bootstrap_draws.csv", index=False)
    write_json(args.output / "bootstrap_summary.json", bootstrap)
    write_json(args.output / "config.json", {
        **vars(args), "reference": str(args.reference), "output": str(args.output),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "catboost_version": catboost.__version__,
        "feature_groups": GROUPS, "shared_catboost_parameters": parameters,
        "preprocessing": "raw numerical features; infinities -> NaN; CatBoost native missing handling; no scaler",
        "selection": "latest training targets, exact Session 18 evaluation IDs",
        "training_input_hash": input_hash(train, ["TransactionID", "TransactionDT", "isFraud", *ALL_FEATURES]),
        "evaluation_input_hash": input_hash(evaluation, ["TransactionID", "TransactionDT", "isFraud", *ALL_FEATURES]),
        "reference_sha256": file_hash(args.reference),
        "full_training_history_rows": len(full_train), "training_rows": len(train),
        "training_frauds": int(train.isFraud.sum()), "evaluation_rows": len(evaluation),
        "evaluation_frauds": int(evaluation.isFraud.sum()), "actual_chronological_blocks": len(slices),
        "history": "Session 13 features from all strictly earlier observed inputs; online validation history",
        "test_access": False, "scope": "controlled development ablation; amount-only is not a rich current-only baseline",
    })
    write_json(args.output / "run_summary.json", {
        "status": "PASS", "completed_models": list(GROUPS), "failures": [],
        "runtime_seconds": time.perf_counter() - start, "test_access": False,
    })
    print("\nMATCHED-ROW ABLATION RESULTS\n" + comparison.drop(columns="features").to_string(index=False))
    print("\nPAIRED BOOTSTRAP\n" + str(bootstrap))
    print("\nCHRONOLOGICAL BLOCKS (AmountOnly and AllFive)\n" +
          pd.DataFrame(block_results).loc[lambda df: df.model.isin(["AmountOnly", "AllFive"])].to_string(index=False))
    print("\nRow bootstrap assumes independent resampling; these are reused development data, not a reserved-test result.")


if __name__ == "__main__":
    main()
