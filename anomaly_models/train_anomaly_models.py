"""Session 17: unsupervised Isolation Forest and autoencoder anomaly rankings."""

import argparse
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from deep_learning.run_behavioral import metrics, seed_everything, write_json

PROCESSED = ROOT / "data/processed"
GRAPH = ROOT / "outputs/graph"
FEATURES = ["TransactionAmt", "graph_card_prior_degree", "graph_product_prior_degree",
            "graph_card_product_prior_count", "graph_card_distinct_products"]


def align_graph_features(raw, graph):
    """Join by ID; fail on missing matches, duplicates or stale graph timestamps."""
    for name, frame in [("raw", raw), ("graph", graph)]:
        if not frame.TransactionID.is_unique or frame.TransactionID.isna().any():
            raise ValueError(f"{name} TransactionIDs must be unique and nonmissing")
    merged = raw.merge(graph[["TransactionID", "TransactionDT", *FEATURES[1:]]],
                       on="TransactionID", how="left", validate="one_to_one",
                       suffixes=("", "_graph"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Every raw transaction needs an aligned graph-feature row")
    if not np.isfinite(merged.TransactionDT.to_numpy(dtype=float)).all():
        raise ValueError("TransactionDT must be finite")
    if not merged.TransactionDT.eq(merged.TransactionDT_graph).all():
        raise ValueError("Graph-feature timestamps differ from original transactions")
    return merged.drop(columns=["_merge", "TransactionDT_graph"]).sort_values(
        ["TransactionDT", "TransactionID"], kind="stable"
    ).reset_index(drop=True)


def load_split(split):
    if split not in ["train", "validation"]:
        raise ValueError("This experiment permits train and validation only")
    raw = pd.read_csv(PROCESSED / f"{split}.csv", usecols=[
        "TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
    ])
    graph = pd.read_csv(GRAPH / f"{split}_graph_features.csv", usecols=[
        "TransactionID", "TransactionDT", *FEATURES[1:],
    ])
    return align_graph_features(raw, graph)


def raw_features(frame):
    """Exactly five inputs; labels/IDs are never included or used for filtering."""
    values = frame[FEATURES].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.maximum(values, 0.0))


class Autoencoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=16, latent_size=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                     nn.Linear(hidden_size, latent_size))
        self.decoder = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(),
                                     nn.Linear(hidden_size, input_size))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_error(original, reconstruction):
    return (reconstruction - original).square().mean(dim=1)


def autoencoder_scores(model, values, batch_size=256):
    model.eval()
    scores = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size):
            scores.append(reconstruction_error(batch, model(batch)).numpy())
    return np.concatenate(scores)


def fit_autoencoder(values, args):
    """Fit only X -> X reconstruction. No labels or validation input accepted."""
    seed_everything(args.seed)
    model = Autoencoder(hidden_size=args.hidden_size, latent_size=args.latent_size)
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        log.append({"epoch": epoch, "training_mse": total_loss / len(values)})
        print(f"Autoencoder epoch {epoch}: training MSE={log[-1]['training_mse']:.6f}", flush=True)
    return model, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-targets", type=int, default=50_000)
    parser.add_argument("--val-targets", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--latent-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/anomaly/session17")
    args = parser.parse_args()
    if min(args.train_targets, args.val_targets, args.epochs, args.batch_size,
           args.hidden_size, args.latent_size, args.trees, args.threads) < 1:
        parser.error("Row limits, dimensions, epochs, trees, threads and batch size must be positive")
    if args.latent_size >= len(FEATURES) or args.learning_rate <= 0:
        parser.error("Latent dimension must be less than five; learning rate must be positive")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train = load_split("train").tail(args.train_targets).copy()
    val = load_split("validation").head(args.val_targets).copy()
    if train.empty or val.empty or train.TransactionDT.max() >= val.TransactionDT.min():
        raise ValueError("Need nonempty, strictly time-separated train and validation selections")
    if train.TransactionID.isin(val.TransactionID).any():
        raise ValueError("Train and validation IDs must be disjoint")
    for split, frame in [("train", train), ("validation", val)]:
        if not frame.isFraud.isin([0, 1]).all():
            raise ValueError(f"{split} evaluation labels must be binary")
        if split == "validation" and frame.isFraud.nunique() != 2:
            raise ValueError("Validation metrics require both label classes")
        frame[["TransactionID", "TransactionDT", "isFraud"]].to_csv(
            args.output / f"{split}_targets.csv", index=False
        )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(raw_features(train)).astype(np.float32)
    X_val = scaler.transform(raw_features(val)).astype(np.float32)
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Preprocessed features must be finite")
    joblib.dump(scaler, args.output / "scaler.joblib")
    write_json(args.output / "preprocessing.json", {
        "features": FEATURES, "fit_rows": len(train), "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(), "variance": scaler.var_.tolist(),
        "missing_policy": "NaN/+inf/-inf -> zero; clip negatives to zero; then log1p",
        "scaling": "fit StandardScaler on selected training rows only; frozen on validation",
    })
    write_json(args.output / "config.json", {
        **vars(args), "output": str(args.output), "features": FEATURES,
        "python": platform.python_version(), "torch": torch.__version__,
        "scikit_learn": sklearn.__version__, "device": "cpu",
        "label_use": "evaluation only; no normal-only selection or model fitting labels",
        "selection": "latest training targets; earliest validation targets",
        "history": "saved Session 13 features; all earlier history retained before target selection",
        "isolation_forest": {"n_estimators": args.trees, "max_samples": min(256, len(train)),
                             "contamination": "auto", "n_jobs": args.threads, "random_state": args.seed},
        "autoencoder": {"architecture": [5, args.hidden_size, args.latent_size, args.hidden_size, 5],
                        "optimizer": "Adam", "loss": "MSELoss", "checkpoint": "final epoch"},
        "random_reference": "NumPy default_rng(seed).random(validation rows)",
        "test_access": False,
    })
    stats = {}
    for split, frame in [("train", train), ("validation", val)]:
        stats[split] = {"rows": len(frame), "frauds": int(frame.isFraud.sum()),
                        "prevalence": float(frame.isFraud.mean()),
                        "start_time": float(frame.TransactionDT.min()),
                        "end_time": float(frame.TransactionDT.max())}
    write_json(args.output / "dataset_statistics.json", stats)
    print(stats, flush=True)
    labels, ids = val.isFraud.to_numpy(), val.TransactionID.to_numpy()
    results = []

    def record(name, scores, training_seconds, inference_seconds):
        results.append({"model": name, "train_examples": len(train) if name != "random" else 0,
                        **metrics(labels, scores, ids), "training_seconds": training_seconds,
                        "inference_seconds": inference_seconds})
        pd.DataFrame({"TransactionID": ids, "isFraud": labels, "model": name,
                      "anomaly_score": scores}).to_csv(args.output / f"{name}_predictions.csv", index=False)
        pd.DataFrame(results).to_csv(args.output / "model_comparison.csv", index=False)

    forest = IsolationForest(n_estimators=args.trees, max_samples=min(256, len(train)),
                             contamination="auto", random_state=args.seed, n_jobs=args.threads)
    timer = time.perf_counter()
    forest.fit(X_train)  # No fraud labels; all selected training rows are used.
    forest_seconds = time.perf_counter() - timer
    timer = time.perf_counter()
    scores = -forest.score_samples(X_val)  # Higher means more anomalous.
    forest_inference = time.perf_counter() - timer
    record("isolation_forest", scores, forest_seconds, forest_inference)
    joblib.dump(forest, args.output / "isolation_forest.joblib")

    timer = time.perf_counter()
    autoencoder, log = fit_autoencoder(X_train, args)
    ae_seconds = time.perf_counter() - timer
    timer = time.perf_counter()
    scores = autoencoder_scores(autoencoder, X_val, args.batch_size)
    record("autoencoder", scores, ae_seconds, time.perf_counter() - timer)
    torch.save({"state_dict": autoencoder.state_dict(), "input_size": 5,
                "hidden_size": args.hidden_size, "latent_size": args.latent_size}, args.output / "autoencoder.pt")
    write_json(args.output / "autoencoder_training.json", {"epochs": log})

    timer = time.perf_counter()
    scores = np.random.default_rng(args.seed).random(len(val))
    record("random", scores, 0.0, time.perf_counter() - timer)
    write_json(args.output / "run_summary.json", {
        "completed_models": [row["model"] for row in results], "failures": [],
        "total_seconds": time.perf_counter() - start, "test_access": False,
        "scores_are_fraud_probabilities": False,
    })
    print("\nVALIDATION RANKING RESULTS")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nAnomaly scores are not calibrated fraud probabilities. Random scores are one seeded realization.")
    print("No matched CatBoost or fusion experiment is performed in this session.")


if __name__ == "__main__":
    main()
