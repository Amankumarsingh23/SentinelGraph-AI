"""Run Session 11 from the repo root: python -m deep_learning.run_behavioral."""

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier

from evaluate_risk import alert_ranking
from feature_engineering import FEATURES, historical_features
from deep_learning.behavioral_data import (
    BASE_COLUMNS, CURRENT_FEATURES, prepare_data, materialize, truncate_windows,
)
from deep_learning.behavioral_models import make_model, forward

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["ProductCD", "card4", "card6"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_phase1(data):
    """Audit train/validation repairs without opening the reserved test files."""
    hashes_path = ROOT / "outputs/phase1_before_repair/split_hashes.json"
    hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
    frames, stored = [], []
    checked_hashes = []
    for split in ["train", "validation"]:
        path = data / f"{split}.csv"
        key = f"data\\processed\\{split}.csv"
        if key in hashes:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != hashes[key]:
                raise ValueError(f"Established split changed: {split}")
            checked_hashes.append(split)
        raw = pd.read_csv(path, usecols=BASE_COLUMNS)[BASE_COLUMNS]
        repaired_path = data / f"{split}_features.csv"
        if not repaired_path.exists():
            raise ValueError(f"Phase 1 repair artifact missing: {repaired_path}")
        repaired = pd.read_csv(repaired_path, usecols=BASE_COLUMNS + FEATURES)
        pd.testing.assert_frame_equal(raw, repaired[BASE_COLUMNS], check_dtype=False)
        frames.append(raw)
        stored.append(repaired[FEATURES])
    combined = pd.concat(frames, ignore_index=True)
    actual = historical_features(combined)
    np.testing.assert_allclose(pd.concat(stored, ignore_index=True), actual,
                               equal_nan=True, rtol=1e-9, atol=1e-8)
    return {"status": "PASS", "rows_checked": len(combined),
            "split_hashes_checked": checked_hashes,
            "test_files_opened": False, "features_checked": FEATURES}


def metrics(labels, scores, ids):
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite model scores")
    selected = alert_ranking(scores, scores, ids)[:min(100, len(ids))]
    hits = int(labels[selected].sum())
    return {"average_precision": float(average_precision_score(labels, scores)),
            "precision_at_100": hits / len(selected),
            "recall_at_100": hits / float(labels.sum()) if labels.sum() else None,
            "validation_prevalence": float(labels.mean()), "val_examples": len(ids)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def tensors(features, part, length, current_only=False):
    windows, lengths = truncate_windows(part["windows"], part["lengths"], length)
    values = materialize(features, windows)
    if current_only:
        values = values[:, :, :CURRENT_FEATURES].copy()
    return TensorDataset(torch.from_numpy(values), torch.from_numpy(lengths),
                         torch.from_numpy(part["labels"][:, None]))


def predict(model, kind, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for x, lengths, _ in loader:
            scores.append(torch.sigmoid(forward(model, kind, x, lengths)).squeeze(1).numpy())
    return np.concatenate(scores)


def train_neural(name, kind, length, features, parts, args, current_only=False):
    seed_everything(args.seed)
    train_data = tensors(features, parts["train"], length, current_only)
    val_data = tensors(features, parts["validation"], length, current_only)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0)
    model = make_model(kind, train_data.tensors[0].shape[-1], length, args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Unweighted loss: scores are not claimed to be calibrated risk estimates.
    loss_fn = nn.BCEWithLogitsLoss()
    best_ap, best_state, stale, best_epoch = -1.0, None, 0, 0
    epochs = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for x, lengths, labels in train_loader:
            optimizer.zero_grad()
            logits = forward(model, kind, x, lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item() * len(x)
        scores = predict(model, kind, val_loader)
        ap = float(average_precision_score(parts["validation"]["labels"], scores))
        epochs.append({"epoch": epoch, "loss": loss_total / len(train_data), "val_ap": ap})
        print(f"{name} epoch {epoch}: loss={epochs[-1]['loss']:.5f} val_AP={ap:.6f}", flush=True)
        if ap > best_ap + 1e-6:
            best_ap, best_epoch = ap, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    start = time.perf_counter()
    scores = predict(model, kind, val_loader)
    inference_seconds = time.perf_counter() - start
    parameters = {
        "kind": kind, "input_size": train_data.tensors[0].shape[-1],
        "hidden_size": args.hidden_size, "max_length": length,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "optimizer": "Adam", "loss": "BCEWithLogitsLoss", "pos_weight": None,
        "gradient_clip_norm": 1.0, "seed": args.seed,
        "patience": args.patience, "max_epochs": args.epochs,
        "dropout": 0.0, "heads": 4 if kind == "transformer" else None,
        "encoder_layers": 1 if kind == "transformer" else None,
        "feedforward_size": 32 if kind == "transformer" else None,
    }
    torch.save({"state_dict": best_state, "parameters": parameters}, args.output / f"{name}.pt")
    write_json(args.output / f"{name}_training.json", {"parameters": parameters, "epochs": epochs})
    return scores, {"training_seconds": training_seconds, "inference_seconds": inference_seconds,
                    "best_epoch": best_epoch, "epochs_run": len(epochs),
                    "parameters": json.dumps(parameters, sort_keys=True)}


def train_catboost(train, val, args):
    columns = ["TransactionAmt", "TransactionDT"] + CATEGORIES
    train, val = train.copy(), val.copy()
    for frame in [train, val]:
        for column in CATEGORIES:
            frame[column] = frame[column].fillna("Missing").astype(str)
        frame[columns[:2]] = frame[columns[:2]].replace([np.inf, -np.inf], np.nan)
    parameters = {"iterations": 200, "depth": 6, "learning_rate": 0.05,
                  "loss_function": "Logloss", "random_seed": args.seed,
                  "thread_count": args.threads, "verbose": False, "allow_writing_files": False}
    model = CatBoostClassifier(**parameters)
    start = time.perf_counter()
    model.fit(train[columns], train.isFraud, cat_features=CATEGORIES)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    scores = model.predict_proba(val[columns])[:, 1]
    inference_seconds = time.perf_counter() - start
    model.save_model(str(args.output / "catboost_matched.cbm"))
    return scores, {"training_seconds": training_seconds, "inference_seconds": inference_seconds,
                    "best_epoch": None, "epochs_run": None,
                    "parameters": json.dumps({**parameters, "features": columns}, sort_keys=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-targets", type=int, default=100_000)
    parser.add_argument("--val-targets", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/behavioral/session11")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads, args.max_length) < 1:
        parser.error("Epochs, patience, batch size, threads and max length must be positive")
    if args.hidden_size < 4 or args.hidden_size % 4 or args.learning_rate <= 0:
        parser.error("Hidden size must be positive and divisible by 4; learning rate must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    config = {**vars(args), "output": str(args.output), "python": platform.python_version(),
              "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
              "platform": platform.platform(), "logical_cpus": os.cpu_count(), "device": "cpu",
              "selection": "latest training targets; earliest validation targets",
              "online_history": "earlier training and earlier validation inputs; never labels",
              "test_access": False}
    write_json(args.output / "config.json", config)
    data = ROOT / "data/processed"
    print("Verifying Phase 1 train/validation artifacts...", flush=True)
    audit = verify_phase1(data)
    write_json(args.output / "phase1_audit.json", audit)
    print(audit, flush=True)
    columns = BASE_COLUMNS + CATEGORIES
    train = pd.read_csv(data / "train.csv", usecols=columns)
    val = pd.read_csv(data / "validation.csv", usecols=columns)
    frame, features, parts, processor = prepare_data(
        train, val, args.train_targets, args.val_targets, args.max_length
    )
    write_json(args.output / "preprocessing.json", processor.metadata())
    stats = {"all_training_history_rows": len(train), "preprocessing_fit_rows": len(train),
             "all_validation_rows": len(val), "features": processor.metadata()["features"]}
    for name, part in parts.items():
        if len(np.unique(part["labels"])) != 2:
            raise ValueError(f"{name} selection must contain both classes")
        selected = frame.iloc[part["rows"]]
        stats[name] = {"examples": len(selected), "frauds": int(part["labels"].sum()),
                       "prevalence": float(part["labels"].mean()),
                       "start_time": float(selected.TransactionDT.min()),
                       "end_time": float(selected.TransactionDT.max()),
                       "mean_length": float(part["lengths"].mean()),
                       "current_only_rows": int((part["lengths"] == 1).sum()),
                       "at_max_length": int((part["lengths"] == args.max_length).sum()),
                       "length_counts": {str(k): int(v) for k, v in
                                         zip(*np.unique(part["lengths"], return_counts=True))}}
        pd.DataFrame({"TransactionID": part["ids"], "isFraud": part["labels"],
                      "length": part["lengths"]}).to_csv(args.output / f"{name}_targets.csv", index=False)
        np.savez_compressed(args.output / f"{name}_windows.npz", indices=part["windows"],
                            lengths=part["lengths"], ids=part["ids"], labels=part["labels"])
    # Store the event table order so window indices can be audited and reconstructed.
    frame[BASE_COLUMNS].to_csv(args.output / "history_events.csv", index=False)
    np.save(args.output / "event_features.npy", features)
    write_json(args.output / "dataset_statistics.json", stats)
    print(json.dumps(stats, indent=2), flush=True)
    results, failures = [], []
    val_part = parts["validation"]

    def record(name, scores, runtime):
        result = {"model": name, "train_examples": len(parts["train"]["ids"]),
                  **metrics(val_part["labels"], scores, val_part["ids"]), **runtime}
        results.append(result)
        pd.DataFrame({"TransactionID": val_part["ids"], "isFraud": val_part["labels"],
                      "model": name, "score": scores}).to_csv(args.output / f"{name}_predictions.csv", index=False)
        pd.DataFrame(results).to_csv(args.output / "model_comparison.csv", index=False)
        print(f"RESULT {name}: AP={result['average_precision']:.6f}", flush=True)

    experiments = [("current_mlp", "mlp", 1, True),
                   ("history_summary_mlp", "mlp", 1, False)]
    for kind in ["gru", "transformer"]:
        for length in sorted(set([min(5, args.max_length), args.max_length])):
            experiments.append((f"{kind}_{length}", kind, length, False))
    for name, kind, length, current_only in experiments:
        try:
            scores, runtime = train_neural(name, kind, length, features, parts, args, current_only)
            record(name, scores, runtime)
        except (RuntimeError, ValueError, MemoryError) as error:
            failures.append({"model": name, "error": repr(error)})
            write_json(args.output / "failures.json", failures)
            print(f"FAILED {name}: {error}", flush=True)
    try:
        scores, runtime = train_catboost(frame.iloc[parts["train"]["rows"]],
                                         frame.iloc[val_part["rows"]], args)
        record("catboost_matched", scores, runtime)
    except Exception as error:
        # CatBoost has its own exception type; preserve the exact failure report.
        failures.append({"model": "catboost_matched", "error": repr(error)})
        write_json(args.output / "failures.json", failures)
        print(f"FAILED catboost_matched: {error}", flush=True)
    historical = ROOT / "outputs/models/baseline_comparison.csv"
    if historical.exists():
        old = pd.read_csv(historical)
        write_json(args.output / "legacy_benchmark.json", {
            "source": str(historical.relative_to(ROOT)), "validation_rows": len(val),
            "matched": False, "results": old.to_dict(orient="records")})
    write_json(args.output / "run_summary.json", {
        "total_seconds": time.perf_counter() - start, "failures": failures,
        "completed_models": [row["model"] for row in results], "test_access": False,
        "timing": "training includes per-epoch validation; final inference measured separately"})
    print(pd.DataFrame(results).drop(columns="parameters").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
