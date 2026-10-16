"""Audit saved alignment and reproduce validation scores without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from torch.utils.data import DataLoader

from deep_learning.behavioral_models import make_model
from deep_learning.run_behavioral import CATEGORIES, ROOT, metrics, predict, tensors, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/behavioral/session11")
    args = parser.parse_args()
    output = args.output
    config = json.loads((output / "config.json").read_text())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    features = np.load(output / "event_features.npy")
    assert np.isfinite(features).all()
    events = pd.read_csv(output / "history_events.csv")
    ids = events.TransactionID.to_numpy()
    times = events.TransactionDT.to_numpy()
    groups = events.card1.to_numpy()
    parts = {}
    for split in ["train", "validation"]:
        saved = np.load(output / f"{split}_windows.npz")
        part = {"windows": saved["indices"], "lengths": saved["lengths"],
                "ids": saved["ids"], "labels": saved["labels"]}
        targets = pd.read_csv(output / f"{split}_targets.csv")
        np.testing.assert_array_equal(targets.TransactionID, part["ids"])
        np.testing.assert_array_equal(targets.isFraud, part["labels"])
        windows, lengths = part["windows"], part["lengths"]
        assert ((lengths >= 1) & (lengths <= config["max_length"])).all()
        current = windows[np.arange(len(windows)), lengths - 1]
        np.testing.assert_array_equal(ids[current], part["ids"])
        np.testing.assert_array_equal(events.isFraud.to_numpy()[current], part["labels"])
        historical = np.arange(windows.shape[1])[None, :] < lengths[:, None] - 1
        valid = np.arange(windows.shape[1])[None, :] < lengths[:, None]
        assert (windows[~valid] == -1).all()
        safe = np.maximum(windows, 0)
        assert ((times[safe] < times[current, None]) | ~historical).all()
        assert ((groups[safe] == groups[current, None]) | ~historical).all()
        parts[split] = part
    comparisons = pd.read_csv(output / "model_comparison.csv")
    checks = []
    part = parts["validation"]
    for row in comparisons.to_dict(orient="records"):
        name = row["model"]
        saved_scores = pd.read_csv(output / f"{name}_predictions.csv")
        np.testing.assert_array_equal(saved_scores.TransactionID, part["ids"])
        np.testing.assert_array_equal(saved_scores.isFraud, part["labels"])
        assert saved_scores.model.eq(name).all()
        if name == "catboost_matched":
            model = CatBoostClassifier()
            model.load_model(str(output / f"{name}.cbm"))
            columns = ["TransactionAmt", "TransactionDT"] + CATEGORIES
            raw = pd.read_csv(ROOT / "data/processed/validation.csv",
                              usecols=["TransactionID"] + columns).set_index("TransactionID")
            raw = raw.loc[part["ids"], columns].copy()
            for column in CATEGORIES:
                raw[column] = raw[column].fillna("Missing").astype(str)
            raw[columns[:2]] = raw[columns[:2]].replace([np.inf, -np.inf], np.nan)
            scores = model.predict_proba(raw)[:, 1]
        else:
            checkpoint = torch.load(output / f"{name}.pt", map_location="cpu", weights_only=True)
            p = checkpoint["parameters"]
            model = make_model(p["kind"], p["input_size"], p["max_length"], p["hidden_size"])
            model.load_state_dict(checkpoint["state_dict"])
            dataset = tensors(features, part, p["max_length"], name == "current_mlp")
            loader = DataLoader(dataset, batch_size=p["batch_size"])
            scores = predict(model, p["kind"], loader)
        np.testing.assert_allclose(scores, saved_scores.score, rtol=1e-6, atol=1e-7)
        actual = metrics(part["labels"], scores, part["ids"])
        for metric in ["average_precision", "precision_at_100", "recall_at_100"]:
            np.testing.assert_allclose(actual[metric], row[metric], rtol=1e-8, atol=1e-10)
        checks.append({"model": name, "rows": len(scores), "status": "PASS",
                       "max_score_error": float(np.max(np.abs(scores - saved_scores.score)))})
        print(f"{name}: checkpoint, IDs, labels and metrics PASS", flush=True)
    write_json(output / "artifact_verification.json", {
        "status": "PASS", "sequence_contracts": "PASS", "test_access": False,
        "models": checks,
    })


if __name__ == "__main__":
    main()
