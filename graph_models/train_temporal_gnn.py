"""Session 15: one-hop GraphSAGE and GAT on strictly historical star graphs."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GATConv

from deep_learning.behavioral_data import BASE_COLUMNS, prepare_data
from deep_learning.run_behavioral import metrics, seed_everything, write_json


def build_neighborhoods(frame, features, part):
    """Adapt tested sequence windows into independent history-to-target graphs.

    Node 0 is current; nodes 1..N are strictly earlier events. Inputs are scaled
    log amount and log1p(age in days relative to target). Labels enter y only.
    No inter-target edges are created, even if graphs reuse a historical event.
    """
    times = frame.TransactionDT.to_numpy(dtype=float)
    ids = frame.TransactionID.to_numpy()
    groups = frame.card1.to_numpy()
    labels = frame.isFraud.to_numpy(dtype=np.float32)
    graphs = []
    for window, length in zip(part["windows"], part["lengths"]):
        valid = window[:length]
        current = valid[-1]
        history = valid[:-1]
        if not (times[history] < times[current]).all():
            raise ValueError("Graph history must be strictly earlier than target")
        if len(history) and (pd.isna(groups[current]) or
                             not (groups[history] == groups[current]).all()):
            raise ValueError("Graph history must share a nonmissing card1 proxy")
        rows = np.r_[current, history]
        age_days = (times[current] - times[rows]) / 86400.0
        x = np.column_stack([features[rows, 0], np.log1p(age_days)]).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Node features must be finite")
        # PyG source_to_target flow: historical nodes send to current node 0.
        edge_index = torch.stack([
            torch.arange(1, len(rows), dtype=torch.long),
            torch.zeros(len(history), dtype=torch.long),
        ])
        graphs.append(Data(
            x=torch.from_numpy(x), edge_index=edge_index,
            y=torch.tensor([labels[current]], dtype=torch.float32),
            target_id=torch.tensor([ids[current]], dtype=torch.long),
            event_rows=torch.from_numpy(rows.astype(np.int64)),
            event_time=torch.from_numpy(times[rows].copy()),
        ))
    np.testing.assert_array_equal([g.target_id.item() for g in graphs], part["ids"])
    np.testing.assert_array_equal([g.y.item() for g in graphs], part["labels"])
    return graphs


class TemporalGNN(nn.Module):
    """One aggregation layer; choose GraphSAGE mean or two-head GAT."""

    def __init__(self, kind, hidden_size=32):
        super().__init__()
        if kind == "sage":
            self.conv = SAGEConv(2, hidden_size, aggr="mean")
        elif kind == "gat":
            if hidden_size % 2:
                raise ValueError("GAT hidden size must be divisible by two heads")
            self.conv = GATConv(2, hidden_size // 2, heads=2, concat=True,
                                add_self_loops=True, dropout=0.0)
        else:
            raise ValueError(f"Unknown architecture: {kind}")
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        hidden = torch.relu(self.conv(batch.x, batch.edge_index))
        # PyG batches disjoint graphs; ptr gives the first (target) node in each.
        target_embeddings = hidden[batch.ptr[:-1]]
        return self.classifier(target_embeddings).squeeze(-1)


def predict(model, loader):
    model.eval()
    scores, ids, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            scores.extend(torch.sigmoid(model(batch)).tolist())
            ids.extend(batch.target_id.tolist())
            labels.extend(batch.y.tolist())
    return np.asarray(scores), np.asarray(ids), np.asarray(labels)


def train_model(kind, train_graphs, val_graphs, args):
    seed_everything(args.seed)
    model = TemporalGNN(kind, args.hidden_size)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    log = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        scores, ids, labels = predict(model, val_loader)
        report = metrics(labels, scores, ids)
        log.append({"epoch": epoch, "training_loss": total_loss / len(train_graphs),
                    "validation_ap": report["average_precision"]})
        print(f"{kind.upper()} epoch {epoch}: loss={log[-1]['training_loss']:.5f}, "
              f"validation AP={report['average_precision']:.6f}", flush=True)
    training_seconds = time.perf_counter() - start
    # Fixed three-epoch endpoint, with no validation checkpoint selection.
    start = time.perf_counter()
    scores, ids, labels = predict(model, val_loader)
    inference_seconds = time.perf_counter() - start
    torch.save({"state_dict": model.state_dict(), "kind": kind,
                "hidden_size": args.hidden_size}, args.output / f"{kind}.pt")
    pd.DataFrame({"TransactionID": ids, "isFraud": labels, "model": kind,
                  "score": scores}).to_csv(args.output / f"{kind}_predictions.csv", index=False)
    write_json(args.output / f"{kind}_training.json", {"epochs": log})
    return {"model": kind, "train_examples": len(train_graphs),
            **metrics(labels, scores, ids), "training_seconds": training_seconds,
            "inference_seconds": inference_seconds, "epochs": args.epochs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-targets", type=int, default=6000)
    parser.add_argument("--val-targets", type=int, default=1500)
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/graph/session15")
    args = parser.parse_args()
    if min(args.train_targets, args.val_targets, args.max_events, args.batch_size,
           args.epochs, args.threads, args.hidden_size) < 1 or args.hidden_size % 2:
        parser.error("Limits must be positive and hidden size must be even")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train = pd.read_csv(ROOT / "data/processed/train.csv", usecols=BASE_COLUMNS)
    val = pd.read_csv(ROOT / "data/processed/validation.csv", usecols=BASE_COLUMNS)
    frame, features, parts, processor = prepare_data(
        train, val, args.train_targets, args.val_targets, args.max_events
    )
    write_json(args.output / "config.json", {
        **vars(args), "output": str(args.output), "torch": torch.__version__,
        "pyg": torch_geometric.__version__, "device": "cpu", "layers": 1,
        "gat_heads": 2, "dropout": 0.0, "loss": "BCEWithLogitsLoss",
        "positive_weight": None, "optimizer": "Adam", "checkpoint": "final epoch",
        "selection": "latest training targets; earliest validation targets",
        "history": "strictly earlier card1 proxy events, including earlier validation inputs",
        "features": ["training-standardized log1p amount", "log1p age in days relative to target"],
        "test_access": False,
    })
    write_json(args.output / "preprocessing.json", {
        "feature": "log1p amount", "fit_rows": len(train),
        "median": float(processor.median[0]), "mean": float(processor.mean[0]),
        "scale": float(processor.scale[0]), "age_transform": "log1p((targetDT - eventDT)/86400)",
        "missing_amount": "nonfinite/negative -> training median log amount",
    })
    datasets, statistics = {}, {"full_training_history_rows": len(train)}
    for split, part in parts.items():
        if len(np.unique(part["labels"])) != 2:
            raise ValueError(f"Selected {split} targets must contain both classes")
        datasets[split] = build_neighborhoods(frame, features, part)
        statistics[split] = {
            "targets": len(part["ids"]), "frauds": int(part["labels"].sum()),
            "prevalence": float(part["labels"].mean()),
            "mean_nodes": float(part["lengths"].mean()),
            "current_only_graphs": int((part["lengths"] == 1).sum()),
            "max_nodes": int(part["lengths"].max()),
        }
        pd.DataFrame({"TransactionID": part["ids"], "isFraud": part["labels"],
                      "nodes": part["lengths"]}).to_csv(args.output / f"{split}_targets.csv", index=False)
        np.savez_compressed(args.output / f"{split}_windows.npz", windows=part["windows"],
                            lengths=part["lengths"], ids=part["ids"], labels=part["labels"])
    write_json(args.output / "dataset_statistics.json", statistics)
    print(json.dumps(statistics, indent=2), flush=True)
    results = []
    for kind in ["sage", "gat"]:
        results.append(train_model(kind, datasets["train"], datasets["validation"], args))
        pd.DataFrame(results).to_csv(args.output / "model_comparison.csv", index=False)
    write_json(args.output / "run_summary.json", {
        "completed_models": [row["model"] for row in results], "failures": [],
        "runtime_seconds": time.perf_counter() - start, "test_access": False,
    })
    print("\nFINAL-EPOCH VALIDATION RESULTS")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nSmall one-hop proxy-history baseline; not a heterogeneous graph or a matched CatBoost comparison.")


if __name__ == "__main__":
    main()
