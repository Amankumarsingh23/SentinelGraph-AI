"""Point-in-time windows for anonymized card1 groups, never verified customers."""

from dataclasses import dataclass
from collections import deque
import numpy as np
import pandas as pd


BASE_COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt", "card1", "isFraud"]
FEATURE_NAMES = [
    "log_amount", "relative_days", "amount_missing", "group_missing",
    "log_gap_seconds", "has_prior_history", "log_prior_count",
]
CURRENT_FEATURES = 4


@dataclass
class Preprocessor:
    """Median imputation and standardization fitted on training events only."""

    origin: float
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features, origin):
        median = np.array([
            np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0
            for column in features.T
        ])
        clean = np.where(np.isfinite(features), features, median)
        scale = clean.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(float(origin), median, clean.mean(axis=0), scale)

    def transform(self, features):
        clean = np.where(np.isfinite(features), features, self.median)
        return ((clean - self.mean) / self.scale).astype(np.float32)

    def metadata(self):
        return {"origin": self.origin, "features": FEATURE_NAMES,
                "median": self.median.tolist(), "mean": self.mean.tolist(),
                "scale": self.scale.tolist()}


def validate_frame(frame):
    if frame.empty or not frame.TransactionID.is_unique or frame.TransactionID.isna().any():
        raise ValueError("Need nonempty data with unique, nonmissing TransactionIDs")
    if not np.isfinite(frame.TransactionDT.to_numpy(dtype=float)).all():
        raise ValueError("TransactionDT must be finite")
    if not frame.isFraud.isin([0, 1]).all():
        raise ValueError("Labels must be binary")


def event_features_and_windows(frame, max_length, origin):
    """Process timestamp batches before updating history: peers cannot see peers.

    Rows must already be sorted by (TransactionDT, TransactionID). History keeps
    up to max_length-1 preceding events per group. Same-time historical peers are
    ordered by ID solely for deterministic truncation, not presumed arrival order.
    Labels are never read here. Missing group IDs always produce current-only rows.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    times = frame.TransactionDT.to_numpy(dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Need finite chronological times")
    n = len(frame)
    amount = pd.to_numeric(frame.TransactionAmt, errors="coerce").to_numpy(dtype=float)
    amount_valid = np.isfinite(amount) & (amount >= 0)
    groups = frame.card1.replace([np.inf, -np.inf], np.nan).to_numpy()
    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    features[:, 0] = np.log1p(np.where(amount_valid, amount, np.nan))
    features[:, 1] = (times - origin) / 86400.0
    features[:, 2] = ~amount_valid
    features[:, 3] = pd.isna(groups)
    features[:, 4] = np.nan
    # -1 is padding; lengths always include the current transaction.
    windows = np.full((n, max_length), -1, dtype=np.int32)
    lengths = np.ones(n, dtype=np.int64)
    history, counts, previous_times = {}, {}, {}
    boundaries = np.r_[0, np.flatnonzero(np.diff(times)) + 1, n]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        for i in range(start, end):
            group = groups[i]
            prior = [] if pd.isna(group) else list(history.get(group, ()))
            indices = prior + [i]
            windows[i, :len(indices)] = indices
            lengths[i] = len(indices)
            if not pd.isna(group):
                count = counts.get(group, 0)
                features[i, 5] = count > 0
                features[i, 6] = np.log1p(count)
                if count:
                    features[i, 4] = np.log1p(times[i] - previous_times[group])
        # Only after all predictions in this timestamp batch may it enter history.
        for i in range(start, end):
            group = groups[i]
            if pd.isna(group):
                continue
            if group not in history:
                history[group] = deque(maxlen=max_length - 1)
            history[group].append(i)
            counts[group] = counts.get(group, 0) + 1
            previous_times[group] = times[i]
    return features, windows, lengths


def truncate_windows(windows, lengths, max_length):
    """Keep the most recent real events, including the current one; right pad."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    result = np.full((len(windows), max_length), -1, dtype=np.int32)
    valid_lengths = np.minimum(lengths, max_length)
    for i, (old, new) in enumerate(zip(lengths, valid_lengths)):
        result[i, :new] = windows[i, old - new:old]
    return result, valid_lengths


def materialize(features, windows):
    """Padding stays exactly zero after preprocessing, not a synthetic event."""
    values = features[np.maximum(windows, 0)].copy()
    values[windows < 0] = 0
    return values


def prepare_data(train, validation, train_targets=100_000, val_targets=20_000,
                 max_length=20):
    validate_frame(train)
    validate_frame(validation)
    if train_targets < 1 or val_targets < 1:
        raise ValueError("Target limits must be positive")
    if train.TransactionDT.max() >= validation.TransactionDT.min():
        raise ValueError("Training must strictly precede validation")
    if train.TransactionID.isin(validation.TransactionID).any():
        raise ValueError("Split IDs must be disjoint")
    train = train.sort_values(["TransactionDT", "TransactionID"], kind="stable")
    validation = validation.sort_values(["TransactionDT", "TransactionID"], kind="stable")
    # Earliest validation targets, latest training targets; retain ALL train history.
    validation = validation.head(val_targets)
    frame = pd.concat([train, validation], ignore_index=True)
    n_train = len(train)
    raw, windows, lengths = event_features_and_windows(
        frame, max_length, float(train.TransactionDT.min())
    )
    processor = Preprocessor.fit(raw[:n_train], train.TransactionDT.min())
    features = processor.transform(raw)
    selection = {
        "train": np.arange(max(0, n_train - train_targets), n_train),
        "validation": np.arange(n_train, len(frame)),
    }
    parts = {}
    for name, rows in selection.items():
        parts[name] = {
            "windows": windows[rows], "lengths": lengths[rows],
            "ids": frame.TransactionID.to_numpy()[rows],
            "labels": frame.isFraud.to_numpy(dtype=np.float32)[rows],
            "rows": rows,
        }
    return frame, features, parts, processor
