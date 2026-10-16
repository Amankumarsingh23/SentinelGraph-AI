"""Pure fixed-reference PSI helpers; importing this module never reads files."""

from numbers import Integral

import numpy as np
import pandas as pd

EPS = 1e-6


def numeric_values(values):
    if np.ndim(values) != 1:
        raise ValueError("Expected one-dimensional numerical observations")
    values = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
    if not len(values):
        raise ValueError("Monitoring windows cannot be empty")
    return values


def validate_edges(edges):
    edges = np.asarray(edges, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or edges[0] != -np.inf or edges[-1] != np.inf
            or np.isnan(edges).any() or not np.all(edges[1:] > edges[:-1])):
        raise ValueError("Edges must be strictly increasing from -inf to +inf")
    return edges


def reference_edges(reference, n_bins=10):
    """Reference-only linear quantiles; collapse ties and keep unbounded tails.

    A constant reference gets a narrow bin containing its exact value, so a
    newly different constant is detectable. An all-missing reference gets one
    numerical bin plus the always-present missing/nonfinite category.
    """
    if not isinstance(n_bins, Integral) or isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("Need at least two nominal quantile bins")
    values = numeric_values(reference)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.array([-np.inf, np.inf])
    if finite.min() == finite.max():
        value = finite[0]
        with np.errstate(over="ignore"):
            inner = [value, np.nextafter(value, np.inf)]
    else:
        inner = np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1], method="linear")
    return validate_edges(np.unique(np.r_[-np.inf, inner, np.inf]))


def bin_counts(values, edges):
    values, edges = numeric_values(values), validate_edges(edges)
    finite = np.isfinite(values)
    counts = np.r_[np.histogram(values[finite], bins=edges)[0], (~finite).sum()]
    if counts.sum() != len(values):
        raise ValueError("Every observation must belong to exactly one category")
    return counts


def smoothed_proportions(values, edges, epsilon=EPS):
    """Add epsilon to each raw proportion, then normalize, including missing."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Smoothing epsilon must be finite and positive")
    counts = bin_counts(values, edges)
    proportions = counts / counts.sum()
    return (proportions + epsilon) / (1 + len(proportions) * epsilon)


def calculate_psi(reference, current, edges, epsilon=EPS):
    expected = smoothed_proportions(reference, edges, epsilon)
    actual = smoothed_proportions(current, edges, epsilon)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def distribution_summary(values):
    values = numeric_values(values)
    finite = values[np.isfinite(values)]
    return {"rows": len(values), "finite_count": len(finite),
            "missing_count": int(np.isnan(values).sum()), "infinite_count": int(np.isinf(values).sum()),
            "missing_rate": float(np.isnan(values).mean()),
            "missing_or_nonfinite_rate": float((~np.isfinite(values)).mean()),
            "negative_count": int((finite < 0).sum()),
            "mean": float(finite.mean()) if len(finite) else None,
            "min": float(finite.min()) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "p95": float(np.quantile(finite, .95)) if len(finite) else None,
            "p99": float(np.quantile(finite, .99)) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None}


def encode_edges(edges):
    return ["-inf" if value == -np.inf else "+inf" if value == np.inf else float(value)
            for value in validate_edges(edges)]


def decode_edges(edges):
    return validate_edges([float(value) for value in edges])
