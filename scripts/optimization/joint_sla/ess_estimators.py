"""Predeclared effective-sample diagnostics for dependent calibration scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..deployability import autocorrelation_effective_sample_size


@dataclass(frozen=True)
class EffectiveSizeEstimate:
    method: str
    nominal_n: int
    effective_n: float
    configured_max_lag: int
    actual_lag_limit: int
    dependence_factor: float


def _finite_vector(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("ESS scores must be one-dimensional.")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("ESS estimation requires finite scores.")
    return values


def _autocorrelations(values: np.ndarray, max_lag: int) -> tuple[np.ndarray, int]:
    n = len(values)
    if max_lag < 1 or n < 3:
        return np.empty(0), 0
    lag_limit = min(int(max_lag), n - 2)
    centered = values - float(np.mean(values))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return np.zeros(lag_limit), lag_limit
    rho = np.asarray(
        [
            float(np.dot(centered[:-lag], centered[lag:]) / denominator)
            for lag in range(1, lag_limit + 1)
        ]
    )
    return rho, lag_limit


def estimate_effective_size(
    scores: np.ndarray,
    *,
    method: str = "positive_autocorrelation_sum",
    max_lag: int = 24,
) -> EffectiveSizeEstimate:
    """Estimate diagnostic effective size using one frozen estimator."""
    values = _finite_vector(scores)
    n = len(values)
    if method == "positive_autocorrelation_sum":
        effective_n, positive_sum, lag_limit, _ = (
            autocorrelation_effective_sample_size(values, max_lag=max_lag)
        )
        factor = 1.0 + 2.0 * positive_sum
    elif method == "geyer_initial_positive_sequence":
        rho, lag_limit = _autocorrelations(values, max_lag)
        pair_sum = 0.0
        for index in range(0, len(rho) - 1, 2):
            pair = float(rho[index] + rho[index + 1])
            if not np.isfinite(pair) or pair <= 0.0:
                break
            pair_sum += pair
        factor = max(1.0, 1.0 + 2.0 * pair_sum)
        effective_n = float(np.clip(n / factor, 1.0, float(n)))
    elif method == "nonoverlapping_batch_means":
        lag_limit = 0
        batch_length = max(1, int(math.floor(math.sqrt(n))))
        batch_count = n // batch_length
        if batch_count < 2 or np.var(values, ddof=1) <= 0.0:
            factor = 1.0
            effective_n = float(n)
        else:
            trimmed = values[: batch_count * batch_length]
            means = trimmed.reshape(batch_count, batch_length).mean(axis=1)
            raw_variance = float(np.var(trimmed, ddof=1))
            batch_variance = float(np.var(means, ddof=1))
            factor = (
                batch_length * batch_variance / raw_variance
                if raw_variance > 0.0
                else 1.0
            )
            factor = max(factor, 1.0)
            effective_n = float(np.clip(n / factor, 1.0, float(n)))
    else:
        raise ValueError(f"Unknown ESS estimator: {method}")
    return EffectiveSizeEstimate(
        method=method,
        nominal_n=n,
        effective_n=float(effective_n),
        configured_max_lag=int(max_lag),
        actual_lag_limit=int(lag_limit),
        dependence_factor=float(factor),
    )
