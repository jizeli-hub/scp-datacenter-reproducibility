"""Autocorrelation diagnostic used by the joint-SLA simulator."""

import numpy as np


def autocorrelation_effective_sample_size(
    residuals: np.ndarray,
    max_lag: int = 24,
) -> tuple[float, float, int, int]:
    """Estimate effective calibration size using only positive autocorrelations."""
    scores = np.asarray(residuals, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        raise ValueError("Effective sample-size estimation requires residuals.")
    if max_lag < 1 or n < 3:
        return float(n), 0.0, 0, 0

    lag_limit = min(int(max_lag), n - 2)

    centered = scores - float(np.mean(scores))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return float(n), 0.0, lag_limit, 0

    positive_sum = 0.0
    positive_lag_count = 0
    for lag in range(1, lag_limit + 1):
        numerator = float(np.dot(centered[:-lag], centered[lag:]))
        rho = numerator / denominator
        if np.isfinite(rho) and rho > 0.0:
            positive_sum += rho
            positive_lag_count += 1

    effective_n = n / (1.0 + 2.0 * positive_sum)
    effective_n = max(1.0, min(float(n), float(effective_n)))
    return effective_n, positive_sum, lag_limit, positive_lag_count
