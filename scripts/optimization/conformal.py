"""One-sided conformal rank used by the joint-SLA simulator."""

import math

import numpy as np


def split_conformal_quantile(
    residuals: np.ndarray,
    alpha: float,
) -> tuple[float, int, bool]:
    """Return the conformal order statistic; the augmented rank maps to infinity."""
    scores = np.asarray(residuals, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("Conformal calibration requires finite residuals.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    raw_rank = (1.0 - alpha) * (len(scores) + 1)
    nearest_rank = round(raw_rank)
    rank = int(
        nearest_rank
        if math.isclose(raw_rank, nearest_rank, rel_tol=0.0, abs_tol=1e-12)
        else math.ceil(raw_rank)
    )
    if rank > len(scores):
        return float("inf"), rank, False
    rank = max(rank, 1)
    return float(np.sort(scores)[rank - 1]), rank, True
