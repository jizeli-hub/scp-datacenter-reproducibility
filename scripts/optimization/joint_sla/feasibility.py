"""Finite-rank constructibility for resource-wise and direct joint conformal."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .definitions import RankFrontier, validate_alpha


NUMERIC_TOLERANCE = 1e-12


def validate_calibration_sizes(sizes: Iterable[int]) -> tuple[int, ...]:
    """Validate nonnegative integer calibration counts."""
    values = tuple(sizes)
    if not values:
        raise ValueError("At least one resource calibration size is required.")
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in values):
        raise ValueError("Calibration sizes must be nonnegative integers.")
    return tuple(int(value) for value in values)


def joint_rank_frontier(
    calibration_sizes: Iterable[int],
    alpha_joint: float,
) -> RankFrontier:
    """Evaluate Proposition 1 for the resource-wise Bonferroni construction."""
    sizes = validate_calibration_sizes(calibration_sizes)
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    alpha_min = float(sum(1.0 / (size + 1) for size in sizes))
    return RankFrontier(
        calibration_sizes=sizes,
        alpha_joint=alpha,
        alpha_min=alpha_min,
        beta_max=max(0.0, 1.0 - alpha_min),
        constructible=bool(alpha_min <= alpha + NUMERIC_TOLERANCE),
    )


def equal_history_required(resource_count: int, alpha_joint: float) -> int:
    """Return the equal per-resource history required by Proposition 1."""
    if isinstance(resource_count, bool) or resource_count < 1:
        raise ValueError("resource_count must be a positive integer.")
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    return max(0, int(math.ceil(resource_count / alpha - NUMERIC_TOLERANCE)) - 1)


def direct_joint_constructible(n_paired: int, alpha_joint: float) -> bool:
    """Check the scalar direct-joint split-conformal rank condition."""
    sizes = validate_calibration_sizes((n_paired,))
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    return bool(sizes[0] >= math.ceil(1.0 / alpha - NUMERIC_TOLERANCE) - 1)
