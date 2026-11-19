"""Direct max-score conformal provisioning for aligned resource vectors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..conformal import split_conformal_quantile
from ..deployability import autocorrelation_effective_sample_size

from .definitions import validate_alpha
from .joint_block_stability import joint_score_block_stability
from .risk_allocation import normalized_mismatch_metrics


@dataclass(frozen=True)
class DirectJointDecision:
    """Calibration-only decision for the scalar direct-joint comparator."""

    status: str
    reason_code: str
    threshold: float | None
    scales: tuple[float, ...]
    effective_size: float
    worst_block_coverage: float | None
    calibration_cost: float | None
    calibration_joint_violation: float | None


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def proper_training_scales(training_demand: np.ndarray) -> np.ndarray:
    """Estimate strictly positive resource scales from proper training only."""
    demand = _matrix(training_demand, "training_demand")
    scales = np.mean(np.abs(demand), axis=0)
    if np.any(scales <= 1e-12):
        raise ValueError("Every resource requires a positive proper-training scale.")
    return scales


def joint_nonconformity_scores(
    residual_vectors: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Return max normalized one-sided residuals for aligned resource rows."""
    residuals = _matrix(residual_vectors, "residual_vectors")
    scale = np.asarray(scales, dtype=float)
    if scale.ndim != 1 or len(scale) != residuals.shape[1]:
        raise ValueError("One scale is required for each residual resource.")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Joint-score scales must be finite and positive.")
    return np.max(residuals / scale, axis=1)


def direct_joint_threshold(
    scores: np.ndarray,
    alpha_joint: float,
) -> tuple[float, int, bool]:
    """Return the scalar split-conformal threshold and finite-rank state."""
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a nonempty finite one-dimensional array.")
    return split_conformal_quantile(values, alpha)


def direct_joint_capacities(
    point_forecasts: np.ndarray,
    scales: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert one scalar joint threshold into resource capacities."""
    point = _matrix(point_forecasts, "point_forecasts")
    scale = np.asarray(scales, dtype=float)
    if scale.shape != (point.shape[1],) or np.any(scale <= 0.0):
        raise ValueError("One positive scale is required per point-forecast resource.")
    if not np.isfinite(threshold):
        raise ValueError("A finite direct-joint threshold is required for capacity.")
    return np.maximum(0.0, point + float(threshold) * scale)


def assess_direct_joint(
    *,
    calibration_demand: np.ndarray,
    calibration_point: np.ndarray,
    scales: np.ndarray,
    alpha_joint: float,
    cost_weights: np.ndarray,
    c_u: float,
    c_o: float,
    max_lag: int,
    n_blocks: int,
    tolerance: float,
) -> DirectJointDecision:
    """Apply rank, ESS, and block diagnostics to direct joint conformal."""
    demand = _matrix(calibration_demand, "calibration_demand")
    point = _matrix(calibration_point, "calibration_point")
    if demand.shape != point.shape:
        raise ValueError("Calibration demand and point forecasts must align.")
    scale = np.asarray(scales, dtype=float)
    residuals = demand - point
    scores = joint_nonconformity_scores(residuals, scale)
    threshold, _, finite = direct_joint_threshold(scores, alpha_joint)
    effective_size = autocorrelation_effective_sample_size(scores, max_lag)[0]
    required = math.ceil(1.0 / float(alpha_joint) - 1e-12) - 1
    common = {
        "scales": tuple(float(value) for value in scale),
        "effective_size": float(effective_size),
    }
    if not finite:
        return DirectJointDecision(
            status="DEFER",
            reason_code="JOINT_RANK_INSUFFICIENT",
            threshold=None,
            worst_block_coverage=None,
            calibration_cost=None,
            calibration_joint_violation=None,
            **common,
        )
    if math.floor(effective_size) < required:
        return DirectJointDecision(
            status="DEFER",
            reason_code="JOINT_ESS_INSUFFICIENT",
            threshold=None,
            worst_block_coverage=None,
            calibration_cost=None,
            calibration_joint_violation=None,
            **common,
        )
    stable, worst = joint_score_block_stability(
        scores,
        alpha_joint,
        n_blocks=n_blocks,
        tolerance=tolerance,
    )
    if not stable:
        return DirectJointDecision(
            status="MONITOR",
            reason_code="JOINT_BLOCK_UNSTABLE",
            threshold=None,
            worst_block_coverage=worst,
            calibration_cost=None,
            calibration_joint_violation=None,
            **common,
        )
    capacity = direct_joint_capacities(point, scale, threshold)
    cost, joint_violation, _, _ = normalized_mismatch_metrics(
        demand,
        capacity,
        scale,
        np.asarray(cost_weights, dtype=float),
        c_u,
        c_o,
    )
    return DirectJointDecision(
        status="ADMIT",
        reason_code="JOINT_SCREEN_PASS",
        threshold=float(threshold),
        worst_block_coverage=worst,
        calibration_cost=cost,
        calibration_joint_violation=joint_violation,
        **common,
    )
