"""Evidence-Constrained Risk Allocation for Joint SLAs (ECRA-JSLA)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..conformal import split_conformal_quantile
from ..deployability import autocorrelation_effective_sample_size

from .definitions import AllocationDecision, validate_alpha
from .feasibility import NUMERIC_TOLERANCE, validate_calibration_sizes
from .joint_block_stability import (
    resourcewise_block_stability,
    resourcewise_block_stability_many,
)
from .ess_estimators import estimate_effective_size


def attainable_allocations(
    calibration_sizes: Iterable[int],
    alpha_joint: float,
) -> list[tuple[float, ...]]:
    """Enumerate conservative resource budgets induced by attainable ranks."""
    sizes = validate_calibration_sizes(calibration_sizes)
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    per_resource = [
        [index / (size + 1) for index in range(1, size + 1) if index / (size + 1) <= alpha + NUMERIC_TOLERANCE]
        for size in sizes
    ]
    if any(not values for values in per_resource):
        return []
    allocations = [
        tuple(float(value) for value in candidate)
        for candidate in itertools.product(*per_resource)
        if sum(candidate) <= alpha + NUMERIC_TOLERANCE
    ]
    return sorted(allocations)


def normalized_mismatch_metrics(
    demand: np.ndarray,
    capacity: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    c_u: float,
    c_o: float,
) -> tuple[float, float, float, float]:
    """Return normalized cost, joint violation, max resource violation, capacity."""
    shortfall = np.maximum(0.0, demand - capacity)
    excess = np.maximum(0.0, capacity - demand)
    per_resource_cost = np.mean(c_u * shortfall + c_o * excess, axis=0) / scales
    cost = float(np.dot(weights, per_resource_cost))
    resource_violation = np.mean(demand > capacity, axis=0)
    joint_violation = float(np.mean(np.any(demand > capacity, axis=1)))
    normalized_capacity = float(np.sum(np.mean(capacity, axis=0) / scales))
    return cost, joint_violation, float(np.max(resource_violation)), normalized_capacity


def allocation_cost_curve(
    *,
    calibration_demand: np.ndarray,
    calibration_point: np.ndarray,
    proper_training_scales: np.ndarray,
    alpha_joint: float,
    cost_weights: Iterable[float],
    c_u: float,
    c_o: float,
    max_lag: int,
    n_blocks: int,
    tolerance: float,
    ess_method: str = "positive_autocorrelation_sum",
) -> pd.DataFrame:
    """Enumerate calibration-only costs and diagnostics for attainable budgets."""
    demand = np.asarray(calibration_demand, dtype=float)
    point = np.asarray(calibration_point, dtype=float)
    if demand.shape != point.shape or demand.ndim != 2:
        raise ValueError("Calibration demand and point forecasts must align.")
    scales = np.asarray(proper_training_scales, dtype=float)
    weights = np.asarray(tuple(cost_weights), dtype=float)
    if scales.shape != (demand.shape[1],) or np.any(scales <= 0.0):
        raise ValueError("One positive proper-training scale is required per resource.")
    if weights.shape != scales.shape or np.any(weights <= 0.0):
        raise ValueError("One positive cost weight is required per resource.")
    weights = weights / weights.sum()
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    residuals = demand - point
    effective_sizes = tuple(
        estimate_effective_size(
            residuals[:, index], method=ess_method, max_lag=max_lag
        ).effective_n
        for index in range(residuals.shape[1])
    )
    allocations = attainable_allocations(
        tuple(len(residuals) for _ in range(residuals.shape[1])), alpha
    )
    diagnostics = resourcewise_block_stability_many(
        residuals,
        allocations,
        alpha,
        n_blocks=n_blocks,
        tolerance=tolerance,
    )
    rows: list[dict[str, object]] = []
    for candidate, diagnostic in zip(allocations, diagnostics, strict=True):
        thresholds = tuple(
            float(split_conformal_quantile(residuals[:, index], budget)[0])
            for index, budget in enumerate(candidate)
        )
        capacity = np.maximum(0.0, point + np.asarray(thresholds))
        shortfall = np.maximum(0.0, demand - capacity)
        excess = np.maximum(0.0, capacity - demand)
        resource_costs = (
            np.mean(c_u * shortfall + c_o * excess, axis=0) / scales
        )
        cost, joint_violation, max_violation, normalized_capacity = (
            normalized_mismatch_metrics(
                demand, capacity, scales, weights, c_u, c_o
            )
        )
        ess_supported = all(
            math.floor(n_eff)
            >= math.ceil(1.0 / budget - NUMERIC_TOLERANCE) - 1
            for n_eff, budget in zip(effective_sizes, candidate, strict=True)
        )
        rows.append(
            {
                "alpha_cpu": candidate[0],
                "alpha_memory": candidate[1],
                "safety_stock_cpu": thresholds[0],
                "safety_stock_memory": thresholds[1],
                "calibration_normalized_cost": cost,
                "calibration_cpu_normalized_cost": float(resource_costs[0]),
                "calibration_memory_normalized_cost": float(resource_costs[1]),
                "calibration_joint_violation": joint_violation,
                "calibration_max_resource_violation": max_violation,
                "calibration_normalized_capacity": normalized_capacity,
                "n_eff_cpu": effective_sizes[0],
                "n_eff_memory": effective_sizes[1],
                "ess_supported": ess_supported,
                "resource_stable": diagnostic.resource_stable,
                "joint_stable": diagnostic.joint_stable,
                "cpu_worst_block_coverage": diagnostic.resource_worst_coverage[0],
                "memory_worst_block_coverage": diagnostic.resource_worst_coverage[1],
                "joint_worst_block_coverage": diagnostic.joint_worst_coverage,
            }
        )
    return pd.DataFrame(rows)


def select_ecra_from_curve(
    curve: pd.DataFrame,
    *,
    cost_weights: Iterable[float],
) -> AllocationDecision:
    """Select ECRA from one calibration-only candidate table.

    Candidate ranks, ESS values, and block diagnostics are independent of the
    operational resource weights. Reusing the frozen candidate table avoids
    recomputing those diagnostics in the predeclared weight sensitivity.
    """
    weights = np.asarray(tuple(cost_weights), dtype=float)
    if weights.shape != (2,) or np.any(weights <= 0.0):
        raise ValueError("Two positive operational weights are required.")
    weights = weights / weights.sum()
    if curve.empty:
        return AllocationDecision(
            "DEFER", "JOINT_RANK_INSUFFICIENT", None, None,
            (), float("nan"), None, None, (), None,
        )
    effective_sizes = (
        float(curve.iloc[0]["n_eff_cpu"]),
        float(curve.iloc[0]["n_eff_memory"]),
    )
    alpha_min_ess = float(
        sum(1.0 / (math.floor(value) + 1) for value in effective_sizes)
    )
    ess = curve.loc[curve["ess_supported"]].copy()
    if ess.empty:
        return AllocationDecision(
            "DEFER", "JOINT_ESS_INSUFFICIENT", None, None,
            effective_sizes, alpha_min_ess, None, None, (), None,
        )
    resource = ess.loc[ess["resource_stable"]].copy()
    if resource.empty:
        row = ess.iloc[0]
        return AllocationDecision(
            "MONITOR", "RESOURCE_BLOCK_UNSTABLE", None, None,
            effective_sizes, alpha_min_ess, None, None,
            (float(row["cpu_worst_block_coverage"]), float(row["memory_worst_block_coverage"])),
            float(row["joint_worst_block_coverage"]),
        )
    stable = resource.loc[resource["joint_stable"]].copy()
    if stable.empty:
        row = resource.iloc[0]
        return AllocationDecision(
            "MONITOR", "JOINT_BLOCK_UNSTABLE", None, None,
            effective_sizes, alpha_min_ess, None, None,
            (float(row["cpu_worst_block_coverage"]), float(row["memory_worst_block_coverage"])),
            float(row["joint_worst_block_coverage"]),
        )
    stable["weighted_cost"] = (
        weights[0] * stable["calibration_cpu_normalized_cost"]
        + weights[1] * stable["calibration_memory_normalized_cost"]
    )
    selected = stable.sort_values(
        [
            "weighted_cost",
            "calibration_joint_violation",
            "calibration_max_resource_violation",
            "calibration_normalized_capacity",
            "alpha_cpu",
            "alpha_memory",
        ],
        kind="stable",
    ).iloc[0]
    return AllocationDecision(
        "ADMIT",
        "JOINT_SCREEN_PASS",
        (float(selected["alpha_cpu"]), float(selected["alpha_memory"])),
        (float(selected["safety_stock_cpu"]), float(selected["safety_stock_memory"])),
        effective_sizes,
        alpha_min_ess,
        float(selected["weighted_cost"]),
        float(selected["calibration_joint_violation"]),
        (float(selected["cpu_worst_block_coverage"]), float(selected["memory_worst_block_coverage"])),
        float(selected["joint_worst_block_coverage"]),
    )


def select_ecra_ablation_stage(
    *,
    calibration_demand: np.ndarray,
    calibration_point: np.ndarray,
    proper_training_scales: np.ndarray,
    alpha_joint: float,
    cost_weights: Iterable[float],
    c_u: float,
    c_o: float,
    max_lag: int,
    n_blocks: int,
    tolerance: float,
    stage: int,
) -> AllocationDecision:
    """Select one calibration-only allocation for a nested admission ablation.

    Stages 0--3 add finite rank, resource ESS, resource block stability, and
    joint block stability. They choose the surviving allocation closest to an
    equal split so the admission components are isolated. Stage 4 applies the
    frozen calibration-cost objective and tie-breakers to the fully admitted
    candidate set.
    """
    if stage not in range(5):
        raise ValueError("stage must be an integer from zero through four.")
    curve = allocation_cost_curve(
        calibration_demand=calibration_demand,
        calibration_point=calibration_point,
        proper_training_scales=proper_training_scales,
        alpha_joint=alpha_joint,
        cost_weights=cost_weights,
        c_u=c_u,
        c_o=c_o,
        max_lag=max_lag,
        n_blocks=n_blocks,
        tolerance=tolerance,
    )
    effective_sizes = (
        tuple(float(curve.iloc[0][column]) for column in ("n_eff_cpu", "n_eff_memory"))
        if not curve.empty
        else tuple(
            autocorrelation_effective_sample_size(
                np.asarray(calibration_demand)[:, index]
                - np.asarray(calibration_point)[:, index],
                max_lag,
            )[0]
            for index in range(np.asarray(calibration_demand).shape[1])
        )
    )
    alpha_min_ess = float(
        sum(1.0 / (math.floor(value) + 1) for value in effective_sizes)
    )
    if curve.empty:
        return AllocationDecision(
            "DEFER",
            "JOINT_RANK_INSUFFICIENT",
            None,
            None,
            effective_sizes,
            alpha_min_ess,
            None,
            None,
            (),
            None,
        )
    candidates = curve.copy()
    if stage >= 1:
        candidates = candidates.loc[candidates["ess_supported"]]
        if candidates.empty:
            return AllocationDecision(
                "DEFER",
                "JOINT_ESS_INSUFFICIENT",
                None,
                None,
                effective_sizes,
                alpha_min_ess,
                None,
                None,
                (),
                None,
            )
    if stage >= 2:
        candidates = candidates.loc[candidates["resource_stable"]]
        if candidates.empty:
            return AllocationDecision(
                "MONITOR",
                "RESOURCE_BLOCK_UNSTABLE",
                None,
                None,
                effective_sizes,
                alpha_min_ess,
                None,
                None,
                (),
                None,
            )
    if stage >= 3:
        candidates = candidates.loc[candidates["joint_stable"]]
        if candidates.empty:
            return AllocationDecision(
                "MONITOR",
                "JOINT_BLOCK_UNSTABLE",
                None,
                None,
                effective_sizes,
                alpha_min_ess,
                None,
                None,
                (),
                None,
            )

    if stage < 4:
        candidates = candidates.assign(
            equal_distance=np.abs(
                candidates["alpha_cpu"] - float(alpha_joint) / 2.0
            )
            + np.abs(candidates["alpha_memory"] - float(alpha_joint) / 2.0)
        ).sort_values(["equal_distance", "alpha_cpu", "alpha_memory"])
    else:
        candidates = candidates.sort_values(
            [
                "calibration_normalized_cost",
                "calibration_joint_violation",
                "calibration_max_resource_violation",
                "calibration_normalized_capacity",
                "alpha_cpu",
                "alpha_memory",
            ]
        )
    selected = candidates.iloc[0]
    return AllocationDecision(
        "ADMIT",
        "JOINT_SCREEN_PASS",
        (float(selected["alpha_cpu"]), float(selected["alpha_memory"])),
        (
            float(selected["safety_stock_cpu"]),
            float(selected["safety_stock_memory"]),
        ),
        effective_sizes,
        alpha_min_ess,
        float(selected["calibration_normalized_cost"]),
        float(selected["calibration_joint_violation"]),
        (
            float(selected["cpu_worst_block_coverage"]),
            float(selected["memory_worst_block_coverage"]),
        ),
        float(selected["joint_worst_block_coverage"]),
    )


def select_ecra_allocation(
    *,
    calibration_demand: np.ndarray,
    calibration_point: np.ndarray,
    proper_training_scales: np.ndarray,
    alpha_joint: float,
    cost_weights: Iterable[float],
    c_u: float,
    c_o: float,
    max_lag: int,
    n_blocks: int,
    tolerance: float,
) -> AllocationDecision:
    """Select the lowest-cost rank/ESS/stability-feasible allocation."""
    demand = np.asarray(calibration_demand, dtype=float)
    point = np.asarray(calibration_point, dtype=float)
    if demand.shape != point.shape or demand.ndim != 2:
        raise ValueError("Calibration demand and point forecasts must align.")
    if not np.isfinite(demand).all() or not np.isfinite(point).all():
        raise ValueError("Calibration inputs must be finite.")
    scales = np.asarray(proper_training_scales, dtype=float)
    weights = np.asarray(tuple(cost_weights), dtype=float)
    if scales.shape != (demand.shape[1],) or np.any(scales <= 0.0):
        raise ValueError("One positive proper-training scale is required per resource.")
    if weights.shape != scales.shape or np.any(weights <= 0.0):
        raise ValueError("One positive cost weight is required per resource.")
    weights = weights / weights.sum()
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    residuals = demand - point
    allocations = attainable_allocations(
        tuple(len(residuals) for _ in range(residuals.shape[1])), alpha
    )
    effective_sizes = tuple(
        autocorrelation_effective_sample_size(residuals[:, index], max_lag)[0]
        for index in range(residuals.shape[1])
    )
    alpha_min_ess = float(
        sum(1.0 / (math.floor(value) + 1) for value in effective_sizes)
    )
    if not allocations:
        return AllocationDecision(
            status="DEFER",
            reason_code="JOINT_RANK_INSUFFICIENT",
            alpha_by_resource=None,
            safety_stocks=None,
            effective_sizes=effective_sizes,
            alpha_min_ess=alpha_min_ess,
            calibration_cost=None,
            calibration_joint_violation=None,
            resource_worst_coverage=(),
            joint_worst_coverage=None,
        )

    ess_candidates: list[tuple[float, ...]] = []
    for candidate in allocations:
        if all(
            math.floor(n_eff) >= math.ceil(1.0 / budget - NUMERIC_TOLERANCE) - 1
            for n_eff, budget in zip(effective_sizes, candidate, strict=True)
        ):
            ess_candidates.append(candidate)
    if not ess_candidates:
        return AllocationDecision(
            status="DEFER",
            reason_code="JOINT_ESS_INSUFFICIENT",
            alpha_by_resource=None,
            safety_stocks=None,
            effective_sizes=effective_sizes,
            alpha_min_ess=alpha_min_ess,
            calibration_cost=None,
            calibration_joint_violation=None,
            resource_worst_coverage=(),
            joint_worst_coverage=None,
        )

    stable: list[tuple[tuple[float, float, float, float, tuple[float, ...]], tuple[float, ...], tuple[float, ...], object]] = []
    last_diagnostic = None
    for candidate in ess_candidates:
        diagnostic = resourcewise_block_stability(
            residuals,
            candidate,
            alpha,
            n_blocks=n_blocks,
            tolerance=tolerance,
        )
        last_diagnostic = diagnostic
        if not diagnostic.resource_stable or not diagnostic.joint_stable:
            continue
        thresholds = tuple(
            float(split_conformal_quantile(residuals[:, index], budget)[0])
            for index, budget in enumerate(candidate)
        )
        capacity = np.maximum(0.0, point + np.asarray(thresholds))
        metrics = normalized_mismatch_metrics(
            demand, capacity, scales, weights, c_u, c_o
        )
        stable.append(((*metrics, candidate), candidate, thresholds, diagnostic))
    if not stable:
        assert last_diagnostic is not None
        return AllocationDecision(
            status="MONITOR",
            reason_code=(
                "RESOURCE_BLOCK_UNSTABLE"
                if not last_diagnostic.resource_stable
                else "JOINT_BLOCK_UNSTABLE"
            ),
            alpha_by_resource=None,
            safety_stocks=None,
            effective_sizes=effective_sizes,
            alpha_min_ess=alpha_min_ess,
            calibration_cost=None,
            calibration_joint_violation=None,
            resource_worst_coverage=last_diagnostic.resource_worst_coverage,
            joint_worst_coverage=last_diagnostic.joint_worst_coverage,
        )
    selected = min(stable, key=lambda item: item[0])
    metrics, candidate, thresholds, diagnostic = selected
    return AllocationDecision(
        status="ADMIT",
        reason_code="JOINT_SCREEN_PASS",
        alpha_by_resource=candidate,
        safety_stocks=thresholds,
        effective_sizes=effective_sizes,
        alpha_min_ess=alpha_min_ess,
        calibration_cost=float(metrics[0]),
        calibration_joint_violation=float(metrics[1]),
        resource_worst_coverage=diagnostic.resource_worst_coverage,
        joint_worst_coverage=diagnostic.joint_worst_coverage,
    )


def assess_fixed_allocation(
    *,
    calibration_demand: np.ndarray,
    calibration_point: np.ndarray,
    proper_training_scales: np.ndarray,
    alpha_joint: float,
    alpha_by_resource: Iterable[float],
    cost_weights: Iterable[float],
    c_u: float,
    c_o: float,
    max_lag: int,
    n_blocks: int,
    tolerance: float,
) -> AllocationDecision:
    """Evaluate one predeclared resource-wise risk allocation for admission."""
    demand = np.asarray(calibration_demand, dtype=float)
    point = np.asarray(calibration_point, dtype=float)
    if demand.shape != point.shape or demand.ndim != 2:
        raise ValueError("Calibration demand and point forecasts must align.")
    residuals = demand - point
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    allocation = tuple(validate_alpha(value, "resource alpha") for value in alpha_by_resource)
    if len(allocation) != demand.shape[1] or sum(allocation) > alpha + NUMERIC_TOLERANCE:
        raise ValueError("Resource risk budgets must align and sum within alpha_joint.")
    scales = np.asarray(proper_training_scales, dtype=float)
    weights = np.asarray(tuple(cost_weights), dtype=float)
    weights = weights / weights.sum()
    effective_sizes = tuple(
        autocorrelation_effective_sample_size(residuals[:, index], max_lag)[0]
        for index in range(residuals.shape[1])
    )
    alpha_min_ess = float(
        sum(1.0 / (math.floor(value) + 1) for value in effective_sizes)
    )
    finite = all(
        len(residuals) >= math.ceil(1.0 / budget - NUMERIC_TOLERANCE) - 1
        for budget in allocation
    )
    if not finite:
        return AllocationDecision(
            "DEFER", "JOINT_RANK_INSUFFICIENT", None, None,
            effective_sizes, alpha_min_ess, None, None,
            (), None,
        )
    ess_ok = all(
        math.floor(size) >= math.ceil(1.0 / budget - NUMERIC_TOLERANCE) - 1
        for size, budget in zip(effective_sizes, allocation, strict=True)
    )
    if not ess_ok:
        return AllocationDecision(
            "DEFER", "JOINT_ESS_INSUFFICIENT", None, None,
            effective_sizes, alpha_min_ess, None, None,
            (), None,
        )
    diagnostic = resourcewise_block_stability(
        residuals,
        allocation,
        alpha,
        n_blocks=n_blocks,
        tolerance=tolerance,
    )
    if not diagnostic.resource_stable or not diagnostic.joint_stable:
        return AllocationDecision(
            "MONITOR",
            "RESOURCE_BLOCK_UNSTABLE" if not diagnostic.resource_stable else "JOINT_BLOCK_UNSTABLE",
            None,
            None,
            effective_sizes,
            alpha_min_ess,
            None,
            None,
            diagnostic.resource_worst_coverage,
            diagnostic.joint_worst_coverage,
        )
    thresholds = tuple(
        float(split_conformal_quantile(residuals[:, index], budget)[0])
        for index, budget in enumerate(allocation)
    )
    capacity = np.maximum(0.0, point + np.asarray(thresholds))
    cost, joint_violation, _, _ = normalized_mismatch_metrics(
        demand, capacity, scales, weights, c_u, c_o
    )
    return AllocationDecision(
        "ADMIT",
        "JOINT_SCREEN_PASS",
        allocation,
        thresholds,
        effective_sizes,
        alpha_min_ess,
        cost,
        joint_violation,
        diagnostic.resource_worst_coverage,
        diagnostic.joint_worst_coverage,
    )
