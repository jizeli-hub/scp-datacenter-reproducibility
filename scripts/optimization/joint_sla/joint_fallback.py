"""Calibration-tuned joint Static and Reactive fallback policies."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

from .definitions import validate_alpha
from .risk_allocation import normalized_mismatch_metrics


@dataclass(frozen=True)
class JointFallbackDecision:
    """Frozen joint fallback family and resource-specific parameters."""

    strategy: str
    parameters: tuple[float, ...]
    calibration_cost: float
    calibration_joint_violation: float
    calibration_feasible: bool


def _family_decision(
    *,
    family: str,
    proper_demand: np.ndarray,
    calibration_demand: np.ndarray,
    calibration_previous: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    alpha_joint: float,
    c_u: float,
    c_o: float,
    grid: tuple[float, ...],
) -> JointFallbackDecision:
    candidate_capacities: list[list[np.ndarray]] = []
    for resource in range(proper_demand.shape[1]):
        if family == "Static":
            candidate_capacities.append(
                [
                    np.full(
                        len(calibration_demand),
                        float(np.quantile(proper_demand[:, resource], parameter)),
                    )
                    for parameter in grid
                ]
            )
        elif family == "Reactive":
            candidate_capacities.append(
                [calibration_previous[:, resource] * (1.0 + parameter) for parameter in grid]
            )
        else:
            raise ValueError(f"Unsupported fallback family: {family}")

    rows: list[tuple[tuple[float, ...], float, float, float, float]] = []
    for indices in np.ndindex(*(len(grid),) * proper_demand.shape[1]):
        parameters = tuple(grid[index] for index in indices)
        capacity = np.column_stack(
            [candidate_capacities[resource][index] for resource, index in enumerate(indices)]
        )
        cost, joint_violation, max_resource_violation, total_capacity = (
            normalized_mismatch_metrics(
                calibration_demand,
                capacity,
                scales,
                weights,
                c_u,
                c_o,
            )
        )
        rows.append((parameters, cost, joint_violation, max_resource_violation, total_capacity))
    feasible = [row for row in rows if row[2] <= alpha_joint + 1e-12]
    if feasible:
        selected = min(feasible, key=lambda row: (row[1], row[2], row[3], row[4], row[0]))
        is_feasible = True
    else:
        selected = min(rows, key=lambda row: (row[2], row[1], row[3], row[4], row[0]))
        is_feasible = False
    return JointFallbackDecision(
        strategy=family,
        parameters=selected[0],
        calibration_cost=selected[1],
        calibration_joint_violation=selected[2],
        calibration_feasible=is_feasible,
    )


def tune_joint_fallbacks(
    *,
    proper_demand: np.ndarray,
    calibration_demand: np.ndarray,
    calibration_previous: np.ndarray,
    scales: np.ndarray,
    alpha_joint: float,
    c_u: float,
    c_o: float,
    cost_weights: Iterable[float],
    static_quantiles: Iterable[float],
    reactive_buffers: Iterable[float],
) -> dict[str, JointFallbackDecision]:
    """Tune both fallback families and choose one without outer-test data."""
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    proper = np.asarray(proper_demand, dtype=float)
    calibration = np.asarray(calibration_demand, dtype=float)
    previous = np.asarray(calibration_previous, dtype=float)
    if proper.ndim != 2 or calibration.shape != previous.shape:
        raise ValueError("Fallback inputs must be aligned resource matrices.")
    if proper.shape[1] != calibration.shape[1]:
        raise ValueError("Proper-training and calibration resources must align.")
    scale = np.asarray(scales, dtype=float)
    weights = np.asarray(tuple(cost_weights), dtype=float)
    weights = weights / weights.sum()
    static = _family_decision(
        family="Static",
        proper_demand=proper,
        calibration_demand=calibration,
        calibration_previous=previous,
        scales=scale,
        weights=weights,
        alpha_joint=alpha,
        c_u=c_u,
        c_o=c_o,
        grid=tuple(float(value) for value in static_quantiles),
    )
    reactive = _family_decision(
        family="Reactive",
        proper_demand=proper,
        calibration_demand=calibration,
        calibration_previous=previous,
        scales=scale,
        weights=weights,
        alpha_joint=alpha,
        c_u=c_u,
        c_o=c_o,
        grid=tuple(float(value) for value in reactive_buffers),
    )
    candidates = [static, reactive]
    feasible = [item for item in candidates if item.calibration_feasible]
    pool = feasible if feasible else candidates
    selected = min(
        pool,
        key=lambda item: (
            item.calibration_cost if feasible else item.calibration_joint_violation,
            item.calibration_joint_violation if feasible else item.calibration_cost,
            item.strategy,
            item.parameters,
        ),
    )
    return {"Static": static, "Reactive": reactive, "Selected": selected}


def _family_decision_fast(
    *,
    family: str,
    proper_demand: np.ndarray,
    calibration_demand: np.ndarray,
    calibration_previous: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    alpha_joint: float,
    c_u: float,
    c_o: float,
    grid: tuple[float, ...],
) -> JointFallbackDecision:
    """Vectorized two-resource equivalent of ``_family_decision``."""
    if calibration_demand.shape[1] != 2:
        raise ValueError("The fast fallback evaluator currently requires two resources.")
    capacities: list[np.ndarray] = []
    for resource in range(2):
        if family == "Static":
            values = np.asarray(
                [np.quantile(proper_demand[:, resource], value) for value in grid]
            )
            capacities.append(np.tile(values, (len(calibration_demand), 1)))
        elif family == "Reactive":
            capacities.append(
                calibration_previous[:, [resource]]
                * (1.0 + np.asarray(grid, dtype=float)[None, :])
            )
        else:
            raise ValueError(f"Unsupported fallback family: {family}")

    resource_costs: list[np.ndarray] = []
    violations: list[np.ndarray] = []
    violation_rates: list[np.ndarray] = []
    normalized_capacities: list[np.ndarray] = []
    for resource in range(2):
        demand = calibration_demand[:, [resource]]
        capacity = capacities[resource]
        shortfall = np.maximum(0.0, demand - capacity)
        excess = np.maximum(0.0, capacity - demand)
        resource_costs.append(
            np.mean(c_u * shortfall + c_o * excess, axis=0) / scales[resource]
        )
        violation = demand > capacity
        violations.append(violation)
        violation_rates.append(np.mean(violation, axis=0))
        normalized_capacities.append(np.mean(capacity, axis=0) / scales[resource])

    left = violations[0].astype(np.int64)
    right = violations[1].astype(np.int64)
    intersections = left.T @ right
    joint_counts = (
        left.sum(axis=0)[:, None]
        + right.sum(axis=0)[None, :]
        - intersections
    )
    joint = joint_counts / len(calibration_demand)
    cost = (
        weights[0] * resource_costs[0][:, None]
        + weights[1] * resource_costs[1][None, :]
    )
    max_resource = np.maximum(
        violation_rates[0][:, None], violation_rates[1][None, :]
    )
    total_capacity = (
        normalized_capacities[0][:, None]
        + normalized_capacities[1][None, :]
    )
    feasible_mask = joint <= alpha_joint + 1e-12
    if feasible_mask.any():
        row_index, col_index = np.where(feasible_mask)
        order = np.lexsort(
            (
                np.asarray(grid)[col_index],
                np.asarray(grid)[row_index],
                total_capacity[row_index, col_index],
                max_resource[row_index, col_index],
                joint[row_index, col_index],
                cost[row_index, col_index],
            )
        )
        is_feasible = True
    else:
        row_index, col_index = np.indices(joint.shape)
        row_index = row_index.ravel()
        col_index = col_index.ravel()
        order = np.lexsort(
            (
                np.asarray(grid)[col_index],
                np.asarray(grid)[row_index],
                total_capacity[row_index, col_index],
                max_resource[row_index, col_index],
                cost[row_index, col_index],
                joint[row_index, col_index],
            )
        )
        is_feasible = False
    i = int(row_index[order[0]])
    j = int(col_index[order[0]])
    return JointFallbackDecision(
        strategy=family,
        parameters=(float(grid[i]), float(grid[j])),
        calibration_cost=float(cost[i, j]),
        calibration_joint_violation=float(joint[i, j]),
        calibration_feasible=is_feasible,
    )


def tune_joint_fallbacks_fast(
    *,
    proper_demand: np.ndarray,
    calibration_demand: np.ndarray,
    calibration_previous: np.ndarray,
    scales: np.ndarray,
    alpha_joint: float,
    c_u: float,
    c_o: float,
    cost_weights: Iterable[float],
    static_quantiles: Iterable[float],
    reactive_buffers: Iterable[float],
) -> dict[str, JointFallbackDecision]:
    """Tune the frozen two-resource fallback grids without Python products."""
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    proper = np.asarray(proper_demand, dtype=float)
    calibration = np.asarray(calibration_demand, dtype=float)
    previous = np.asarray(calibration_previous, dtype=float)
    scale = np.asarray(scales, dtype=float)
    weights = np.asarray(tuple(cost_weights), dtype=float)
    weights = weights / weights.sum()
    static = _family_decision_fast(
        family="Static", proper_demand=proper,
        calibration_demand=calibration, calibration_previous=previous,
        scales=scale, weights=weights, alpha_joint=alpha, c_u=c_u, c_o=c_o,
        grid=tuple(float(value) for value in static_quantiles),
    )
    reactive = _family_decision_fast(
        family="Reactive", proper_demand=proper,
        calibration_demand=calibration, calibration_previous=previous,
        scales=scale, weights=weights, alpha_joint=alpha, c_u=c_u, c_o=c_o,
        grid=tuple(float(value) for value in reactive_buffers),
    )
    candidates = [static, reactive]
    feasible = [item for item in candidates if item.calibration_feasible]
    pool = feasible if feasible else candidates
    selected = min(
        pool,
        key=lambda item: (
            item.calibration_cost if feasible else item.calibration_joint_violation,
            item.calibration_joint_violation if feasible else item.calibration_cost,
            item.strategy,
            item.parameters,
        ),
    )
    return {"Static": static, "Reactive": reactive, "Selected": selected}


def apply_joint_fallback(
    decision: JointFallbackDecision,
    *,
    proper_demand: np.ndarray,
    previous_demand: np.ndarray,
) -> np.ndarray:
    """Apply a frozen joint fallback to a calibration or outer-test block."""
    proper = np.asarray(proper_demand, dtype=float)
    previous = np.asarray(previous_demand, dtype=float)
    if decision.strategy == "Static":
        values = [
            float(np.quantile(proper[:, resource], parameter))
            for resource, parameter in enumerate(decision.parameters)
        ]
        return np.tile(np.asarray(values), (len(previous), 1))
    if decision.strategy == "Reactive":
        return previous * (1.0 + np.asarray(decision.parameters))
    raise ValueError(f"Unsupported fallback strategy: {decision.strategy}")
