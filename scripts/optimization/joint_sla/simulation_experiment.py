"""Leakage-free controlled functional validation for joint-SLA provisioning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .direct_joint_conformal import (
    DirectJointDecision,
    assess_direct_joint,
    direct_joint_capacities,
)
from .final_reporting import joint_violation_indicator
from .joint_fallback import (
    JointFallbackDecision,
    apply_joint_fallback,
    tune_joint_fallbacks_fast,
)
from .risk_allocation import (
    allocation_cost_curve,
    assess_fixed_allocation,
    select_ecra_from_curve,
)
from .simulation import (
    PROPER_TRAINING_LENGTH,
    SimulationSeries,
    fit_causal_log_ar1,
)


STATIC_QUANTILES = tuple(
    np.unique(np.concatenate([np.linspace(0.50, 0.995, 100), [0.997, 0.999, 0.9995]]))
)
REACTIVE_BUFFERS = tuple(
    np.unique(np.concatenate([np.linspace(0.0, 1.0, 101), [1.25, 1.50, 2.0]]))
)


def capacity_metrics(
    demand: np.ndarray,
    capacity: np.ndarray,
    *,
    scales: np.ndarray,
    weights: np.ndarray,
    c_u: float,
    c_o: float,
) -> dict[str, float]:
    """Return resource, union-event, capacity, and normalized cost metrics."""
    realized = np.asarray(demand, dtype=float)
    provisioned = np.asarray(capacity, dtype=float)
    scale = np.asarray(scales, dtype=float)
    weight = np.asarray(weights, dtype=float)
    weight = weight / weight.sum()
    if realized.shape != provisioned.shape or realized.ndim != 2:
        raise ValueError("Demand and capacity must be aligned resource matrices.")
    shortfall = np.maximum(0.0, realized - provisioned)
    excess = np.maximum(0.0, provisioned - realized)
    resource_cost = np.mean(c_u * shortfall + c_o * excess, axis=0) / scale
    resource_violation = np.mean(realized > provisioned, axis=0)
    joint = joint_violation_indicator(realized, provisioned)
    normalized_hourly_shortfall = (shortfall / scale) @ weight
    conditional_shortfall = (
        float(normalized_hourly_shortfall[joint].mean())
        if np.any(joint)
        else 0.0
    )
    return {
        "joint_violation": float(np.mean(joint)),
        "cpu_violation": float(resource_violation[0]),
        "memory_violation": float(resource_violation[1]),
        "normalized_joint_cost": float(np.dot(weight, resource_cost)),
        "cpu_normalized_cost": float(resource_cost[0]),
        "memory_normalized_cost": float(resource_cost[1]),
        "normalized_capacity": float(
            np.sum(np.mean(provisioned, axis=0) / scale)
        ),
        "mean_shortfall_cpu": float(np.mean(shortfall[:, 0]) / scale[0]),
        "mean_shortfall_memory": float(np.mean(shortfall[:, 1]) / scale[1]),
        "conditional_normalized_shortfall": conditional_shortfall,
        "mean_excess_cpu": float(np.mean(excess[:, 0]) / scale[0]),
        "mean_excess_memory": float(np.mean(excess[:, 1]) / scale[1]),
    }


def _decision_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _allocation_capacity(
    test_point: np.ndarray,
    safety_stocks: tuple[float, ...],
) -> np.ndarray:
    return np.maximum(0.0, test_point + np.asarray(safety_stocks, dtype=float))


def _row(
    *,
    series: SimulationSeries,
    service_level: float,
    weights: tuple[float, float],
    method: str,
    admission_status: str,
    reason_code: str,
    finite_rank_constructible: bool,
    selected_alpha: tuple[float, float] | None,
    safety_stocks: tuple[float, float] | None,
    effective_sizes: tuple[float, ...],
    alpha_min_ess: float | None,
    resource_worst_coverage: tuple[float, ...],
    joint_worst_coverage: float | None,
    calibration_cost: float | None,
    calibration_joint_violation: float | None,
    selected_capacity: np.ndarray | None,
    routed_capacity: np.ndarray,
    fallback: JointFallbackDecision | None,
    scales: np.ndarray,
    c_u: float,
    c_o: float,
) -> dict[str, object]:
    alpha = 1.0 - service_level
    weight_array = np.asarray(weights, dtype=float)
    selected_metrics = (
        capacity_metrics(
            series.test_demand,
            selected_capacity,
            scales=scales,
            weights=weight_array,
            c_u=c_u,
            c_o=c_o,
        )
        if selected_capacity is not None
        else None
    )
    routed_metrics = capacity_metrics(
        series.test_demand,
        routed_capacity,
        scales=scales,
        weights=weight_array,
        c_u=c_u,
        c_o=c_o,
    )
    unsafe = bool(
        admission_status == "ADMIT"
        and selected_metrics is not None
        and selected_metrics["joint_violation"] > alpha + 1e-12
    )
    decision_payload = {
        "regime": series.regime,
        "n_cal": series.n_cal,
        "seed": series.seed,
        "service_level": service_level,
        "weights": weights,
        "method": method,
        "admission_status": admission_status,
        "reason_code": reason_code,
        "selected_alpha": selected_alpha,
        "safety_stocks": safety_stocks,
        "fallback_strategy": fallback.strategy if fallback else None,
        "fallback_parameters": fallback.parameters if fallback else None,
    }
    result: dict[str, object] = {
        "regime": series.regime,
        "n_cal": series.n_cal,
        "seed": series.seed,
        "rng_seed": series.rng_seed,
        "service_level": service_level,
        "alpha_joint": alpha,
        "weight_cpu": weights[0],
        "weight_memory": weights[1],
        "method": method,
        "admission_status": admission_status,
        "reason_code": reason_code,
        "finite_rank_constructible": finite_rank_constructible,
        "selected_alpha_cpu": selected_alpha[0] if selected_alpha else np.nan,
        "selected_alpha_memory": selected_alpha[1] if selected_alpha else np.nan,
        "safety_stock_cpu": safety_stocks[0] if safety_stocks else np.nan,
        "safety_stock_memory": safety_stocks[1] if safety_stocks else np.nan,
        "n_eff_cpu": effective_sizes[0] if len(effective_sizes) > 0 else np.nan,
        "n_eff_memory": effective_sizes[1] if len(effective_sizes) > 1 else np.nan,
        "alpha_min_ess": alpha_min_ess if alpha_min_ess is not None else np.nan,
        "cpu_worst_block_coverage": (
            resource_worst_coverage[0] if len(resource_worst_coverage) > 0 else np.nan
        ),
        "memory_worst_block_coverage": (
            resource_worst_coverage[1] if len(resource_worst_coverage) > 1 else np.nan
        ),
        "joint_worst_block_coverage": (
            joint_worst_coverage if joint_worst_coverage is not None else np.nan
        ),
        "calibration_normalized_cost": (
            calibration_cost if calibration_cost is not None else np.nan
        ),
        "calibration_joint_violation": (
            calibration_joint_violation
            if calibration_joint_violation is not None
            else np.nan
        ),
        "fallback_strategy": fallback.strategy if fallback else "NONE",
        "fallback_parameters": (
            json.dumps(fallback.parameters) if fallback else "[]"
        ),
        "fallback_calibration_feasible": (
            fallback.calibration_feasible if fallback else np.nan
        ),
        "unsafe_commitment": unsafe,
        "post_routing_joint_feasible": bool(
            routed_metrics["joint_violation"] <= alpha + 1e-12
        ),
        "test_samples": len(series.test_demand),
        "decision_fingerprint": _decision_hash(decision_payload),
        "outer_test_used_for_selection": False,
    }
    for name, value in routed_metrics.items():
        result[f"post_routing_{name}"] = value
    for name in (
        "joint_violation",
        "cpu_violation",
        "memory_violation",
        "normalized_joint_cost",
        "cpu_normalized_cost",
        "memory_normalized_cost",
        "normalized_capacity",
    ):
        result[f"selected_{name}"] = (
            selected_metrics[name] if selected_metrics is not None else np.nan
        )
    return result


def evaluate_simulation_series(
    series: SimulationSeries,
    *,
    service_levels: Iterable[float] = (0.90, 0.95, 0.99),
    weight_settings: Iterable[tuple[float, float]] = (
        (0.50, 0.50),
        (0.25, 0.75),
        (0.75, 0.25),
    ),
    c_u: float = 5.0,
    c_o: float = 1.0,
    max_lag: int = 24,
    n_blocks: int = 4,
    tolerance: float = 0.05,
) -> list[dict[str, object]]:
    """Evaluate all frozen methods without exposing outer test to selection."""
    forecasts = fit_causal_log_ar1(series)
    calibration_point = forecasts.calibration(series.n_cal)
    test_point = forecasts.test(series.n_cal)
    scales = np.mean(series.proper_demand, axis=0)
    calibration_previous = series.all_demand[
        PROPER_TRAINING_LENGTH - 1 : PROPER_TRAINING_LENGTH + series.n_cal - 1
    ]
    test_previous = series.all_demand[
        PROPER_TRAINING_LENGTH + series.n_cal - 1 : -1
    ]
    rows: list[dict[str, object]] = []
    for service_level in service_levels:
        alpha = 1.0 - float(service_level)
        curve = allocation_cost_curve(
            calibration_demand=series.calibration_demand,
            calibration_point=calibration_point,
            proper_training_scales=scales,
            alpha_joint=alpha,
            cost_weights=(0.5, 0.5),
            c_u=c_u,
            c_o=c_o,
            max_lag=max_lag,
            n_blocks=n_blocks,
            tolerance=tolerance,
        )
        for weights in weight_settings:
            normalized_weights = tuple(
                float(value / sum(weights)) for value in weights
            )
            fallbacks = tune_joint_fallbacks_fast(
                proper_demand=series.proper_demand,
                calibration_demand=series.calibration_demand,
                calibration_previous=calibration_previous,
                scales=scales,
                alpha_joint=alpha,
                c_u=c_u,
                c_o=c_o,
                cost_weights=normalized_weights,
                static_quantiles=STATIC_QUANTILES,
                reactive_buffers=REACTIVE_BUFFERS,
            )
            selected_fallback = fallbacks["Selected"]
            fallback_capacity = apply_joint_fallback(
                selected_fallback,
                proper_demand=series.proper_demand,
                previous_demand=test_previous,
            )

            fixed_specs = (
                ("EqualResourceWise", (alpha / 2.0, alpha / 2.0)),
                ("Fixed25CPU75Memory", (alpha * 0.25, alpha * 0.75)),
            )
            for method, allocation in fixed_specs:
                decision = assess_fixed_allocation(
                    calibration_demand=series.calibration_demand,
                    calibration_point=calibration_point,
                    proper_training_scales=scales,
                    alpha_joint=alpha,
                    alpha_by_resource=allocation,
                    cost_weights=normalized_weights,
                    c_u=c_u,
                    c_o=c_o,
                    max_lag=max_lag,
                    n_blocks=n_blocks,
                    tolerance=tolerance,
                )
                selected_capacity = (
                    _allocation_capacity(test_point, decision.safety_stocks)
                    if decision.status == "ADMIT"
                    and decision.safety_stocks is not None
                    else None
                )
                rows.append(
                    _row(
                        series=series,
                        service_level=float(service_level),
                        weights=normalized_weights,
                        method=method,
                        admission_status=decision.status,
                        reason_code=decision.reason_code,
                        finite_rank_constructible=decision.reason_code
                        != "JOINT_RANK_INSUFFICIENT",
                        selected_alpha=decision.alpha_by_resource,
                        safety_stocks=decision.safety_stocks,
                        effective_sizes=decision.effective_sizes,
                        alpha_min_ess=decision.alpha_min_ess,
                        resource_worst_coverage=decision.resource_worst_coverage,
                        joint_worst_coverage=decision.joint_worst_coverage,
                        calibration_cost=decision.calibration_cost,
                        calibration_joint_violation=decision.calibration_joint_violation,
                        selected_capacity=selected_capacity,
                        routed_capacity=(
                            selected_capacity
                            if selected_capacity is not None
                            else fallback_capacity
                        ),
                        fallback=(None if selected_capacity is not None else selected_fallback),
                        scales=scales,
                        c_u=c_u,
                        c_o=c_o,
                    )
                )

            direct = assess_direct_joint(
                calibration_demand=series.calibration_demand,
                calibration_point=calibration_point,
                scales=scales,
                alpha_joint=alpha,
                cost_weights=np.asarray(normalized_weights),
                c_u=c_u,
                c_o=c_o,
                max_lag=max_lag,
                n_blocks=n_blocks,
                tolerance=tolerance,
            )
            direct_capacity = (
                direct_joint_capacities(test_point, scales, direct.threshold)
                if direct.status == "ADMIT" and direct.threshold is not None
                else None
            )
            direct_stock = (
                tuple(float(direct.threshold * value) for value in scales)
                if direct.threshold is not None
                else None
            )
            rows.append(
                _row(
                    series=series,
                    service_level=float(service_level),
                    weights=normalized_weights,
                    method="DirectJointConformal",
                    admission_status=direct.status,
                    reason_code=direct.reason_code,
                    finite_rank_constructible=direct.reason_code
                    != "JOINT_RANK_INSUFFICIENT",
                    selected_alpha=None,
                    safety_stocks=direct_stock,
                    effective_sizes=(direct.effective_size,),
                    alpha_min_ess=(1.0 / (np.floor(direct.effective_size) + 1.0)),
                    resource_worst_coverage=(),
                    joint_worst_coverage=direct.worst_block_coverage,
                    calibration_cost=direct.calibration_cost,
                    calibration_joint_violation=direct.calibration_joint_violation,
                    selected_capacity=direct_capacity,
                    routed_capacity=(
                        direct_capacity if direct_capacity is not None else fallback_capacity
                    ),
                    fallback=(None if direct_capacity is not None else selected_fallback),
                    scales=scales,
                    c_u=c_u,
                    c_o=c_o,
                )
            )

            ecra = select_ecra_from_curve(curve, cost_weights=normalized_weights)
            ecra_capacity = (
                _allocation_capacity(test_point, ecra.safety_stocks)
                if ecra.status == "ADMIT" and ecra.safety_stocks is not None
                else None
            )
            rows.append(
                _row(
                    series=series,
                    service_level=float(service_level),
                    weights=normalized_weights,
                    method="ECRA-JSLA",
                    admission_status=ecra.status,
                    reason_code=ecra.reason_code,
                    finite_rank_constructible=not curve.empty,
                    selected_alpha=ecra.alpha_by_resource,
                    safety_stocks=ecra.safety_stocks,
                    effective_sizes=ecra.effective_sizes,
                    alpha_min_ess=ecra.alpha_min_ess,
                    resource_worst_coverage=ecra.resource_worst_coverage,
                    joint_worst_coverage=ecra.joint_worst_coverage,
                    calibration_cost=ecra.calibration_cost,
                    calibration_joint_violation=ecra.calibration_joint_violation,
                    selected_capacity=ecra_capacity,
                    routed_capacity=(
                        ecra_capacity if ecra_capacity is not None else fallback_capacity
                    ),
                    fallback=(None if ecra_capacity is not None else selected_fallback),
                    scales=scales,
                    c_u=c_u,
                    c_o=c_o,
                )
            )

            for method, decision in (
                ("Reactive", fallbacks["Reactive"]),
                ("Static", fallbacks["Static"]),
            ):
                capacity = apply_joint_fallback(
                    decision,
                    proper_demand=series.proper_demand,
                    previous_demand=test_previous,
                )
                rows.append(
                    _row(
                        series=series,
                        service_level=float(service_level),
                        weights=normalized_weights,
                        method=method,
                        admission_status="NOT-APPLICABLE",
                        reason_code="CALIBRATION_TUNED_OPERATIONAL_BASELINE",
                        finite_rank_constructible=False,
                        selected_alpha=None,
                        safety_stocks=None,
                        effective_sizes=(),
                        alpha_min_ess=None,
                        resource_worst_coverage=(),
                        joint_worst_coverage=None,
                        calibration_cost=decision.calibration_cost,
                        calibration_joint_violation=decision.calibration_joint_violation,
                        selected_capacity=capacity,
                        routed_capacity=capacity,
                        fallback=None,
                        scales=scales,
                        c_u=c_u,
                        c_o=c_o,
                    )
                )
    return rows
