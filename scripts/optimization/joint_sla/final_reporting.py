"""Corrected analytical frontiers and admission summaries for the final revision."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .feasibility import NUMERIC_TOLERANCE, direct_joint_constructible, joint_rank_frontier


def direct_required_history(alpha_joint: float) -> int:
    """Return the nominal paired history needed by direct joint conformal."""
    if not 0.0 < float(alpha_joint) < 1.0:
        raise ValueError("alpha_joint must lie strictly between zero and one.")
    return max(
        0,
        int(math.ceil(1.0 / float(alpha_joint) - NUMERIC_TOLERANCE)) - 1,
    )


def resourcewise_required_history(resources: int, alpha_joint: float) -> int:
    """Return equal-length history required by a resource-wise construction."""
    if isinstance(resources, bool) or int(resources) != resources or resources < 1:
        raise ValueError("resources must be a positive integer.")
    if not 0.0 < float(alpha_joint) < 1.0:
        raise ValueError("alpha_joint must lie strictly between zero and one.")
    return max(
        0,
        int(
            math.ceil(
                int(resources) / float(alpha_joint) - NUMERIC_TOLERANCE
            )
        )
        - 1,
    )


def additional_history_requirements(
    n_cal: int,
    service_level: float,
    *,
    resources: int,
) -> dict[str, int | float | bool]:
    """Return separate resource-wise and direct nominal evidence shortfalls."""
    if isinstance(n_cal, bool) or int(n_cal) != n_cal or n_cal < 0:
        raise ValueError("n_cal must be a nonnegative integer.")
    beta = float(service_level)
    if not 0.0 < beta < 1.0:
        raise ValueError("service_level must lie strictly between zero and one.")
    alpha = 1.0 - beta
    rw_required = resourcewise_required_history(resources, alpha)
    direct_required = direct_required_history(alpha)
    return {
        "n_cal": int(n_cal),
        "service_level": beta,
        "alpha_joint": alpha,
        "rw_required_history": rw_required,
        "direct_required_history": direct_required,
        "rw_additional_paired_hours": max(0, rw_required - int(n_cal)),
        "direct_additional_paired_hours": max(0, direct_required - int(n_cal)),
        "rw_feasible": int(n_cal) >= rw_required,
        "direct_feasible": int(n_cal) >= direct_required,
    }


def build_corrected_frontier_table(
    calibration_lengths: Iterable[int],
    service_levels: Iterable[float],
    *,
    resources: int = 2,
) -> pd.DataFrame:
    """Build the corrected resource-wise/direct nominal frontier table."""
    rows: list[dict[str, object]] = []
    for n_cal in calibration_lengths:
        for service_level in service_levels:
            requirements = additional_history_requirements(
                int(n_cal), float(service_level), resources=resources
            )
            alpha = float(requirements["alpha_joint"])
            frontier = joint_rank_frontier((int(n_cal),) * resources, alpha)
            rows.append(
                {
                    "resources": resources,
                    "n_cal": int(n_cal),
                    "service_level": float(service_level),
                    "alpha_joint": alpha,
                    "alpha_min_rw": frontier.alpha_min,
                    "beta_max_rw": frontier.beta_max,
                    "rw_feasible": frontier.constructible,
                    "rw_additional_paired_hours": requirements[
                        "rw_additional_paired_hours"
                    ],
                    "direct_feasible": direct_joint_constructible(int(n_cal), alpha),
                    "direct_additional_paired_hours": requirements[
                        "direct_additional_paired_hours"
                    ],
                }
            )
    return pd.DataFrame(rows).sort_values(["n_cal", "service_level"]).reset_index(
        drop=True
    )


def joint_violation_indicator(
    demand: np.ndarray, capacity: np.ndarray
) -> np.ndarray:
    """Return one Boolean union-event indicator per aligned resource vector."""
    realized = np.asarray(demand, dtype=float)
    provisioned = np.asarray(capacity, dtype=float)
    if realized.ndim != 2 or realized.shape != provisioned.shape:
        raise ValueError("Demand and capacity must be aligned resource matrices.")
    if not np.isfinite(realized).all() or np.isnan(provisioned).any():
        raise ValueError("Demand must be finite and capacity must not contain NaN.")
    return np.any(realized > provisioned, axis=1)


def admission_summary(rows: pd.DataFrame) -> dict[str, object]:
    """Summarize admission without assigning zero risk to a zero-admit stage."""
    from .definitions import normalize_admission_frame

    rows = normalize_admission_frame(rows)
    unsafe_column = next(
        (name for name in ("unsafe_commitment", "unsafe_admission", "unsafe_pass") if name in rows),
        None,
    )
    if unsafe_column is None:
        raise ValueError("Rows lack an unsafe-commitment audit column.")
    required = {"admission_status", unsafe_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing admission columns: {sorted(missing)}")
    passed = rows["admission_status"].eq("ADMIT")
    warning = rows["admission_status"].eq("MONITOR")
    insufficient = rows["admission_status"].eq("DEFER")
    unsafe = rows[unsafe_column].astype(bool)
    pass_count = int(passed.sum())
    result: dict[str, object] = {
        "contexts": int(len(rows)),
        "admit_count": pass_count,
        "monitor_count": int(warning.sum()),
        "defer_count": int(insufficient.sum()),
        "unsafe_count": int(unsafe.sum()),
        "unsafe_all": float(unsafe.mean()) if len(rows) else float("nan"),
        "unsafe_admitted": (
            float(unsafe.sum() / pass_count) if pass_count else float("nan")
        ),
        "fallback_rate": float((~passed).mean()) if len(rows) else float("nan"),
    }
    if "post_routing_cost" in rows:
        result["post_routing_cost"] = float(rows["post_routing_cost"].mean())
    if "operationally_feasible_after_fallback" in rows:
        result["operationally_feasible_after_fallback"] = int(
            rows["operationally_feasible_after_fallback"].astype(bool).sum()
        )
    return result


def admission_summary_table(
    rows: pd.DataFrame, group_columns: Iterable[str] = ("stage",)
) -> pd.DataFrame:
    """Aggregate admission rows and provide an explicit display value for N/A."""
    groups = tuple(group_columns)
    output: list[dict[str, object]] = []
    for keys, group in rows.groupby(list(groups), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        summary = admission_summary(group)
        value = float(summary["unsafe_admitted"])
        summary["unsafe_admitted_display"] = (
            "N/A" if math.isnan(value) else f"{100.0 * value:.1f}%"
        )
        output.append({**dict(zip(groups, keys, strict=True)), **summary})
    return pd.DataFrame(output)


# Backward-compatible reader aliases for artifacts created before the
# admission vocabulary was standardized. New code should use the names above.
screening_summary = admission_summary
screening_summary_table = admission_summary_table
