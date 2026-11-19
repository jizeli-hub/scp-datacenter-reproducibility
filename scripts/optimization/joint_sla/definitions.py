"""Shared types and validation for joint-SLA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


AdmissionStatus = Literal["ADMIT", "MONITOR", "DEFER"]

LEGACY_ADMISSION_STATUS = {
    "SCREEN-PASS": "ADMIT",
    "STABILITY-WARNING": "MONITOR",
    "EVIDENCE-INSUFFICIENT": "DEFER",
}
VALID_ADMISSION_STATUSES = frozenset({"ADMIT", "MONITOR", "DEFER"})


def admission_status_from_checks(
    *,
    rank_supported: bool,
    effective_size_supported: bool,
    block_stable: bool | None,
) -> AdmissionStatus:
    """Resolve admission status in Algorithm 1 branch order.

    ``block_stable=None`` means that block stability was not evaluated.  It is
    valid only after rank and effective-size support have both been established.
    """
    if not bool(rank_supported):
        return "DEFER"
    if not bool(effective_size_supported):
        return "DEFER"
    if block_stable is False:
        return "MONITOR"
    return "ADMIT"


def normalize_admission_status(value: object) -> str:
    """Return the canonical admission status for current or legacy records."""
    text = str(value)
    canonical = LEGACY_ADMISSION_STATUS.get(text, text)
    if canonical not in VALID_ADMISSION_STATUSES:
        raise ValueError(f"Unsupported admission status: {value!r}")
    return canonical


def normalize_admission_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Read canonical or legacy status schemas without mutating the input.

    Existing CSVs may expose ``screen_status`` with legacy values. The reader
    retains that source column for provenance and adds canonical
    ``admission_status``. New writers emit only the canonical column.
    """
    normalized = frame.copy()
    if "admission_status" in normalized.columns:
        canonical = normalized["admission_status"].map(normalize_admission_status)
        if "screen_status" in normalized.columns:
            legacy = normalized["screen_status"].map(normalize_admission_status)
            if not canonical.equals(legacy):
                raise ValueError("Canonical and legacy admission status columns disagree.")
        normalized["admission_status"] = canonical
        return normalized
    if "screen_status" not in normalized.columns:
        raise ValueError("Frame lacks admission_status and legacy screen_status.")
    normalized["admission_status"] = normalized["screen_status"].map(
        normalize_admission_status
    )
    return normalized


@dataclass(frozen=True)
class RankFrontier:
    """Nominal finite-rank frontier for one resource-wise construction."""

    calibration_sizes: tuple[int, ...]
    alpha_joint: float
    alpha_min: float
    beta_max: float
    constructible: bool


@dataclass(frozen=True)
class EvidencePlan:
    """Paired-history evidence shortfall and SLA-relaxation plan."""

    calibration_sizes: tuple[int, ...]
    alpha_joint: float
    alpha_min_rank: float
    beta_max_rank: float
    additional_paired_observations: int
    required_service_relaxation: float
    reason_code: str


@dataclass(frozen=True)
class ResourceEvidencePlan:
    """Minimum-cost resource-specific integer evidence plan."""

    calibration_sizes: tuple[int, ...]
    alpha_joint: float
    weights: tuple[float, ...]
    additional_by_resource: tuple[int, ...]
    weighted_cost: float


@dataclass(frozen=True)
class BlockStabilityResult:
    """Calibration-internal resource and joint block diagnostics."""

    resource_worst_coverage: tuple[float, ...]
    joint_worst_coverage: float
    resource_stable: bool
    joint_stable: bool
    evaluable_blocks: int


@dataclass(frozen=True)
class AllocationDecision:
    """Calibration-only decision returned by ECRA-JSLA."""

    status: AdmissionStatus
    reason_code: str
    alpha_by_resource: tuple[float, ...] | None
    safety_stocks: tuple[float, ...] | None
    effective_sizes: tuple[float, ...]
    alpha_min_ess: float
    calibration_cost: float | None
    calibration_joint_violation: float | None
    resource_worst_coverage: tuple[float, ...]
    joint_worst_coverage: float | None


def validate_alpha(alpha: float, name: str = "alpha") -> float:
    """Return a finite violation probability strictly between zero and one."""
    value = float(alpha)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.")
    return value


def validate_aligned_frame(
    frame: pd.DataFrame,
    *,
    allow_source_empty: bool,
) -> pd.DataFrame:
    """Validate aligned CPU-memory rows and a common observed population."""
    required = {
        "datetime",
        "demand_sum",
        "demand_sum_mem",
        "active_machines",
        "active_machines_mem",
        "is_imputed",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Aligned frame is missing columns: {missing}")
    checked = frame.copy()
    checked["datetime"] = pd.to_datetime(checked["datetime"], errors="raise")
    if checked["datetime"].duplicated().any():
        raise ValueError("Aligned frame contains duplicate timestamps.")
    if not checked["datetime"].is_monotonic_increasing:
        raise ValueError("Aligned frame timestamps must be sorted.")
    if not np.array_equal(
        checked["active_machines"].to_numpy(),
        checked["active_machines_mem"].to_numpy(),
    ):
        raise ValueError("CPU and memory do not use the same machine population.")
    for column in ("demand_sum", "demand_sum_mem"):
        values = pd.to_numeric(checked[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"Aligned frame contains invalid {column} values.")
    if not allow_source_empty and checked["is_imputed"].astype(bool).any():
        raise ValueError("Evaluation window contains source-empty timestamps.")
    return checked
