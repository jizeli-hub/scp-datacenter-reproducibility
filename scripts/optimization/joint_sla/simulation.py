"""Frozen bivariate log-demand generator for controlled joint-SLA validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np


BASE_SEED = 20260812
PROPER_TRAINING_LENGTH = 1000
OUTER_TEST_LENGTH = 240
BURN_IN = 500
LOG_MEAN = np.asarray([4.605170185988092, 5.010635294096255], dtype=float)
INNOVATION_COVARIANCE = np.asarray(
    [[0.0144, 0.0048], [0.0048, 0.0064]], dtype=float
)
REGIME_INDEX = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
CALIBRATION_INDEX = {48: 1, 96: 2, 192: 3, 384: 4}
AR_DIAGONAL = {
    "S1": np.asarray([0.10, 0.10]),
    "S2": np.asarray([0.50, 0.50]),
    "S3": np.asarray([0.30, 0.30]),
    "S4": np.asarray([0.10, 0.10]),
}


@dataclass(frozen=True)
class SimulationSeries:
    regime: str
    n_cal: int
    seed: int
    rng_seed: int
    all_demand: np.ndarray

    @property
    def proper_demand(self) -> np.ndarray:
        return self.all_demand[:PROPER_TRAINING_LENGTH]

    @property
    def calibration_demand(self) -> np.ndarray:
        start = PROPER_TRAINING_LENGTH
        return self.all_demand[start : start + self.n_cal]

    @property
    def test_demand(self) -> np.ndarray:
        return self.all_demand[PROPER_TRAINING_LENGTH + self.n_cal :]

    def with_test_demand(self, test_demand: np.ndarray) -> "SimulationSeries":
        replacement = np.asarray(test_demand, dtype=float)
        if replacement.shape != self.test_demand.shape:
            raise ValueError("Replacement test demand must preserve shape.")
        values = self.all_demand.copy()
        values[PROPER_TRAINING_LENGTH + self.n_cal :] = replacement
        return replace(self, all_demand=values)


@dataclass(frozen=True)
class AR1Forecasts:
    """Causal forecasts with parameters estimated on proper training only."""

    point_forecasts: np.ndarray
    intercepts: tuple[float, float]
    coefficients: tuple[float, float]
    fit_end: int

    def calibration(self, n_cal: int) -> np.ndarray:
        start = PROPER_TRAINING_LENGTH
        return self.point_forecasts[start : start + n_cal]

    def test(self, n_cal: int) -> np.ndarray:
        return self.point_forecasts[PROPER_TRAINING_LENGTH + n_cal :]


def _rng_seed(regime: str, n_cal: int, seed: int) -> int:
    if regime not in REGIME_INDEX:
        raise ValueError(f"Unknown regime: {regime}")
    if n_cal not in CALIBRATION_INDEX:
        raise ValueError(f"Unsupported calibration length: {n_cal}")
    if not 0 <= int(seed) <= 99:
        raise ValueError("seed must be an integer from 0 through 99.")
    return (
        BASE_SEED
        + 100000 * REGIME_INDEX[regime]
        + 1000 * CALIBRATION_INDEX[n_cal]
        + int(seed)
    )


def generate_simulation_series(
    regime: str,
    *,
    n_cal: int,
    seed: int,
) -> SimulationSeries:
    """Generate one predeclared proper/calibration/test demand realization."""
    rng_seed = _rng_seed(regime, n_cal, seed)
    rng = np.random.default_rng(rng_seed)
    total = BURN_IN + PROPER_TRAINING_LENGTH + n_cal + OUTER_TEST_LENGTH
    latent = np.empty((total, 2), dtype=float)
    latent[0] = LOG_MEAN
    cholesky = np.linalg.cholesky(INNOVATION_COVARIANCE)
    phi = AR_DIAGONAL[regime]
    for index in range(1, total):
        if regime == "S3":
            base = rng.standard_t(df=5, size=2) * np.sqrt(3.0 / 5.0)
        else:
            base = rng.normal(size=2)
        innovation = cholesky @ base
        latent[index] = LOG_MEAN + phi * (latent[index - 1] - LOG_MEAN) + innovation
    demand = np.exp(latent[BURN_IN:])
    if regime == "S4":
        test_start = PROPER_TRAINING_LENGTH + n_cal
        demand[test_start:] *= np.asarray([1.15, 1.25])
    return SimulationSeries(
        regime=regime,
        n_cal=int(n_cal),
        seed=int(seed),
        rng_seed=rng_seed,
        all_demand=demand,
    )


def fit_causal_log_ar1(series: SimulationSeries) -> AR1Forecasts:
    """Fit separate log-AR(1) models on proper training and freeze parameters."""
    log_demand = np.log(series.all_demand)
    point = np.full_like(series.all_demand, np.nan, dtype=float)
    intercepts: list[float] = []
    coefficients: list[float] = []
    for resource in range(series.all_demand.shape[1]):
        x = log_demand[: PROPER_TRAINING_LENGTH - 1, resource]
        y = log_demand[1:PROPER_TRAINING_LENGTH, resource]
        design = np.column_stack([np.ones(len(x)), x])
        intercept, coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
        intercepts.append(float(intercept))
        coefficients.append(float(coefficient))
        point[1:, resource] = np.exp(
            float(intercept) + float(coefficient) * log_demand[:-1, resource]
        )
    return AR1Forecasts(
        point_forecasts=point,
        intercepts=tuple(intercepts),
        coefficients=tuple(coefficients),
        fit_end=PROPER_TRAINING_LENGTH,
    )


def calibration_fingerprint(series: SimulationSeries) -> str:
    """Hash only proper-training and calibration observations."""
    end = PROPER_TRAINING_LENGTH + series.n_cal
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(series.all_demand[:end]).tobytes())
    digest.update(f"{series.regime}|{series.n_cal}|{series.seed}".encode())
    return digest.hexdigest()
