"""Leave-one-block-out stability diagnostics for joint resource claims."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..conformal import split_conformal_quantile

from .definitions import BlockStabilityResult, validate_alpha


def joint_score_block_stability(
    scores: np.ndarray,
    alpha_joint: float,
    *,
    n_blocks: int,
    tolerance: float,
) -> tuple[bool, float]:
    """Evaluate leave-one-block-out stability of scalar joint scores."""
    values = np.asarray(scores, dtype=float)
    alpha = validate_alpha(alpha_joint, "alpha_joint")
    if values.ndim != 1 or len(values) < n_blocks or not np.isfinite(values).all():
        raise ValueError("Finite scalar scores must support every requested block.")
    if n_blocks < 2 or tolerance < 0.0:
        raise ValueError("Invalid block count or tolerance.")
    coverages: list[float] = []
    all_indices = np.arange(len(values))
    for block in np.array_split(all_indices, n_blocks):
        mask = np.ones(len(values), dtype=bool)
        mask[block] = False
        threshold, _, finite = split_conformal_quantile(values[mask], alpha)
        coverages.append(
            float(np.mean(values[block] <= threshold)) if finite else 0.0
        )
    worst = float(min(coverages))
    return bool(worst + 1e-12 >= 1.0 - alpha - tolerance), worst


def resourcewise_block_stability(
    residual_vectors: np.ndarray,
    alpha_by_resource: Iterable[float],
    alpha_joint: float,
    *,
    n_blocks: int,
    tolerance: float,
) -> BlockStabilityResult:
    """Evaluate resource and union-event stability inside calibration only."""
    residuals = np.asarray(residual_vectors, dtype=float)
    if residuals.ndim != 2 or residuals.shape[0] < n_blocks:
        raise ValueError("Residual vectors must support every requested block.")
    if not np.isfinite(residuals).all():
        raise ValueError("Residual vectors must be finite.")
    alphas = tuple(validate_alpha(value, "resource alpha") for value in alpha_by_resource)
    if len(alphas) != residuals.shape[1]:
        raise ValueError("One resource alpha is required per residual column.")
    joint_alpha = validate_alpha(alpha_joint, "alpha_joint")
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least two.")
    if tolerance < 0.0:
        raise ValueError("tolerance cannot be negative.")

    resource_coverages: list[list[float]] = [[] for _ in alphas]
    joint_coverages: list[float] = []
    all_indices = np.arange(len(residuals))
    for block in np.array_split(all_indices, n_blocks):
        train_mask = np.ones(len(residuals), dtype=bool)
        train_mask[block] = False
        thresholds: list[float] = []
        finite = True
        for resource, alpha in enumerate(alphas):
            threshold, _, is_finite = split_conformal_quantile(
                residuals[train_mask, resource], alpha
            )
            thresholds.append(float(threshold))
            finite = finite and bool(is_finite)
        if not finite:
            for values in resource_coverages:
                values.append(0.0)
            joint_coverages.append(0.0)
            continue
        covered = residuals[block] <= np.asarray(thresholds)
        for resource in range(len(alphas)):
            resource_coverages[resource].append(float(np.mean(covered[:, resource])))
        joint_coverages.append(float(np.mean(np.all(covered, axis=1))))

    resource_worst = tuple(float(min(values)) for values in resource_coverages)
    joint_worst = float(min(joint_coverages))
    return BlockStabilityResult(
        resource_worst_coverage=resource_worst,
        joint_worst_coverage=joint_worst,
        resource_stable=all(
            coverage + 1e-12 >= 1.0 - alpha - tolerance
            for coverage, alpha in zip(resource_worst, alphas, strict=True)
        ),
        joint_stable=bool(
            joint_worst + 1e-12 >= 1.0 - joint_alpha - tolerance
        ),
        evaluable_blocks=n_blocks,
    )


def resourcewise_block_stability_many(
    residual_vectors: np.ndarray,
    alpha_candidates: Iterable[tuple[float, float]],
    alpha_joint: float,
    *,
    n_blocks: int,
    tolerance: float,
) -> list[BlockStabilityResult]:
    """Vectorized equivalent of ``resourcewise_block_stability`` for two resources."""
    residuals = np.asarray(residual_vectors, dtype=float)
    candidates = [tuple(float(value) for value in item) for item in alpha_candidates]
    if residuals.ndim != 2 or residuals.shape[1] != 2 or residuals.shape[0] < n_blocks:
        raise ValueError("Two-resource residual vectors must support every block.")
    if not np.isfinite(residuals).all():
        raise ValueError("Residual vectors must be finite.")
    if not candidates:
        return []
    if n_blocks < 2 or tolerance < 0.0:
        raise ValueError("Invalid block count or tolerance.")
    joint_alpha = validate_alpha(alpha_joint, "alpha_joint")
    for candidate in candidates:
        if len(candidate) != 2:
            raise ValueError("Every candidate must contain CPU and memory alpha.")
        for value in candidate:
            validate_alpha(value, "resource alpha")

    unique_cpu = sorted({candidate[0] for candidate in candidates})
    unique_memory = sorted({candidate[1] for candidate in candidates})
    cpu_index = {value: index for index, value in enumerate(unique_cpu)}
    memory_index = {value: index for index, value in enumerate(unique_memory)}
    cpu_coverage = np.ones((n_blocks, len(unique_cpu)), dtype=float)
    memory_coverage = np.ones((n_blocks, len(unique_memory)), dtype=float)
    joint_coverage = np.ones(
        (n_blocks, len(unique_cpu), len(unique_memory)), dtype=float
    )
    all_indices = np.arange(len(residuals))
    for block_number, block in enumerate(np.array_split(all_indices, n_blocks)):
        train_mask = np.ones(len(residuals), dtype=bool)
        train_mask[block] = False
        cpu_thresholds = np.asarray(
            [split_conformal_quantile(residuals[train_mask, 0], alpha)[0] for alpha in unique_cpu]
        )
        memory_thresholds = np.asarray(
            [split_conformal_quantile(residuals[train_mask, 1], alpha)[0] for alpha in unique_memory]
        )
        cpu_covered = residuals[block, 0][:, None] <= cpu_thresholds[None, :]
        memory_covered = residuals[block, 1][:, None] <= memory_thresholds[None, :]
        cpu_coverage[block_number] = np.mean(cpu_covered, axis=0)
        memory_coverage[block_number] = np.mean(memory_covered, axis=0)
        joint_coverage[block_number] = (
            cpu_covered.astype(float).T @ memory_covered.astype(float) / len(block)
        )

    results: list[BlockStabilityResult] = []
    for cpu_alpha, memory_alpha in candidates:
        cpu_position = cpu_index[cpu_alpha]
        memory_position = memory_index[memory_alpha]
        cpu_worst = float(np.min(cpu_coverage[:, cpu_position]))
        memory_worst = float(np.min(memory_coverage[:, memory_position]))
        joint_worst = float(
            np.min(joint_coverage[:, cpu_position, memory_position])
        )
        results.append(
            BlockStabilityResult(
                resource_worst_coverage=(cpu_worst, memory_worst),
                joint_worst_coverage=joint_worst,
                resource_stable=bool(
                    cpu_worst + 1e-12 >= 1.0 - cpu_alpha - tolerance
                    and memory_worst + 1e-12 >= 1.0 - memory_alpha - tolerance
                ),
                joint_stable=bool(
                    joint_worst + 1e-12 >= 1.0 - joint_alpha - tolerance
                ),
                evaluable_blocks=n_blocks,
            )
        )
    return results
