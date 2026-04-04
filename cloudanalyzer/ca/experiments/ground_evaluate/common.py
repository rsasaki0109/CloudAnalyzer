"""Shared datasets and metrics for ground segmentation evaluation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ca.core.ground_evaluate import GroundEvaluateRequest


@dataclass(slots=True)
class GroundEvaluateDatasetCase:
    """Comparable ground segmentation scenario for evaluation experiments."""

    name: str
    description: str
    request: GroundEvaluateRequest
    expected_min_f1: float


def _flat_ground_dataset() -> GroundEvaluateDatasetCase:
    """Flat ground plane at z=0 with objects above. Perfect segmentation."""
    rng = np.random.default_rng(42)
    ground = rng.uniform([-10, -10, -0.1], [10, 10, 0.1], size=(200, 3))
    nonground = rng.uniform([-10, -10, 1.0], [10, 10, 5.0], size=(100, 3))
    return GroundEvaluateDatasetCase(
        name="flat_ground_perfect",
        description="Flat ground with objects above; identical estimation and reference.",
        request=GroundEvaluateRequest(
            estimated_ground=ground.copy(),
            estimated_nonground=nonground.copy(),
            reference_ground=ground.copy(),
            reference_nonground=nonground.copy(),
            voxel_size=0.5,
        ),
        expected_min_f1=0.99,
    )


def _noisy_segmentation_dataset() -> GroundEvaluateDatasetCase:
    """Ground with some points misclassified near the boundary."""
    rng = np.random.default_rng(123)
    ground = rng.uniform([-10, -10, -0.1], [10, 10, 0.1], size=(200, 3))
    nonground = rng.uniform([-10, -10, 0.8], [10, 10, 5.0], size=(100, 3))

    # Estimated: 10% of ground points leak into nonground, 5% of nonground leak into ground
    n_ground_leak = 20
    n_nonground_leak = 5
    est_ground = np.vstack([ground[n_ground_leak:], nonground[:n_nonground_leak]])
    est_nonground = np.vstack([ground[:n_ground_leak], nonground[n_nonground_leak:]])

    return GroundEvaluateDatasetCase(
        name="noisy_boundary",
        description="Ground segmentation with 10% ground leak and 5% nonground leak near boundary.",
        request=GroundEvaluateRequest(
            estimated_ground=est_ground,
            estimated_nonground=est_nonground,
            reference_ground=ground,
            reference_nonground=nonground,
            voxel_size=0.5,
        ),
        expected_min_f1=0.7,
    )


def _sloped_terrain_dataset() -> GroundEvaluateDatasetCase:
    """Sloped ground surface; estimation misses the upper slope region."""
    rng = np.random.default_rng(456)
    xs = rng.uniform(-10, 10, size=200)
    ys = rng.uniform(-10, 10, size=200)
    zs = 0.15 * xs + rng.normal(0, 0.05, size=200)  # gentle slope
    ground = np.column_stack([xs, ys, zs])
    nonground = rng.uniform([-10, -10, 2.0], [10, 10, 6.0], size=(100, 3))

    # Estimation: miss upper slope (x > 7) ground points
    upper_mask = ground[:, 0] > 7.0
    est_ground = ground[~upper_mask]
    est_nonground = np.vstack([ground[upper_mask], nonground])

    return GroundEvaluateDatasetCase(
        name="sloped_terrain_miss",
        description="Sloped ground where upper slope region is missed by the estimator.",
        request=GroundEvaluateRequest(
            estimated_ground=est_ground,
            estimated_nonground=est_nonground,
            reference_ground=ground,
            reference_nonground=nonground,
            voxel_size=0.5,
        ),
        expected_min_f1=0.7,
    )


def build_default_datasets() -> list[GroundEvaluateDatasetCase]:
    """Create deterministic ground evaluation datasets."""
    return [
        _flat_ground_dataset(),
        _noisy_segmentation_dataset(),
        _sloped_terrain_dataset(),
    ]


def perturb_request(request: GroundEvaluateRequest, noise_std: float = 0.05) -> GroundEvaluateRequest:
    """Add small positional noise to estimated points for stability checks."""
    rng = np.random.default_rng(789)
    return GroundEvaluateRequest(
        estimated_ground=request.estimated_ground + rng.normal(0, noise_std, request.estimated_ground.shape),
        estimated_nonground=request.estimated_nonground + rng.normal(0, noise_std, request.estimated_nonground.shape),
        reference_ground=request.reference_ground.copy(),
        reference_nonground=request.reference_nonground.copy(),
        voxel_size=request.voxel_size,
    )
