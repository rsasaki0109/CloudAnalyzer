"""Stable, minimal interface for point-cloud map evaluation.

This slice started as ``ca/experiments/map_evaluate/`` with two
non-comparable strategies on separate metric-family lanes:

- ``nn_thresholds`` — reference-based (GT-aware) MapEval-style accuracy and
  completeness@τ via per-point nearest-neighbor distances (scipy cKDTree).
- ``voxel_entropy`` — reference-free self-consistency proxy via voxelized
  neighborhood occupancy entropy.

Because the two strategies do not compete head-to-head, "promoting to core"
means lifting the **request/result contract plus the adopted reference-based
strategy** here, leaving ``voxel_entropy`` in ``ca/experiments/map_evaluate``
as the orthogonal reference-free option. Callers that just want GT-based
metrics can now depend on ``ca.core.map_evaluate`` directly and the CLI
no longer reaches into ``ca.experiments``.

The reference-free lane remains experimental until we settle on a single
GT-free metric — anyone needing it for now imports from
``ca.experiments.map_evaluate.voxel_entropy`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Protocol, cast

import numpy as np


# ---------------------------------------------------------------- contract


MAP_EVAL_PROTOCOL_VERSION = "cloudanalyzer.mapeval_awd_scs.v1"


@dataclass(frozen=True, slots=True)
class MapEvalProtocol:
    """Explicit parameters for the MapEval AWD/SCS-compatible lane.

    The upstream MapEval implementation is a separate C++/TBB project.  This
    object makes the parameters that affect CloudAnalyzer's Python result
    serializable and comparable, rather than relying on hidden defaults.
    """

    voxel_size_m: float = 3.0
    min_voxel_points: int = 100
    neighbor_radius: int = 5
    accuracy_level_m: tuple[float, ...] = (0.2, 0.1, 0.08, 0.05, 0.01)
    name: str = MAP_EVAL_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.voxel_size_m <= 0:
            raise ValueError("MapEval voxel_size_m must be > 0")
        if self.min_voxel_points < 1:
            raise ValueError("MapEval min_voxel_points must be >= 1")
        if self.neighbor_radius < 1:
            raise ValueError("MapEval neighbor_radius must be >= 1")
        if not self.accuracy_level_m or any(value <= 0 for value in self.accuracy_level_m):
            raise ValueError("MapEval accuracy_level_m must contain positive values")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible protocol description."""
        return {
            "name": self.name,
            "version": MAP_EVAL_PROTOCOL_VERSION,
            "voxel_size_m": float(self.voxel_size_m),
            "min_voxel_points": int(self.min_voxel_points),
            "neighbor_radius": int(self.neighbor_radius),
            "accuracy_level_m": [float(value) for value in self.accuracy_level_m],
            "implementation": "cloudanalyzer-python",
            "upstream_reference": "https://github.com/JokerJohn/Cloud_Map_Evaluation",
        }


@dataclass(slots=True)
class MapEvaluateRequest:
    """Compare an estimated point cloud map to a reference (optional).

    Strategies with ``reference_required=True`` raise on ``reference_points=None``;
    reference-free strategies treat the same field as unused.
    """

    estimated_points: np.ndarray  # (N, 3)
    reference_points: np.ndarray | None = None  # (M, 3) when present
    # MapEval-inspired: allow using an external coarse alignment.
    initial_transform_4x4: np.ndarray | None = None  # shape (4, 4)
    # "none" = use estimated_points as-is, "initial" = pre-apply initial_transform_4x4.
    align_mode: str = "none"
    downsample_voxel_size: float = 0.0
    # Optional output dir for strategy-specific artifacts (colored PLYs, etc.)
    artifact_dir: str | None = None
    # MapEval "accuracy_level" thresholds for inlier ratios.
    thresholds_m: tuple[float, ...] = (0.2, 0.1, 0.08, 0.05, 0.01)
    # A coarse voxel size useful for structure-aware proxies.
    structure_voxel_size: float = 0.5
    # Explicit MapEval-compatible AWD/SCS protocol.  ``None`` retains the
    # historical structure_voxel_size/thresholds_m behavior.
    map_eval_protocol: MapEvalProtocol | None = None


@dataclass(slots=True)
class MapEvaluateResult:
    """Common output for all strategies.

    Classification fields (``metric_family``, ``reference_required``, ``mode``,
    ``sampling_policy``) describe *how* the metrics were computed so consumers
    (CI gates, reports, batch aggregators) can keep reference-based and
    reference-free metrics in separate lanes without parsing metric names.
    """

    strategy: str
    design: str
    metrics: dict[str, float]
    artifacts: dict[str, Any]
    metric_family: str = "unspecified"
    reference_required: bool = False
    mode: str = "exact"
    sampling_policy: dict[str, Any] = field(default_factory=dict)


class MapEvaluateStrategy(Protocol):
    """Protocol kept in core after promoting the reference-based strategy."""

    name: str
    design: str

    def evaluate(self, request: MapEvaluateRequest) -> MapEvaluateResult:
        """Evaluate a map and return classified metrics."""


# ---------------------------------------------------------------- helpers


def _require_xyz(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be shape (N, 3); got {arr.shape}")
    return arr


def _require_transform_4x4(matrix: np.ndarray, name: str) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{name} must be shape (4, 4); got {mat.shape}")
    return mat


def apply_transform(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    pts = _require_xyz(points, "points")
    t = _require_transform_4x4(transform_4x4, "transform_4x4")
    if pts.shape[0] == 0:
        return pts
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (hom @ t.T)[:, :3]
    return np.asarray(out, dtype=np.float64)


def aligned_estimated_points(request: MapEvaluateRequest) -> np.ndarray:
    est = _require_xyz(request.estimated_points, "estimated_points")
    mode = (request.align_mode or "none").strip().lower()
    if mode == "none":
        return est
    if mode == "initial":
        if request.initial_transform_4x4 is None:
            raise ValueError("align_mode='initial' requires initial_transform_4x4.")
        return apply_transform(est, request.initial_transform_4x4)
    raise ValueError(f"Unsupported align_mode: {request.align_mode!r}. Use 'none' or 'initial'.")


def ensure_artifact_dir(request: MapEvaluateRequest, subdir: str) -> Path | None:
    """Return an ensured artifact directory path or None if disabled."""
    if request.artifact_dir is None:
        return None
    root = Path(request.artifact_dir)
    out = root / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def _regularized_covariance(sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sym = (np.asarray(sigma, dtype=np.float64) + np.asarray(sigma, dtype=np.float64).T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, eps)
    return np.asarray(eigvecs @ np.diag(eigvals) @ eigvecs.T, dtype=np.float64)


def wasserstein_distance_gaussian(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """L2 Wasserstein distance between 3D Gaussians (MapEval eq. 6)."""
    # Chunk reducers can arrive at the same Gaussian through slightly
    # different summation orders.  Treat that roundoff as the exact identity
    # case before the eigendecompositions amplify it in nearly rank-deficient
    # covariances (for example, a two-point voxel).
    if np.allclose(mu1, mu2, rtol=1e-12, atol=1e-12) and np.allclose(
        sigma1, sigma2, rtol=1e-12, atol=1e-12
    ):
        return 0.0
    mu_diff = np.asarray(mu1, dtype=np.float64) - np.asarray(mu2, dtype=np.float64)
    s1 = _regularized_covariance(sigma1)
    s2 = _regularized_covariance(sigma2)
    # The Bures term needs the *symmetric positive-semidefinite* matrix
    # square root.  A Cholesky factor is not a matrix square root and its
    # trace gives incorrect distances for non-diagonal covariances.
    eigvals, eigvecs = np.linalg.eigh(s1)
    sqrt_s1 = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0.0))) @ eigvecs.T
    # ``middle`` is already positive semidefinite up to roundoff.  Applying
    # the absolute covariance floor here would change the Bures distance for
    # normal small-map covariances (and can make a non-zero mean shift collapse
    # to zero after the final ``max(0, ...)``).  Only clamp eigenvalues when
    # taking the square root; keep the matrix itself unshifted.
    middle = sqrt_s1 @ s2 @ sqrt_s1
    middle = (middle + middle.T) / 2.0
    middle_eigvals = np.linalg.eigvalsh(middle)
    trace_sqrt_middle = float(np.sqrt(np.maximum(middle_eigvals, 0.0)).sum())
    dist_sq = float(mu_diff @ mu_diff) + float(np.trace(s1 + s2)) - 2.0 * trace_sqrt_middle
    # Roundoff around identical distributions can leave a few ulps above zero.
    if abs(dist_sq) <= 1e-12 * max(float(np.trace(s1 + s2)), 1.0):
        dist_sq = 0.0
    return float(np.sqrt(max(0.0, dist_sq)))


@dataclass(slots=True)
class VoxelGaussian:
    mu: np.ndarray
    sigma: np.ndarray
    num_points: int


_WASSERSTEIN_BATCH_SIZE = 256
_WASSERSTEIN_BATCH_MIN_PAIRS = 128
_WASSERSTEIN_BATCH_SAFE_MAGNITUDE = 1e100


def _batch_diagonal_3d(values: np.ndarray) -> np.ndarray:
    diagonal = np.zeros((len(values), 3, 3), dtype=np.float64)
    diagonal[:, np.arange(3), np.arange(3)] = values
    return diagonal


def _regularized_covariances_batch(sigma: np.ndarray) -> np.ndarray:
    sym = (sigma + sigma.transpose(0, 2, 1)) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, 1e-6)
    regularized: np.ndarray = eigvecs @ _batch_diagonal_3d(eigvals) @ eigvecs.transpose(0, 2, 1)
    return regularized


def _wasserstein_distances_batch(pairs: list[tuple[VoxelGaussian, VoxelGaussian]]) -> np.ndarray:
    """Apply the scalar Gaussian formula to a bounded batch of voxel pairs."""
    def scalar() -> np.ndarray:
        return np.asarray([
            wasserstein_distance_gaussian(a.mu, a.sigma, b.mu, b.sigma) for a, b in pairs
        ], dtype=np.float64)

    if len(pairs) < _WASSERSTEIN_BATCH_MIN_PAIRS or np.geterr() != {
        "divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"
    }:
        return scalar()
    # Keep scalar coercion/exception behavior for nonstandard Gaussian arrays.
    if any(
        type(array) is not np.ndarray or array.dtype != np.float64 or array.shape != shape
        for pair in pairs for voxel in pair
        for array, shape in [(voxel.mu, (3,)), (voxel.sigma, (3, 3))]
    ):
        return scalar()
    mu1 = np.stack([a.mu for a, _ in pairs])
    mu2 = np.stack([b.mu for _, b in pairs])
    sigma1 = np.stack([a.sigma for a, _ in pairs])
    sigma2 = np.stack([b.sigma for _, b in pairs])
    # This bound leaves ample headroom for covariance products and mean norms.
    # Non-finite/extreme batches retain ordered scalar warnings and exceptions.
    limit = _WASSERSTEIN_BATCH_SAFE_MAGNITUDE
    if any(not (np.max(array) <= limit and np.min(array) >= -limit) for array in [mu1, mu2, sigma1, sigma2]):
        return scalar()
    identity = np.all(np.isclose(mu1, mu2, rtol=1e-12, atol=1e-12), axis=1)
    identity &= np.all(np.isclose(sigma1, sigma2, rtol=1e-12, atol=1e-12), axis=(1, 2))
    result = np.zeros(len(pairs), dtype=np.float64)
    active = np.flatnonzero(~identity)
    if not len(active):
        return result
    diff = mu1[active] - mu2[active]
    s1 = _regularized_covariances_batch(sigma1[active])
    s2 = _regularized_covariances_batch(sigma2[active])
    eigvals, eigvecs = np.linalg.eigh(s1)
    sqrt_s1 = eigvecs @ _batch_diagonal_3d(np.sqrt(np.maximum(eigvals, 0.0))) @ eigvecs.transpose(0, 2, 1)
    middle = sqrt_s1 @ s2 @ sqrt_s1
    middle = (middle + middle.transpose(0, 2, 1)) / 2.0
    eigvals = np.linalg.eigvalsh(middle)
    trace_sqrt_middle = np.sqrt(np.maximum(eigvals, 0.0)).sum(axis=1)
    # Retain the scalar dot operation rather than substituting a new reduction.
    norms = np.asarray([float(row @ row) for row in diff], dtype=np.float64)
    trace = np.trace(s1 + s2, axis1=1, axis2=2)
    dist_sq = norms + trace - 2.0 * trace_sqrt_middle
    dist_sq[np.abs(dist_sq) <= 1e-12 * np.maximum(trace, 1.0)] = 0.0
    result[active] = np.sqrt(np.maximum(0.0, dist_sq))
    return result


def _voxel_wasserstein_distances(
    est_map: dict[tuple[int, int, int], VoxelGaussian],
    ref_map: dict[tuple[int, int, int], VoxelGaussian],
    min_voxel_points: int,
) -> Iterator[tuple[tuple[int, int, int], float]]:
    keys: list[tuple[int, int, int]] = []
    pairs: list[tuple[VoxelGaussian, VoxelGaussian]] = []
    for index, est_voxel in est_map.items():
        ref_voxel = ref_map.get(index)
        if ref_voxel is None or est_voxel.num_points < min_voxel_points or ref_voxel.num_points < min_voxel_points:
            continue
        keys.append(index)
        pairs.append((est_voxel, ref_voxel))
        if len(pairs) == _WASSERSTEIN_BATCH_SIZE:
            values = _wasserstein_distances_batch(pairs)
            yield from ((key, float(value)) for key, value in zip(keys, values, strict=True))
            keys.clear()
            pairs.clear()
            del values
    if pairs:
        values = _wasserstein_distances_batch(pairs)
        yield from ((key, float(value)) for key, value in zip(keys, values, strict=True))


_VOXEL_MEAN_BINCOUNT_MIN_POINTS = 64


def _voxel_coordinate_sums(points: np.ndarray, inverse: np.ndarray, num_voxels: int) -> np.ndarray:
    """Sum each coordinate by voxel without changing point addition order."""
    sums = np.zeros((num_voxels, 3), dtype=np.float64)
    if len(points) < _VOXEL_MEAN_BINCOUNT_MIN_POINTS:
        np.add.at(sums, inverse, points)
        return sums
    # Keep the scatter's warning/error behavior for non-finite or extreme
    # values. The bound leaves headroom even if all points share one voxel.
    limit = np.finfo(np.float64).max / len(points) / 2.0
    if not (np.max(points) <= limit and np.min(points) >= -limit):
        np.add.at(sums, inverse, points)
        return sums
    for axis in range(3):
        sums[:, axis] = np.bincount(inverse, weights=points[:, axis], minlength=num_voxels)
    return sums


def _voxel_covariance_sums(centered: np.ndarray, inverse: np.ndarray, num_voxels: int) -> np.ndarray:
    """Accumulate symmetric products in the original point order."""
    covariance_sums = np.zeros((num_voxels, 3, 3), dtype=np.float64)
    if len(centered) == 0:
        return covariance_sums
    # Preserve scatter warning/error behavior for non-finite/extreme inputs.
    # Leave headroom for every product and all points landing in one voxel.
    limit = np.sqrt(np.finfo(np.float64).max / len(centered)) / 2.0
    if not (np.max(centered) <= limit and np.min(centered) >= -limit):
        np.add.at(covariance_sums, inverse, centered[:, :, None] * centered[:, None, :])
        return covariance_sums
    # bincount adds weights in input order, just as add.at did. Six components
    # avoid the full (N, 3, 3) product temporary; finite scalar multiplication
    # gives identical symmetric entries.
    for row in range(3):
        for col in range(row, 3):
            values = np.bincount(inverse, weights=centered[:, row] * centered[:, col], minlength=num_voxels)
            covariance_sums[:, row, col] = values
            covariance_sums[:, col, row] = values
    return covariance_sums


_PACKED_VOXEL_MIN_POINTS = 64


def _unique_voxel_indices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group integer XYZ keys with an order-preserving bounded scalar code."""
    if len(indices) < _PACKED_VOXEL_MIN_POINTS:
        return np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    lower = tuple(int(value) for value in np.min(indices, axis=0))
    upper = tuple(int(value) for value in np.max(indices, axis=0))
    widths = tuple(hi - lo + 1 for lo, hi in zip(lower, upper))
    # Compute the bound in Python integers before any fixed-width arithmetic.
    # Every local coordinate, radix and intermediate then fits signed int64.
    if widths[0] * widths[1] * widths[2] > np.iinfo(np.int64).max:
        return np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    codes = indices[:, 0] - lower[0]
    codes *= widths[1]
    codes += indices[:, 1] - lower[1]
    codes *= widths[2]
    codes += indices[:, 2] - lower[2]
    unique, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    keys = np.column_stack([
        unique // (widths[1] * widths[2]) + lower[0],
        (unique // widths[2]) % widths[1] + lower[1],
        unique % widths[2] + lower[2],
    ])
    return keys, inverse, counts


def build_voxel_gaussians(points: np.ndarray, voxel_size: float) -> dict[tuple[int, int, int], VoxelGaussian]:
    """Voxelize a point cloud into per-voxel Gaussian summaries."""
    pts = _require_xyz(points, "points")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0.")
    if pts.shape[0] == 0:
        return {}
    # Sort once and reduce contiguous groups.  This avoids one Python list and
    # one small ndarray allocation per input point on million-point maps.
    indices = np.floor(pts / voxel_size).astype(np.int64)
    unique, inverse, counts = _unique_voxel_indices(indices)
    sums = _voxel_coordinate_sums(pts, inverse, unique.shape[0])
    means = sums / counts[:, None]
    centered = pts - means[inverse]
    covariance_sums = _voxel_covariance_sums(centered, inverse, unique.shape[0])
    voxels: dict[tuple[int, int, int], VoxelGaussian] = {}
    for raw_key, mu, covariance_sum, raw_count in zip(unique, means, covariance_sums, counts):
        count = int(cast(int, raw_count))
        if count > 1:
            sigma = covariance_sum / (count - 1)
        else:
            sigma = np.eye(3, dtype=np.float64) * 1e-6
        key_values = tuple(int(value) for value in raw_key)
        if len(key_values) != 3:
            raise ValueError(f"voxel index must have three coordinates; got {key_values!r}")
        key: tuple[int, int, int] = (key_values[0], key_values[1], key_values[2])
        voxels[key] = VoxelGaussian(mu=mu, sigma=sigma, num_points=count)
    return voxels


_STREAM_MOMENT_BATCH_MIN_POINTS = 1_024
_STREAM_MOMENT_BATCH_MAX_GROUPS = 1_024
_STREAM_MOMENT_BATCH_MAX_GROUP_POINTS = 128
_STREAM_MOMENT_BATCH_MIN_SAME_SIZE = 8
_STREAM_MOMENT_BATCH_MAX_POINTS = 32_768
_STREAM_MOMENT_BATCH_MIN_SELECTED = 128
_STREAM_MOMENT_BATCH_SAFE_MAGNITUDE = 1e150


@dataclass(slots=True)
class _StreamGroupMoments:
    selected: np.ndarray
    mean: np.ndarray
    m2: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


def _stream_moments_batch_allowed(points: np.ndarray, counts: np.ndarray) -> bool:
    if len(points) < _STREAM_MOMENT_BATCH_MIN_POINTS or len(counts) < _STREAM_MOMENT_BATCH_MIN_SELECTED:
        return False
    if np.geterr() != {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}:
        return False
    return bool(
        np.isfinite(points).all()
        and np.max(points) <= _STREAM_MOMENT_BATCH_SAFE_MAGNITUDE
        and np.min(points) >= -_STREAM_MOMENT_BATCH_SAFE_MAGNITUDE
    )


def _stream_group_moments(
    points: np.ndarray, starts: np.ndarray, counts: np.ndarray
) -> _StreamGroupMoments | None:
    """Reduce enough equal-size groups inside one bounded group window."""
    if len(counts) < _STREAM_MOMENT_BATCH_MIN_SELECTED:
        return None
    sizes = np.bincount(
        counts[counts <= _STREAM_MOMENT_BATCH_MAX_GROUP_POINTS],
        minlength=_STREAM_MOMENT_BATCH_MAX_GROUP_POINTS + 1,
    )
    sizes[0] = 0
    eligible = np.flatnonzero(sizes >= _STREAM_MOMENT_BATCH_MIN_SAME_SIZE)
    if np.sum(sizes[eligible]) < _STREAM_MOMENT_BATCH_MIN_SELECTED:
        return None
    moments = _StreamGroupMoments(
        selected=np.zeros(len(counts), dtype=bool),
        mean=np.empty((len(counts), 3), dtype=np.float64),
        m2=np.empty((len(counts), 3, 3), dtype=np.float64),
        minimum=np.empty((len(counts), 3), dtype=np.float64),
        maximum=np.empty((len(counts), 3), dtype=np.float64),
    )
    for raw_count in eligible:
        count = int(raw_count)
        slots = np.flatnonzero(counts == count)
        batch_groups = _STREAM_MOMENT_BATCH_MAX_POINTS // count
        offsets = np.arange(count)
        for first in range(0, len(slots), batch_groups):
            block = slots[first:first + batch_groups]
            gathered = points[starts[block, None] + offsets]
            means = np.mean(gathered, axis=1)
            centered = gathered - means[:, None, :]
            moments.mean[block] = means
            moments.m2[block] = centered.transpose(0, 2, 1) @ centered
            moments.minimum[block] = np.min(gathered, axis=1)
            moments.maximum[block] = np.max(gathered, axis=1)
            moments.selected[block] = True
            # Release cohort scratch before allocating the next point block.
            del gathered, means, centered
    return moments


def _update_stream_voxel(
    accumulator: Any, points: np.ndarray, moments: _StreamGroupMoments | None, slot: int, *, finite: bool = False
) -> None:
    if moments is None or not moments.selected[slot]:
        if finite:
            accumulator._update_finite(points)
        else:
            accumulator.update(points)
        return
    mean, m2 = moments.mean[slot], moments.m2[slot]
    minimum, maximum = moments.minimum[slot], moments.maximum[slot]
    # Initial state must own its rows: retaining a view would keep an entire
    # moment window alive after it has been consumed.
    if accumulator.count == 0:
        mean, m2 = mean.copy(), m2.copy()
    if accumulator.minimum is None:
        minimum = minimum.copy()
    if accumulator.maximum is None:
        maximum = maximum.copy()
    accumulator._merge_moments(len(points), mean, m2)
    accumulator._merge_bounds(minimum, maximum)


def build_voxel_gaussians_from_chunks(
    chunks: Any,
    voxel_size: float,
) -> dict[tuple[int, int, int], VoxelGaussian]:
    """Build voxel Gaussian summaries from an iterable of XYZ chunks."""
    from ca.io import PointAccumulator

    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0.")
    accumulators: dict[tuple[int, int, int], PointAccumulator] = {}
    for chunk in chunks:
        pts = _require_xyz(np.asarray(chunk, dtype=np.float64), "point chunk")
        if pts.shape[0] == 0:
            continue
        indices = np.floor(pts / voxel_size).astype(np.int64)
        unique, inverse, counts = _unique_voxel_indices(indices)
        # Stable grouping retains the preceding boolean selection's point
        # order within each voxel and avoids scanning the full chunk per key.
        order = np.argsort(inverse, kind="stable")
        ordered_points = pts[order]
        starts = np.cumsum(counts) - counts if _stream_moments_batch_allowed(ordered_points, counts) else None
        moments = None
        group_end = 0
        for group_index, (raw_key, raw_count) in enumerate(zip(unique, counts)):
            slot = group_index % _STREAM_MOMENT_BATCH_MAX_GROUPS
            if starts is not None and slot == 0:
                # Drop the consumed window before preparing the next one.
                moments = None
                stop = group_index + _STREAM_MOMENT_BATCH_MAX_GROUPS
                moments = _stream_group_moments(ordered_points, starts[group_index:stop], counts[group_index:stop])
            group_start, group_end = group_end, group_end + int(raw_count)
            key_values = tuple(int(value) for value in raw_key)
            if len(key_values) != 3:
                raise ValueError(f"voxel index must have three coordinates; got {key_values!r}")
            key: tuple[int, int, int] = (key_values[0], key_values[1], key_values[2])
            accumulator = accumulators.get(key)
            if accumulator is None:
                accumulator = PointAccumulator()
                accumulators[key] = accumulator
            _update_stream_voxel(
                accumulator, ordered_points[group_start:group_end], moments, slot, finite=starts is not None
            )
        # Release grouping scratch before requesting the next input chunk.
        del ordered_points, order, indices, unique, inverse, counts, starts, moments

    voxels: dict[tuple[int, int, int], VoxelGaussian] = {}
    for key, accumulator in accumulators.items():
        summary = accumulator.finalize()
        voxels[key] = VoxelGaussian(
            mu=np.asarray(cast(np.ndarray, summary["mean"]), dtype=np.float64),
            sigma=np.asarray(cast(np.ndarray, summary["covariance"]), dtype=np.float64),
            num_points=cast(int, summary["count"]),
        )
    return voxels


def _neighbor_indices(index: tuple[int, int, int], radius: int) -> list[tuple[int, int, int]]:
    ix, iy, iz = index
    neighbors: list[tuple[int, int, int]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors.append((ix + dx, iy + dy, iz + dz))
    return neighbors


_SCS_INDEX_MIN_VOXELS = 64
_SCS_INDEX_EXACT_SPAN = 1 << 52
_SCS_QUERY_BATCH_MIN_VOXELS = 1_024
_SCS_QUERY_BATCH_MAX_CENTERS = 256
_SCS_QUERY_BATCH_MAX_HITS = 65_536


def _scs_spatial_index(
    distances: dict[tuple[int, int, int], float],
    radius: int,
) -> tuple[Any, list[tuple[int, int, int]], tuple[int, int, int]] | None:
    """Index eligible integer cells with exactly represented local coordinates."""
    if len(distances) < _SCS_INDEX_MIN_VOXELS or type(radius) is not int or not 2 <= radius <= _SCS_INDEX_EXACT_SPAN:
        return None
    if any(type(key) is not tuple or len(key) != 3 or any(type(axis) is not int for axis in key) for key in distances):
        return None
    keys = sorted(distances)
    origin = tuple(min(key[axis] for key in keys) for axis in range(3))
    local = [tuple(key[axis] - origin[axis] for axis in range(3)) for key in keys]
    if any(axis > _SCS_INDEX_EXACT_SPAN for point in local for axis in point):
        return None
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(local, dtype=np.float64))
    return tree, keys, cast(tuple[int, int, int], origin)


def _scs_neighbor_values(
    distances: dict[tuple[int, int, int], float],
    radius: int,
) -> Iterator[list[float]]:
    """Keep center insertion order and original lexicographic neighbor order."""
    prepared = _scs_spatial_index(distances, radius)
    if prepared is None:
        for index in distances:
            yield [distances[neighbor] for neighbor in _neighbor_indices(index, radius) if neighbor in distances]
        return
    tree, keys, origin = prepared
    # Each center can return at most every indexed cell. Bound the combined
    # hit lists before issuing a vector query, including dense neighborhoods.
    batch_size = min(_SCS_QUERY_BATCH_MAX_CENTERS, _SCS_QUERY_BATCH_MAX_HITS // len(keys))
    if len(keys) < _SCS_QUERY_BATCH_MIN_VOXELS or batch_size < 2:
        for index in distances:
            query = tuple(index[axis] - origin[axis] for axis in range(3))
            slots = tree.query_ball_point(query, float(radius), p=np.inf, return_sorted=True)
            yield [distances[keys[slot]] for slot in slots if keys[slot] != index]
        return
    centers = iter(distances)
    while block := list(islice(centers, batch_size)):
        queries = np.asarray([[index[axis] - origin[axis] for axis in range(3)] for index in block], dtype=np.float64)
        results = tree.query_ball_point(queries, float(radius), p=np.inf, return_sorted=True, workers=1)
        for index, slots in zip(block, results, strict=True):
            yield [distances[keys[slot]] for slot in slots if keys[slot] != index]
        # Release this batch before asking the tree for the next one.
        del results, slots, queries, block


_SCS_TERM_BATCH_MIN_VOXELS = 64
_SCS_TERM_BATCH_MIN_CENTERS = 16
_SCS_TERM_BATCH_MIN_SAME_SIZE = 4
_SCS_TERM_BATCH_MAX_CENTERS = 1_024
_SCS_TERM_BATCH_MAX_VALUES = 65_536


def _scs_term(neighbor_ws: list[float]) -> float:
    mean_neighbor = float(np.mean(neighbor_ws))
    if mean_neighbor <= np.finfo(np.float64).eps:
        return 0.0 if np.allclose(neighbor_ws, 0.0) else float("nan")
    return float(np.std(neighbor_ws)) / mean_neighbor


def _scs_terms_batch(rows: list[list[float]]) -> np.ndarray:
    """Reduce equal-length, already validated neighborhoods in row order."""
    if len(rows) < _SCS_TERM_BATCH_MIN_CENTERS:
        return np.asarray([_scs_term(row) for row in rows], dtype=np.float64)
    groups: dict[int, list[int]] = {}
    for slot, row in enumerate(rows):
        groups.setdefault(len(row), []).append(slot)
    terms = np.empty(len(rows), dtype=np.float64)
    for slots in groups.values():
        if len(slots) < _SCS_TERM_BATCH_MIN_SAME_SIZE:
            terms[slots] = [_scs_term(rows[slot]) for slot in slots]
            continue
        values = np.asarray([rows[slot] for slot in slots], dtype=np.float64)
        means = np.mean(values, axis=1)
        positive = means > np.finfo(np.float64).eps
        indices = np.asarray(slots)
        terms[indices[positive]] = np.std(values[positive], axis=1) / means[positive]
        terms[indices[~positive]] = np.where(np.all(np.isclose(values[~positive], 0.0), axis=1), 0.0, np.nan)
    return terms


def _scs_terms(distances: dict[tuple[int, int, int], float], radius: int) -> Iterator[float]:
    """Bound SCS batches while retaining center and within-neighborhood order."""
    def scalar() -> Iterator[float]:
        for row in _scs_neighbor_values(distances, radius):
            if row:
                yield _scs_term(row)

    if (
        len(distances) < _SCS_TERM_BATCH_MIN_VOXELS
        or np.geterr() != {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
        or any(type(value) is not float for value in distances.values())
    ):
        yield from scalar()
        return
    values = np.asarray(list(distances.values()), dtype=np.float64)
    # Every neighborhood is a subset of these values. Leave headroom for
    # squared deviations and their sum; reject unsafe inputs before reordering
    # any reductions so scalar warnings/exceptions retain their sequence.
    limit = np.sqrt(np.finfo(np.float64).max / len(distances)) / 4.0
    safe = np.min(values) >= 0.0 and np.max(values) <= limit
    del values
    if not safe:
        yield from scalar()
        return
    pending: list[list[float]] = []
    count = 0
    for row in _scs_neighbor_values(distances, radius):
        if not row:
            continue
        if pending and (len(pending) >= _SCS_TERM_BATCH_MAX_CENTERS or count + len(row) > _SCS_TERM_BATCH_MAX_VALUES):
            yield from (float(value) for value in _scs_terms_batch(pending))
            pending.clear()
            count = 0
        if len(row) > _SCS_TERM_BATCH_MAX_VALUES:
            # A single large neighborhood keeps the preceding scalar path.
            yield _scs_term(row)
        else:
            pending.append(row)
            count += len(row)
    if pending:
        yield from (float(value) for value in _scs_terms_batch(pending))


def _compute_voxel_wasserstein_metrics_from_maps(
    est_map: dict[tuple[int, int, int], VoxelGaussian],
    ref_map: dict[tuple[int, int, int], VoxelGaussian],
    *,
    min_voxel_points: int,
    neighbor_radius: int,
) -> dict[str, float]:
    """Compute AWD/SCS from already-built voxel Gaussian maps."""
    if min_voxel_points < 1:
        raise ValueError("min_voxel_points must be >= 1 for AWD/SCS.")
    if neighbor_radius < 1:
        raise ValueError("neighbor_radius must be >= 1 for AWD/SCS.")

    wasserstein_distances = dict(_voxel_wasserstein_distances(est_map, ref_map, min_voxel_points))

    if not wasserstein_distances:
        return {
            "awd_m": float("nan"),
            "scs": float("nan"),
            "n_awd_voxels": 0.0,
            "n_scs_voxels": 0.0,
        }

    ws_values = list(wasserstein_distances.values())
    awd_m = float(np.mean(ws_values))

    scs_terms = list(_scs_terms(wasserstein_distances, neighbor_radius))
    scs = float(np.nanmean(scs_terms)) if scs_terms else float("nan")
    return {
        "awd_m": awd_m,
        "scs": scs,
        "n_awd_voxels": float(len(wasserstein_distances)),
        "n_scs_voxels": float(len(scs_terms)),
    }


def compute_voxel_wasserstein_metrics(
    estimated_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    voxel_size: float,
    min_voxel_points: int = 100,
    neighbor_radius: int = 5,
) -> dict[str, float]:
    """Compute MapEval-style AWD and SCS from voxelized Gaussian summaries."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0 for AWD/SCS.")
    if min_voxel_points < 1:
        raise ValueError("min_voxel_points must be >= 1 for AWD/SCS.")
    if neighbor_radius < 1:
        raise ValueError("neighbor_radius must be >= 1 for AWD/SCS.")
    est_map = build_voxel_gaussians(estimated_points, voxel_size)
    ref_map = build_voxel_gaussians(reference_points, voxel_size)
    return _compute_voxel_wasserstein_metrics_from_maps(
        est_map,
        ref_map,
        min_voxel_points=min_voxel_points,
        neighbor_radius=neighbor_radius,
    )


def compute_voxel_wasserstein_metrics_from_chunks(
    estimated_chunks: Any,
    reference_chunks: Any,
    *,
    voxel_size: float,
    min_voxel_points: int = 100,
    neighbor_radius: int = 5,
) -> dict[str, float]:
    """Compute AWD/SCS while retaining only per-voxel moments in memory."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0 for AWD/SCS.")
    est_map = build_voxel_gaussians_from_chunks(estimated_chunks, voxel_size)
    ref_map = build_voxel_gaussians_from_chunks(reference_chunks, voxel_size)
    return _compute_voxel_wasserstein_metrics_from_maps(
        est_map,
        ref_map,
        min_voxel_points=min_voxel_points,
        neighbor_radius=neighbor_radius,
    )


_DOWNSAMPLE_SINGLETON_BATCH_POINTS = 65_536
_DOWNSAMPLE_SMALL_GROUP_MAX_POINTS = 8
_DOWNSAMPLE_SMALL_GROUP_MIN_GROUPS = 4_096
_DOWNSAMPLE_SMALL_GROUP_BATCH_POINTS = 65_536


def _sorted_voxel_centroids(points: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Reduce sorted groups without retaining one ndarray per centroid."""
    out = np.empty((len(starts), 3), dtype=np.float64)
    ends = np.concatenate((starts[1:], [len(points)]))
    remaining: np.ndarray | range
    if np.isfinite(points).all():
        singletons = ends - starts == 1
        slots = np.flatnonzero(singletons)
        # Keep NumPy's sum/divide mean arithmetic, including signed zero and
        # subnormals, rather than substituting the input point directly.
        for start in range(0, len(slots), _DOWNSAMPLE_SINGLETON_BATCH_POINTS):
            block = slots[start:start + _DOWNSAMPLE_SINGLETON_BATCH_POINTS]
            out[block] = np.mean(points[starts[block], None, :], axis=1)
        remaining = np.flatnonzero(~singletons)
        # Batch only enough small groups to amortize selection/gather work.
        # Keep the preceding path when underflow reporting is requested, and
        # leave headroom for every addition in the batched means.
        if len(remaining) >= _DOWNSAMPLE_SMALL_GROUP_MIN_GROUPS and np.geterr()["under"] == "ignore":
            counts = ends[remaining] - starts[remaining]
            batchable = (counts >= 2) & (counts <= _DOWNSAMPLE_SMALL_GROUP_MAX_POINTS)
            limit = np.finfo(np.float64).max / _DOWNSAMPLE_SMALL_GROUP_MAX_POINTS / 2.0
            if (
                np.count_nonzero(batchable) >= _DOWNSAMPLE_SMALL_GROUP_MIN_GROUPS
                and np.max(points) <= limit
                and np.min(points) >= -limit
            ):
                for count in range(2, _DOWNSAMPLE_SMALL_GROUP_MAX_POINTS + 1):
                    slots = remaining[counts == count]
                    offsets = np.arange(count)
                    batch_groups = _DOWNSAMPLE_SMALL_GROUP_BATCH_POINTS // count
                    for start in range(0, len(slots), batch_groups):
                        block = slots[start:start + batch_groups]
                        out[block] = np.mean(points[starts[block, None] + offsets], axis=1)
                remaining = remaining[~batchable]
    else:
        # Preserve ordered mean calls and invalid/overflow behavior.
        remaining = range(len(starts))
    for slot in remaining:
        out[slot] = points[starts[slot]:ends[slot]].mean(axis=0)
    return out


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Deterministic voxel downsample by centroid per voxel."""
    pts = _require_xyz(points, "points")
    if pts.shape[0] == 0 or voxel_size <= 0:
        return pts
    keys = np.floor(pts / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    pts_sorted = pts[order]
    # Lexsort already makes equal keys contiguous; compare adjacent rows
    # instead of sorting/classifying the same integer coordinates again.
    changed = np.any(keys_sorted[1:] != keys_sorted[:-1], axis=1)
    starts = np.concatenate(([0], np.flatnonzero(changed) + 1))
    return _sorted_voxel_centroids(pts_sorted, starts)


# ------------------------------------------------------- adopted strategy


_PARALLEL_NN_MIN_POINTS = 65_536
_MIDPOINT_NN_MIN_POINTS = 65_536


def _min_distances_kdtree(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point min Euclidean distance from ``a`` to ``b`` via scipy cKDTree."""
    if a.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if b.shape[0] == 0:
        return np.full((a.shape[0],), np.inf, dtype=np.float64)

    from scipy.spatial import cKDTree

    target = np.asarray(b, dtype=np.float64)
    if min(len(a), len(b)) >= _MIDPOINT_NN_MIN_POINTS:
        # Retain compact bounds and exact queries; avoid median partitioning
        # when both maps are large enough to amortize the query workload.
        tree = cKDTree(target, balanced_tree=False)
    else:
        tree = cKDTree(target)
    workers = -1 if a.shape[0] >= _PARALLEL_NN_MIN_POINTS else 1
    distances, _ = tree.query(np.asarray(a, dtype=np.float64), k=1, workers=workers)
    return np.asarray(distances, dtype=np.float64)


def _error_colors(dist_m: np.ndarray, vmax_m: float) -> np.ndarray:
    """Green→red ramp for distance visualization."""
    vmax = float(max(vmax_m, 1e-12))
    x = np.clip(dist_m / vmax, 0.0, 1.0)
    r = x
    g = 1.0 - x
    b = np.zeros_like(x)
    return np.column_stack([r, g, b]).astype(np.float64)


def _write_colored_ply(points: np.ndarray, colors: np.ndarray, path: str) -> None:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


@dataclass(slots=True)
class NNThresholdMapEvaluateStrategy:
    """MapEval-style accuracy@τ / completeness@τ via per-point NN distances.

    Promoted from ``ca/experiments/map_evaluate/nn_thresholds.py`` as the
    adopted reference-based map evaluation strategy. The reference-free
    ``voxel_entropy`` lane stays under ``ca.experiments`` because the two
    strategies do not compete on the same metric family.
    """

    name: str = "nn_thresholds"
    design: str = "functional"

    def evaluate(self, request: MapEvaluateRequest) -> MapEvaluateResult:
        est_aligned = aligned_estimated_points(request)
        est = voxel_downsample(_require_xyz(est_aligned, "estimated_points"), request.downsample_voxel_size)
        if request.reference_points is None:
            raise ValueError("nn_thresholds requires reference_points (GT).")
        ref = voxel_downsample(_require_xyz(request.reference_points, "reference_points"), request.downsample_voxel_size)

        structure_voxel = float(request.structure_voxel_size)
        thresholds = tuple(float(x) for x in request.thresholds_m)
        protocol = request.map_eval_protocol
        if protocol is None:
            if structure_voxel <= 0:
                protocol = None
            else:
                protocol = MapEvalProtocol(
                    voxel_size_m=structure_voxel,
                    accuracy_level_m=thresholds,
                )
        if protocol is not None:
            structure_voxel = float(protocol.voxel_size_m)
            thresholds = tuple(float(x) for x in protocol.accuracy_level_m)

        est_to_ref = _min_distances_kdtree(est, ref)
        ref_to_est = _min_distances_kdtree(ref, est)

        metrics: dict[str, float] = {
            "n_est": float(est.shape[0]),
            "n_ref": float(ref.shape[0]),
            "mean_est_to_ref_m": float(np.mean(est_to_ref)) if est_to_ref.size else float("inf"),
            "mean_ref_to_est_m": float(np.mean(ref_to_est)) if ref_to_est.size else float("inf"),
            "chamfer_m": (
                float(np.mean(est_to_ref)) + float(np.mean(ref_to_est))
                if (est_to_ref.size and ref_to_est.size)
                else float("inf")
            ),
        }

        for t in thresholds:
            metrics[f"accuracy@{t:.3f}m"] = float(np.mean(est_to_ref <= t)) if est_to_ref.size else 0.0
            metrics[f"completeness@{t:.3f}m"] = float(np.mean(ref_to_est <= t)) if ref_to_est.size else 0.0

        if structure_voxel > 0:
            voxel_metrics = compute_voxel_wasserstein_metrics(
                est,
                ref,
                voxel_size=structure_voxel,
                min_voxel_points=(protocol.min_voxel_points if protocol else 100),
                neighbor_radius=(protocol.neighbor_radius if protocol else 5),
            )
            # Keep the public metric mapping JSON/comparison friendly: an
            # unavailable metric is represented by its zero support count,
            # rather than a NaN value that is unequal to itself.
            metrics.update(
                {
                    key: value
                    for key, value in voxel_metrics.items()
                    if not key.startswith(("awd_", "scs")) or np.isfinite(value)
                }
            )
            metrics["n_awd_voxels"] = voxel_metrics["n_awd_voxels"]
            metrics["n_scs_voxels"] = voxel_metrics["n_scs_voxels"]

        # One scalar summary (F-score) using the first threshold.
        t0 = thresholds[0] if thresholds else 0.2
        acc = metrics.get(f"accuracy@{t0:.3f}m", 0.0)
        com = metrics.get(f"completeness@{t0:.3f}m", 0.0)
        fscore = (2 * acc * com / (acc + com)) if (acc + com) > 0 else 0.0
        metrics[f"fscore@{t0:.3f}m"] = float(fscore)

        artifacts: dict[str, object] = {
            "thresholds_m": thresholds,
            "downsample_voxel_size": float(request.downsample_voxel_size),
            "align_mode": request.align_mode,
            "map_eval_protocol": protocol.as_dict() if protocol is not None else None,
        }

        # Optional: MapEval-style raw/inlier error visualization PLYs.
        out_dir = ensure_artifact_dir(request, "map_evaluate/nn_thresholds")
        if out_dir is not None and est.shape[0] > 0:
            vmax = thresholds[0] if thresholds else 0.2
            raw_colors = _error_colors(est_to_ref, vmax_m=vmax)
            raw_path = str(out_dir / "estimated_error_raw.ply")
            _write_colored_ply(est, raw_colors, raw_path)

            inlier_mask = est_to_ref <= vmax
            inlier_points = est[inlier_mask]
            inlier_colors = raw_colors[inlier_mask]
            inlier_path = str(out_dir / f"estimated_error_inlier_{vmax:.3f}m.ply")
            _write_colored_ply(inlier_points, inlier_colors, inlier_path)

            artifacts["estimated_error_raw_ply"] = raw_path
            artifacts[f"estimated_error_inlier_{vmax:.3f}m_ply"] = inlier_path

        downsample_voxel = float(request.downsample_voxel_size)
        mode = "voxelized" if downsample_voxel > 0 else "exact"
        sampling_policy: dict[str, object] = {
            "downsample_voxel_size_m": downsample_voxel,
            "structure_voxel_size_m": structure_voxel,
            "thresholds_m": list(thresholds),
            "align_mode": request.align_mode,
            "nn_backend": "scipy_ckdtree",
            "map_eval_protocol": protocol.as_dict() if protocol is not None else None,
        }

        return MapEvaluateResult(
            strategy=self.name,
            design=self.design,
            metrics=metrics,
            artifacts=artifacts,
            metric_family="reference_based_nn_thresholds",
            reference_required=True,
            mode=mode,
            sampling_policy=sampling_policy,
        )


def evaluate_map(
    request: MapEvaluateRequest,
    strategy: MapEvaluateStrategy | None = None,
) -> MapEvaluateResult:
    """Evaluate a map using the stabilized strategy (defaults to NNThreshold)."""
    eval_strategy = strategy or NNThresholdMapEvaluateStrategy()
    return eval_strategy.evaluate(request)


def evaluate_map_streaming(
    estimated_chunks: Any,
    reference_chunks: Any,
    *,
    protocol: MapEvalProtocol | None = None,
) -> MapEvaluateResult:
    """Evaluate AWD/SCS from chunk iterators without materializing all points.

    This is intentionally a separate result lane: nearest-neighbor accuracy,
    completeness, and Chamfer require a resident reference index, while AWD
    and SCS can be reduced from per-voxel moments.
    """
    selected_protocol = protocol or MapEvalProtocol()
    metrics = compute_voxel_wasserstein_metrics_from_chunks(
        estimated_chunks,
        reference_chunks,
        voxel_size=selected_protocol.voxel_size_m,
        min_voxel_points=selected_protocol.min_voxel_points,
        neighbor_radius=selected_protocol.neighbor_radius,
    )
    protocol_dict = selected_protocol.as_dict()
    return MapEvaluateResult(
        strategy="mapeval_awd_scs_streaming",
        design="functional",
        metrics=metrics,
        artifacts={"map_eval_protocol": protocol_dict},
        metric_family="reference_based_mapeval_awd_scs",
        reference_required=True,
        mode="streaming",
        sampling_policy={
            "map_eval_protocol": protocol_dict,
            "nn_backend": "not_applicable_streaming",
        },
    )


__all__ = [
    "MapEvaluateRequest",
    "MapEvaluateResult",
    "MapEvaluateStrategy",
    "MAP_EVAL_PROTOCOL_VERSION",
    "MapEvalProtocol",
    "NNThresholdMapEvaluateStrategy",
    "VoxelGaussian",
    "aligned_estimated_points",
    "apply_transform",
    "build_voxel_gaussians",
    "build_voxel_gaussians_from_chunks",
    "compute_voxel_wasserstein_metrics",
    "compute_voxel_wasserstein_metrics_from_chunks",
    "ensure_artifact_dir",
    "evaluate_map",
    "evaluate_map_streaming",
    "voxel_downsample",
    "wasserstein_distance_gaussian",
]
