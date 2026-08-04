"""Reproducible parity harness for the upstream MapEval AWD/SCS lane.

The upstream project is a small C++ application rather than a library.  It
also hard-codes its configuration path and contains two implementation details
that are important for reproducibility:

* the Open3D voxel builder normalizes the covariance accumulator more than
  once before the Gaussian distance is evaluated; and
* the Gaussian Bures term uses Cholesky factors instead of a symmetric matrix
  square root.

CloudAnalyzer keeps the mathematically corrected implementation in
``ca.core.map_evaluate``.  This module provides a separate, explicitly named
``official_compatibility`` lane so an external upstream executable can be
compared without silently weakening the core metric.  The report includes both
lanes and records the source commit, fixture hashes, runtime, and memory
measurements.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from ca import __version__
from ca.core.map_evaluate import (
    MapEvalProtocol,
    build_voxel_gaussians,
    compute_voxel_wasserstein_metrics,
    voxel_downsample,
)


PARITY_REPORT_SCHEMA_VERSION = "cloudanalyzer.mapeval_parity.v1"
UPSTREAM_REPOSITORY = "https://github.com/JokerJohn/Cloud_Map_Evaluation"
UPSTREAM_COMMIT = "5955f495df5fbf39f0a184c5d275823b3b5db31b"
UPSTREAM_LICENSE = "MIT"
UPSTREAM_DEPENDENCIES = {
    "open3d": "0.15.1 (README baseline; >=0.11 stated)",
    "eigen3": "required by upstream CMake",
    "pcl": "required by upstream CMake",
    "yaml_cpp": "required by upstream CMake",
    "tbb": "required by upstream CMake",
    "openmp": "required by upstream CMake",
    "os": "Ubuntu 20.04 (README baseline)",
}

DEFAULT_PROTOCOL = MapEvalProtocol()
DEFAULT_DOWNSAMPLE_VOXEL_SIZE_M = 0.01
DEFAULT_ABS_TOLERANCE = 1e-5
DEFAULT_REL_TOLERANCE = 1e-5


@dataclass(frozen=True, slots=True)
class MapEvalParityFixture:
    """Small deterministic fixture with two neighboring dense AWD voxels."""

    estimated_points: np.ndarray
    reference_points: np.ndarray
    seed: int = 17
    points_per_voxel: int = 128
    name: str = "dense-neighbor-voxels-v1"

    def __post_init__(self) -> None:
        for name, points in (
            ("estimated_points", self.estimated_points),
            ("reference_points", self.reference_points),
        ):
            array = np.asarray(points)
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(f"{name} must be shape (N, 3); got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite points")
        if self.estimated_points.shape != self.reference_points.shape:
            raise ValueError("estimated_points and reference_points must have the same shape")
        if self.points_per_voxel < 100:
            raise ValueError("points_per_voxel must be >= 100 for the official AWD filter")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": int(self.seed),
            "points_per_voxel": int(self.points_per_voxel),
            "estimated_point_count": int(self.estimated_points.shape[0]),
            "reference_point_count": int(self.reference_points.shape[0]),
            "voxel_count": 2,
            "construction": (
                "Two adjacent 3m voxels; each point occupies a distinct 1cm "
                "downsample cell; estimate has deterministic sub-centimeter noise."
            ),
        }


def make_mapeval_smoke_fixture(
    *,
    seed: int = 17,
    points_per_voxel: int = 128,
) -> MapEvalParityFixture:
    """Build a deterministic fixture that survives official 1cm downsampling.

    The official implementation filters AWD voxels below 100 points *after*
    its 1cm Open3D downsample.  A random cloud would therefore make a poor
    smoke fixture: it can spread points across many tiny cells and silently
    produce an empty AWD result.  This fixture deliberately uses a 2cm lattice
    inside each 3m voxel.
    """

    count = int(points_per_voxel)
    if count < 100:
        raise ValueError("points_per_voxel must be >= 100")
    side = int(np.ceil(count ** (1.0 / 3.0)))
    flat = np.arange(count, dtype=np.int64)
    lattice = np.column_stack(
        (
            flat % side,
            (flat // side) % side,
            flat // (side * side),
        )
    ).astype(np.float64)
    # Keep the cloud away from the 3m voxel boundaries.  The estimate offset
    # below is deliberately non-zero, so a near-boundary fixture could make
    # the two maps use different voxel keys and hide a numerical mismatch.
    local = 0.401 + 0.02 * lattice

    rng = np.random.default_rng(seed)
    reference_parts: list[np.ndarray] = []
    estimated_parts: list[np.ndarray] = []
    for voxel_index in (0, 1):
        base = local.copy()
        base[:, 0] += 3.0 * voxel_index
        reference = base + rng.normal(0.0, 0.00035, size=base.shape)
        estimate = reference + np.array([0.012, -0.006, 0.004])
        estimate += rng.normal(0.0, 0.00020, size=base.shape)
        reference_parts.append(reference)
        estimated_parts.append(estimate)

    return MapEvalParityFixture(
        estimated_points=np.vstack(estimated_parts).astype(np.float64),
        reference_points=np.vstack(reference_parts).astype(np.float64),
        seed=seed,
        points_per_voxel=count,
    )


def write_ascii_pcd(path: str | Path, points: np.ndarray) -> Path:
    """Write a dependency-light ASCII PCD readable by Open3D and MapEval."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must be shape (N, 3); got {values.shape}")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {values.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {values.shape[0]}\n"
        "DATA ascii\n"
    )
    with output.open("w", encoding="ascii", newline="\n") as file:
        file.write(header)
        np.savetxt(file, values, fmt="%.17g")
    return output


def write_fixture(fixture: MapEvalParityFixture, root: str | Path) -> dict[str, Path]:
    """Materialize the common fixture and return its two input paths."""

    directory = Path(root)
    estimated = write_ascii_pcd(directory / "estimated" / "map.pcd", fixture.estimated_points)
    reference = write_ascii_pcd(directory / "reference" / "map.pcd", fixture.reference_points)
    (directory / "fixture.json").write_text(
        json.dumps(fixture.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"estimated": estimated, "reference": reference}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file in fixed-size chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regularized_covariance(sigma: np.ndarray) -> np.ndarray:
    symmetric = (sigma + sigma.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return np.asarray(
        eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-6)) @ eigenvectors.T,
        dtype=np.float64,
    )


def _official_wasserstein_distance(
    mu1: np.ndarray,
    sigma1_stored: np.ndarray,
    count1: int,
    mu2: np.ndarray,
    sigma2_stored: np.ndarray,
    count2: int,
) -> float:
    """Port ``VoxelCalculator::computeWassersteinDistanceGaussian`` exactly."""

    if count1 > 1:
        sigma1 = _regularized_covariance(sigma1_stored / (count1 - 1))
    else:
        sigma1 = np.eye(3, dtype=np.float64)
    if count2 > 1:
        sigma2 = _regularized_covariance(sigma2_stored / (count2 - 1))
    else:
        sigma2 = np.eye(3, dtype=np.float64)

    # The upstream code calls the Cholesky factor ``sigma_sqrt`` and takes its
    # trace.  Reproduce that behavior here; this is intentionally not the
    # corrected Bures square root used by ca.core.map_evaluate.
    factor1 = np.linalg.cholesky(sigma1)
    factor_product = factor1 @ sigma2 @ factor1.T
    factor2 = np.linalg.cholesky(factor_product)
    mean_delta = np.asarray(mu1) - np.asarray(mu2)
    distance_sq = float(mean_delta @ mean_delta)
    distance_sq += float(np.trace(sigma1 + sigma2))
    distance_sq -= 2.0 * float(np.trace(factor2))
    return float(np.sqrt(max(0.0, distance_sq)))


def _official_voxel_map(
    points: np.ndarray,
    voxel_size: float,
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, int]]:
    """Reproduce the Open3D path's stored voxel moments.

    ``buildVoxelMap`` first stores sample covariance, ``computeVoxelEntropy``
    divides it again, and ``computeWassersteinDistanceGaussian`` divides it a
    third time while preparing the Gaussian.  The returned ``sigma`` is the
    value immediately before that final division.
    """

    base_map = build_voxel_gaussians(points, voxel_size)
    output: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, int]] = {}
    for key, voxel in base_map.items():
        count = int(voxel.num_points)
        if count > 10:
            stored_sigma = np.asarray(voxel.sigma, dtype=np.float64) / (count - 1)
        else:
            # The Open3D overload leaves sigma at zero for counts <= 10.
            stored_sigma = np.zeros((3, 3), dtype=np.float64)
        output[key] = (np.asarray(voxel.mu), stored_sigma, count)
    return output


def _neighbor_keys(index: tuple[int, int, int], radius: int) -> list[tuple[int, int, int]]:
    return [
        (index[0] + dx, index[1] + dy, index[2] + dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]


def compute_official_compatible_metrics(
    estimated_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    protocol: MapEvalProtocol = DEFAULT_PROTOCOL,
) -> dict[str, float]:
    """Compute AWD/SCS using the fixed upstream source behavior."""

    estimated_map = _official_voxel_map(estimated_points, protocol.voxel_size_m)
    reference_map = _official_voxel_map(reference_points, protocol.voxel_size_m)
    distances: dict[tuple[int, int, int], float] = {}
    for index, (est_mu, est_sigma, est_count) in estimated_map.items():
        reference = reference_map.get(index)
        if reference is None:
            continue
        ref_mu, ref_sigma, ref_count = reference
        if est_count < protocol.min_voxel_points or ref_count < protocol.min_voxel_points:
            continue
        distances[index] = _official_wasserstein_distance(
            ref_mu,
            ref_sigma,
            ref_count,
            est_mu,
            est_sigma,
            est_count,
        )

    if not distances:
        return {
            "awd_m": float("nan"),
            "scs": float("nan"),
            "n_awd_voxels": 0.0,
            "n_scs_voxels": 0.0,
        }

    awd = float(np.mean(list(distances.values())))
    scs_terms: list[float] = []
    for index in distances:
        neighbors = [distances[key] for key in _neighbor_keys(index, protocol.neighbor_radius) if key in distances]
        if not neighbors:
            continue
        mean_neighbor = float(np.mean(neighbors))
        if mean_neighbor == 0.0:
            scs_terms.append(float("nan"))
        else:
            scs_terms.append(float(np.std(neighbors) / mean_neighbor))
    return {
        "awd_m": awd,
        "scs": float(np.mean(scs_terms)) if scs_terms else float("nan"),
        "n_awd_voxels": float(len(distances)),
        "n_scs_voxels": float(len(scs_terms)),
    }


def _rss_bytes(who: int) -> int | None:
    try:
        value = int(resource.getrusage(who).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    # Linux and the other common Unix implementations expose KiB here; macOS
    # exposes bytes.  The runner is Linux-first but records the method below.
    return value * 1024 if sys.platform != "darwin" else value


def _measure(callable_: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    before = _rss_bytes(resource.RUSAGE_SELF)
    started = time.perf_counter()
    value = callable_()
    elapsed = time.perf_counter() - started
    after = _rss_bytes(resource.RUSAGE_SELF)
    peak = after
    delta = None if before is None or after is None else max(0, after - before)
    return value, {
        "elapsed_seconds": float(elapsed),
        "peak_rss_bytes": peak,
        "peak_rss_delta_bytes": delta,
        "memory_measurement": "resource.RUSAGE_SELF.ru_maxrss high-water mark",
    }


def _read_proc_rss(pid: int) -> int | None:
    status_path = Path(f"/proc/{pid}/status")
    try:
        text = status_path.read_text(encoding="ascii")
    except (FileNotFoundError, OSError):
        return None
    for field in ("VmHWM", "VmRSS"):
        match = re.search(rf"^{field}:\s+(\d+)\s+kB$", text, re.MULTILINE)
        if match:
            return int(match.group(1)) * 1024
    return None


def _run_external(
    executable: Path,
    *,
    cwd: Path,
) -> tuple[int, str, str, dict[str, Any]]:
    started = time.perf_counter()
    children_before = _rss_bytes(resource.RUSAGE_CHILDREN)
    process = subprocess.Popen(
        [str(executable)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak = 0
    while process.poll() is None:
        current = _read_proc_rss(process.pid)
        if current is not None:
            peak = max(peak, current)
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    current = _read_proc_rss(process.pid)
    if current is not None:
        peak = max(peak, current)
    children_after = _rss_bytes(resource.RUSAGE_CHILDREN)
    if children_after is not None:
        peak = max(peak, children_after)
    return process.returncode, stdout, stderr, {
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_rss_bytes": peak or None,
        "peak_rss_delta_bytes": (
            None
            if children_before is None or children_after is None
            else max(0, children_after - children_before)
        ),
        "memory_measurement": "child /proc VmHWM with RUSAGE_CHILDREN fallback",
    }


def _git_metadata(upstream_dir: str | Path | None) -> dict[str, Any]:
    if upstream_dir is None:
        return {
            "directory": None,
            "checked_out_commit": None,
            "matches_fixed_commit": None,
            "dirty": None,
        }
    directory = Path(upstream_dir).resolve()
    if not directory.exists():
        return {
            "directory": str(directory),
            "checked_out_commit": None,
            "matches_fixed_commit": False,
            "dirty": None,
            "error": "upstream directory does not exist",
        }
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        dirty = subprocess.run(
            ["git", "-C", str(directory), "diff", "--quiet"],
            check=False,
        ).returncode != 0
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "directory": str(directory),
            "checked_out_commit": None,
            "matches_fixed_commit": False,
            "dirty": None,
            "error": str(exc),
        }
    return {
        "directory": str(directory),
        "checked_out_commit": commit,
        "matches_fixed_commit": commit == UPSTREAM_COMMIT,
        "dirty": dirty,
    }


def _write_official_config(
    path: Path,
    *,
    estimate_dir: Path,
    reference_path: Path,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "registration_methods": 2,
        "icp_max_distance": 1.0,
        "accuracy_level": [0.2, 0.1, 0.08, 0.05, 0.01],
        "initial_matrix": np.eye(4).tolist(),
        "save_immediate_result": True,
        "evaluate_mme": False,
        "use_tbb_mme": True,
        "evaluate_gt_mme": False,
        "nn_radius": 0.1,
        "estimate_map_path": str(estimate_dir) + os.sep,
        "gt_map_path": str(reference_path),
        "scene_name": "cloudanalyzer-parity",
        # Identity is intentional: it makes the AWD/SCS comparison independent
        # of the upstream ICP implementation and matches aligned map inputs.
        "evaluate_using_initial": True,
        "evaluate_noise_gt": False,
        "vmd_voxel_size": DEFAULT_PROTOCOL.voxel_size_m,
        "downsample_size": DEFAULT_DOWNSAMPLE_VOXEL_SIZE_M,
        "use_visualization": False,
        "enable_debug": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def _parse_official_result(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"official result file was not produced: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    vmd = re.findall(r"^VMD:\s*([-+0-9.eE]+)\s*$", text, re.MULTILINE)
    scs = re.findall(r"^SCS:\s*([-+0-9.eE]+)\s*$", text, re.MULTILINE)
    if not vmd or not scs:
        raise ValueError(f"official result file has no VMD/SCS lines: {path}")
    return {"awd_m": float(vmd[-1]), "scs": float(scs[-1])}


def _compare(
    expected: dict[str, float],
    actual: dict[str, float],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    passed = True
    for name in ("awd_m", "scs"):
        reference = float(expected[name])
        candidate = float(actual[name])
        finite = np.isfinite(reference) and np.isfinite(candidate)
        difference = abs(candidate - reference) if finite else float("nan")
        allowed = absolute_tolerance + relative_tolerance * abs(reference)
        metric_passed = bool(finite and difference <= allowed)
        passed &= metric_passed
        checks[name] = {
            "expected": reference,
            "actual": candidate,
            "absolute_difference": difference,
            "allowed_difference": allowed,
            "passed": metric_passed,
        }
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "metrics": checks,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, Path):
        return str(value)
    return value


def run_parity_harness(
    output_dir: str | Path,
    *,
    official_executable: str | Path | None = None,
    upstream_dir: str | Path | None = None,
    fixture: MapEvalParityFixture | None = None,
    absolute_tolerance: float = DEFAULT_ABS_TOLERANCE,
    relative_tolerance: float = DEFAULT_REL_TOLERANCE,
) -> dict[str, Any]:
    """Run the local lane and optionally an external upstream executable.

    Without ``official_executable`` this is a successful, CI-safe smoke run;
    the upstream comparison is marked ``not_run`` instead of being guessed.
    Supplying an executable makes a metric mismatch fail the returned report's
    ``comparison.passed`` field and the CLI exits non-zero.
    """

    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    selected_fixture = fixture or make_mapeval_smoke_fixture()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixture_paths = write_fixture(selected_fixture, output / "fixture")

    protocol = DEFAULT_PROTOCOL
    estimated_downsampled = voxel_downsample(
        selected_fixture.estimated_points,
        DEFAULT_DOWNSAMPLE_VOXEL_SIZE_M,
    )
    reference_downsampled = voxel_downsample(
        selected_fixture.reference_points,
        DEFAULT_DOWNSAMPLE_VOXEL_SIZE_M,
    )
    cloudanalyzer_metrics, cloudanalyzer_runtime = _measure(
        lambda: compute_voxel_wasserstein_metrics(
            estimated_downsampled,
            reference_downsampled,
            voxel_size=protocol.voxel_size_m,
            min_voxel_points=protocol.min_voxel_points,
            neighbor_radius=protocol.neighbor_radius,
        )
    )
    compatibility_metrics, compatibility_runtime = _measure(
        lambda: compute_official_compatible_metrics(
            estimated_downsampled,
            reference_downsampled,
            protocol=protocol,
        )
    )

    report: dict[str, Any] = {
        "schema_version": PARITY_REPORT_SCHEMA_VERSION,
        "status": "smoke_pass",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "license": UPSTREAM_LICENSE,
            "dependencies": UPSTREAM_DEPENDENCIES,
            "source_checkout": _git_metadata(upstream_dir),
        },
        "runtime": {
            "cloudanalyzer_version": __version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "protocol": {
            **protocol.as_dict(),
            "downsample_voxel_size_m": DEFAULT_DOWNSAMPLE_VOXEL_SIZE_M,
            "official_evaluate_using_initial": True,
            "official_registration_method": "GICP (skipped by evaluate_using_initial)",
        },
        "fixture": {
            **selected_fixture.as_dict(),
            "paths": {name: str(path) for name, path in fixture_paths.items()},
            "sha256": {name: sha256_file(path) for name, path in fixture_paths.items()},
            "downsampled_point_count": {
                "estimated": int(estimated_downsampled.shape[0]),
                "reference": int(reference_downsampled.shape[0]),
            },
        },
        "lanes": {
            "cloudanalyzer_corrected": {
                "description": "Public core lane with symmetric Bures matrix square root.",
                "metrics": cloudanalyzer_metrics,
                "runtime": cloudanalyzer_runtime,
            },
            "official_compatibility": {
                "description": "Python port of the fixed upstream source behavior.",
                "metrics": compatibility_metrics,
                "runtime": compatibility_runtime,
            },
        },
        "known_implementation_differences": [
            "Upstream Open3D voxel covariance is normalized in buildVoxelMap, computeVoxelEntropy, and the Gaussian distance path.",
            "Upstream computeWassersteinDistanceGaussian uses Cholesky factors; CloudAnalyzer core uses the symmetric Bures square root.",
        ],
        "official_execution": {
            "status": "not_run",
            "executable": None,
            "config": None,
            "result_file": None,
        },
        "comparison": {
            "status": "not_run",
            "passed": None,
            "reference_lane": "official_compatibility",
            "candidate_lane": "official_executable",
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
        },
    }

    if official_executable is not None:
        executable = Path(official_executable).expanduser().resolve()
        official_root = output / "official-run"
        official_build = official_root / "build"
        official_estimated = official_root / "estimated"
        official_build.mkdir(parents=True, exist_ok=True)
        official_estimated.mkdir(parents=True, exist_ok=True)
        official_estimated_map = official_estimated / "map.pcd"
        shutil.copy2(fixture_paths["estimated"], official_estimated_map)
        config_path = official_root / "config" / "config.yaml"
        config = _write_official_config(
            config_path,
            estimate_dir=official_estimated,
            reference_path=fixture_paths["reference"],
        )
        result_file = official_estimated / "map_results" / "map_results.txt"
        if result_file.exists():
            result_file.unlink()
        execution: dict[str, Any] = {
            "status": "unavailable",
            "executable": str(executable),
            "config": str(config_path),
            "result_file": str(result_file),
            "config_values": config,
        }
        if not executable.is_file() or not os.access(executable, os.X_OK):
            execution["reason"] = "official executable is missing or not executable"
            report["official_execution"] = execution
            report["comparison"] = {
                "status": "unavailable",
                "passed": None,
                "reference_lane": "official_compatibility",
                "candidate_lane": "official_executable",
                "reason": execution["reason"],
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": relative_tolerance,
            }
            report["status"] = "official_unavailable"
        else:
            returncode, stdout, stderr, runtime = _run_external(executable, cwd=official_build)
            execution.update(
                {
                    "status": "completed" if returncode == 0 else "error",
                    "returncode": returncode,
                    "runtime": runtime,
                    "stdout_tail": stdout[-4000:],
                    "stderr_tail": stderr[-4000:],
                }
            )
            report["official_execution"] = execution
            if returncode != 0:
                report["comparison"] = {
                    "status": "error",
                    "passed": False,
                    "reference_lane": "official_compatibility",
                    "candidate_lane": "official_executable",
                    "reason": f"official executable returned {returncode}",
                    "absolute_tolerance": absolute_tolerance,
                    "relative_tolerance": relative_tolerance,
                }
                report["status"] = "official_error"
            else:
                try:
                    official_metrics = _parse_official_result(result_file)
                    report["official_execution"]["metrics"] = official_metrics
                    comparison = _compare(
                        compatibility_metrics,
                        official_metrics,
                        absolute_tolerance=absolute_tolerance,
                        relative_tolerance=relative_tolerance,
                    )
                    comparison["reference_lane"] = "official_compatibility"
                    comparison["candidate_lane"] = "official_executable"
                    report["comparison"] = comparison
                    report["status"] = "parity_pass" if comparison["passed"] else "parity_fail"
                except (FileNotFoundError, ValueError) as exc:
                    report["comparison"] = {
                        "status": "error",
                        "passed": False,
                        "reference_lane": "official_compatibility",
                        "candidate_lane": "official_executable",
                        "reason": str(exc),
                        "absolute_tolerance": absolute_tolerance,
                        "relative_tolerance": relative_tolerance,
                    }
                    report["status"] = "official_error"

    report = _json_safe(report)
    (output / "mapeval_parity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "DEFAULT_ABS_TOLERANCE",
    "DEFAULT_DOWNSAMPLE_VOXEL_SIZE_M",
    "DEFAULT_PROTOCOL",
    "DEFAULT_REL_TOLERANCE",
    "MapEvalParityFixture",
    "PARITY_REPORT_SCHEMA_VERSION",
    "UPSTREAM_COMMIT",
    "UPSTREAM_DEPENDENCIES",
    "UPSTREAM_LICENSE",
    "UPSTREAM_REPOSITORY",
    "compute_official_compatible_metrics",
    "make_mapeval_smoke_fixture",
    "run_parity_harness",
    "sha256_file",
    "write_ascii_pcd",
    "write_fixture",
]
