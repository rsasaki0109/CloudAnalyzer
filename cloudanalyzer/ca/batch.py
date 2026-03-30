"""Batch processing module."""

import shlex
from pathlib import Path

from ca.evaluate import evaluate
from ca.io import SUPPORTED_EXTENSIONS, load_point_cloud
from ca.info import get_info
from ca.log import logger
from ca.pareto import mark_quality_size_recommended
from ca.trajectory import SUPPORTED_TRAJECTORY_EXTENSIONS, evaluate_trajectory


def _quality_gate_result(
    auc: float,
    chamfer_distance: float,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
) -> dict | None:
    """Build pass/fail metadata for optional batch quality gates."""
    if min_auc is None and max_chamfer is None:
        return None

    reasons = []
    if min_auc is not None and auc < min_auc:
        reasons.append(f"AUC {auc:.4f} < min_auc {min_auc:.4f}")
    if max_chamfer is not None and chamfer_distance > max_chamfer:
        reasons.append(
            f"Chamfer {chamfer_distance:.4f} > max_chamfer {max_chamfer:.4f}"
        )

    return {
        "passed": not reasons,
        "min_auc": min_auc,
        "max_chamfer": max_chamfer,
        "reasons": reasons,
    }


def _find_point_cloud_files(directory: str, recursive: bool = False) -> list[Path]:
    """Find supported point cloud files in a directory."""
    dirpath = Path(directory)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in dirpath.glob(pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No point cloud files found in %s", directory)
        return []

    logger.info("Found %d point cloud file(s) in %s", len(files), directory)
    return files


def _find_trajectory_files(directory: str, recursive: bool = False) -> list[Path]:
    """Find supported trajectory files in a directory."""
    dirpath = Path(directory)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in dirpath.glob(pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_TRAJECTORY_EXTENSIONS
    )

    if not files:
        logger.warning("No trajectory files found in %s", directory)
        return []

    logger.info("Found %d trajectory file(s) in %s", len(files), directory)
    return files


def _find_files(directory: str, recursive: bool = False) -> list[Path]:
    """Find all files in a directory."""
    dirpath = Path(directory)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = "**/*" if recursive else "*"
    return sorted(f for f in dirpath.glob(pattern) if f.is_file())


def _match_by_relative_or_stem(
    source_path: Path,
    source_root: Path,
    target_root: Path,
    target_files: list[Path],
) -> Path | None:
    """Match a related artifact file by relative path or unique stem."""
    relative = source_path.relative_to(source_root)
    exact_path = target_root / relative
    if exact_path.is_file():
        return exact_path

    relative_without_ext = relative.with_suffix("").as_posix()
    same_relative = [
        path for path in target_files
        if path.relative_to(target_root).with_suffix("").as_posix() == relative_without_ext
    ]
    if len(same_relative) == 1:
        return same_relative[0]
    if len(same_relative) > 1:
        logger.warning("Ambiguous artifact match for %s in %s", source_path, target_root)
        return None

    same_stem = [path for path in target_files if path.stem == source_path.stem]
    if len(same_stem) == 1:
        return same_stem[0]
    if len(same_stem) > 1:
        logger.warning("Ambiguous stem match for %s in %s", source_path, target_root)
    return None


def _compression_stats(
    source_path: Path,
    source_root: Path,
    compressed_root: Path | None = None,
    compressed_files: list[Path] | None = None,
    baseline_root: Path | None = None,
    baseline_files: list[Path] | None = None,
) -> dict | None:
    """Collect optional compression-related file size metadata."""
    source_size = source_path.stat().st_size
    baseline_path = source_path
    baseline_size = source_size
    compressed_path = None
    compressed_size = None

    if baseline_root is not None and baseline_files is not None:
        matched_baseline = _match_by_relative_or_stem(
            source_path,
            source_root,
            baseline_root,
            baseline_files,
        )
        if matched_baseline is not None:
            baseline_path = matched_baseline
            baseline_size = matched_baseline.stat().st_size

    if compressed_root is None or compressed_files is None:
        return None

    matched_compressed = _match_by_relative_or_stem(
        source_path,
        source_root,
        compressed_root,
        compressed_files,
    )
    if matched_compressed is None:
        return None

    compressed_path = matched_compressed
    compressed_size = matched_compressed.stat().st_size
    size_ratio = compressed_size / baseline_size if baseline_size > 0 else 0.0

    return {
        "source_size_bytes": source_size,
        "baseline_path": str(baseline_path),
        "baseline_size_bytes": baseline_size,
        "compressed_path": str(compressed_path),
        "compressed_size_bytes": compressed_size,
        "size_ratio": size_ratio,
        "space_saving_ratio": 1.0 - size_ratio,
    }


def _inspection_commands(source_path: str, reference_path: str) -> dict[str, str]:
    """Build follow-up commands for interactive inspection."""
    source = shlex.quote(source_path)
    reference = shlex.quote(reference_path)
    source_stem = Path(source_path).stem
    reference_stem = Path(reference_path).stem
    snapshot_name = f"{source_stem}_vs_{reference_stem}_heatmap.png"
    return {
        "web_heatmap": f"ca web {source} {reference} --heatmap",
        "heatmap3d": f"ca heatmap3d {source} {reference} -o {shlex.quote(snapshot_name)}",
    }


def _trajectory_inspection_commands(source_path: str, reference_path: str) -> dict[str, str]:
    """Build follow-up commands for trajectory inspection."""
    source = shlex.quote(source_path)
    reference = shlex.quote(reference_path)
    source_stem = Path(source_path).stem
    reference_stem = Path(reference_path).stem
    report_name = shlex.quote(f"{source_stem}_vs_{reference_stem}_trajectory_report.html")
    aligned_report_name = shlex.quote(
        f"{source_stem}_vs_{reference_stem}_trajectory_aligned_report.html"
    )
    return {
        "traj_evaluate": f"ca traj-evaluate {source} {reference} --report {report_name}",
        "traj_evaluate_aligned": (
            f"ca traj-evaluate {source} {reference} --align-origin --report {aligned_report_name}"
        ),
        "traj_evaluate_rigid": (
            f"ca traj-evaluate {source} {reference} --align-rigid --report "
            f"{shlex.quote(f'{source_stem}_vs_{reference_stem}_trajectory_rigid_report.html')}"
        ),
    }


def batch_info(directory: str, recursive: bool = False) -> list[dict]:
    """Run info on all point cloud files in a directory.

    Args:
        directory: Directory path to scan.
        recursive: If True, scan subdirectories too.

    Returns:
        List of info dicts for each file found.
    """
    files = _find_point_cloud_files(directory, recursive=recursive)
    if not files:
        return []

    results = []
    for f in files:
        logger.debug("Processing: %s", f)
        try:
            info = get_info(str(f))
            results.append(info)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipped %s: %s", f, e)

    return results


def batch_evaluate(
    directory: str,
    reference_path: str,
    recursive: bool = False,
    thresholds: list[float] | None = None,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    compressed_dir: str | None = None,
    baseline_dir: str | None = None,
) -> list[dict]:
    """Evaluate all point cloud files in a directory against one reference.

    Args:
        directory: Directory path to scan.
        reference_path: Reference point cloud path.
        recursive: If True, scan subdirectories too.
        thresholds: Optional list of F1 thresholds.
        min_auc: Optional minimum AUC required to pass.
        max_chamfer: Optional maximum Chamfer distance required to pass.
        compressed_dir: Optional directory with compressed artifacts.
        baseline_dir: Optional directory with original uncompressed artifacts.

    Returns:
        List of evaluation summary dicts for each file found.
    """
    files = _find_point_cloud_files(directory, recursive=recursive)
    if not files:
        return []
    source_root = Path(directory)

    # Validate the reference once so we fail fast on bad input.
    load_point_cloud(reference_path)
    compressed_root = Path(compressed_dir) if compressed_dir is not None else None
    compressed_files = (
        _find_files(compressed_dir, recursive=recursive)
        if compressed_dir is not None else None
    )
    baseline_root = Path(baseline_dir) if baseline_dir is not None else None
    baseline_files = (
        _find_files(baseline_dir, recursive=recursive)
        if baseline_dir is not None else None
    )

    results = []
    for f in files:
        logger.debug("Evaluating: %s", f)
        try:
            eval_result = evaluate(str(f), reference_path, thresholds=thresholds)
            best_f1 = max(eval_result["f1_scores"], key=lambda score: score["f1"])
            quality_gate = _quality_gate_result(
                eval_result["auc"],
                eval_result["chamfer_distance"],
                min_auc=min_auc,
                max_chamfer=max_chamfer,
            )
            compression = _compression_stats(
                f,
                source_root,
                compressed_root=compressed_root,
                compressed_files=compressed_files,
                baseline_root=baseline_root,
                baseline_files=baseline_files,
            )
            results.append(
                {
                    "path": str(f),
                    "num_points": eval_result["source_points"],
                    "reference_path": reference_path,
                    "reference_points": eval_result["target_points"],
                    "chamfer_distance": eval_result["chamfer_distance"],
                    "hausdorff_distance": eval_result["hausdorff_distance"],
                    "auc": eval_result["auc"],
                    "best_f1": best_f1,
                    "f1_scores": eval_result["f1_scores"],
                    "quality_gate": quality_gate,
                    "inspect": _inspection_commands(str(f), reference_path),
                    "compression": compression,
                }
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipped %s: %s", f, e)

    mark_quality_size_recommended(
        results,
        min_auc=min_auc,
        max_chamfer=max_chamfer,
    )
    return results


def trajectory_batch_evaluate(
    directory: str,
    reference_dir: str,
    recursive: bool = False,
    max_time_delta: float = 0.05,
    align_origin: bool = False,
    align_rigid: bool = False,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> list[dict]:
    """Evaluate all trajectory files in a directory against matched references."""
    files = _find_trajectory_files(directory, recursive=recursive)
    if not files:
        return []

    source_root = Path(directory)
    reference_root = Path(reference_dir)
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Directory not found: {reference_dir}")
    reference_files = _find_trajectory_files(reference_dir, recursive=recursive)

    results = []
    for f in files:
        reference_path = _match_by_relative_or_stem(
            f,
            source_root,
            reference_root,
            reference_files,
        )
        if reference_path is None:
            logger.warning("Skipped %s: no matched reference trajectory in %s", f, reference_dir)
            continue

        logger.debug("Evaluating trajectory: %s", f)
        try:
            eval_result = evaluate_trajectory(
                str(f),
                str(reference_path),
                max_time_delta=max_time_delta,
                align_origin=align_origin,
                align_rigid=align_rigid,
                max_ate=max_ate,
                max_rpe=max_rpe,
                max_drift=max_drift,
                min_coverage=min_coverage,
            )
            results.append(
                {
                    "path": str(f),
                    "reference_path": str(reference_path),
                    "alignment": eval_result["alignment"],
                    "matching": eval_result["matching"],
                    "estimated_poses": eval_result["matching"]["estimated_poses"],
                    "reference_poses": eval_result["matching"]["reference_poses"],
                    "matched_poses": eval_result["matching"]["matched_poses"],
                    "coverage_ratio": eval_result["matching"]["coverage_ratio"],
                    "ate": eval_result["ate"],
                    "rpe_translation": eval_result["rpe_translation"],
                    "drift": eval_result["drift"],
                    "quality_gate": eval_result["quality_gate"],
                    "inspect": _trajectory_inspection_commands(str(f), str(reference_path)),
                }
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipped %s: %s", f, e)

    return results
