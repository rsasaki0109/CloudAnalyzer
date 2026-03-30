"""Combined run evaluation for map + trajectory outputs."""

from __future__ import annotations

import shlex
from pathlib import Path

from ca.evaluate import evaluate
from ca.io import SUPPORTED_EXTENSIONS
from ca.log import logger
from ca.trajectory import SUPPORTED_TRAJECTORY_EXTENSIONS, evaluate_trajectory


def _map_quality_gate(
    auc: float,
    chamfer_distance: float,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
) -> dict | None:
    """Build optional map quality gate metadata."""
    if min_auc is None and max_chamfer is None:
        return None

    reasons: list[str] = []
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


def _map_inspection_commands(source_path: str, reference_path: str) -> dict[str, str]:
    """Build follow-up commands for map inspection."""
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
    rigid_report_name = shlex.quote(
        f"{source_stem}_vs_{reference_stem}_trajectory_rigid_report.html"
    )
    return {
        "traj_evaluate": f"ca traj-evaluate {source} {reference} --report {report_name}",
        "traj_evaluate_aligned": (
            f"ca traj-evaluate {source} {reference} --align-origin --report {aligned_report_name}"
        ),
        "traj_evaluate_rigid": (
            f"ca traj-evaluate {source} {reference} --align-rigid --report {rigid_report_name}"
        ),
    }


def _run_inspection_command(
    map_path: str,
    map_reference_path: str,
    trajectory_path: str,
    trajectory_reference_path: str,
    thresholds: list[float] | None = None,
    max_time_delta: float = 0.05,
    align_origin: bool = False,
    align_rigid: bool = False,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> str:
    """Build a reproducible combined run-evaluate command."""
    map_stem = Path(map_path).stem
    trajectory_stem = Path(trajectory_path).stem
    report_name = f"{map_stem}_and_{trajectory_stem}_run_report.html"
    args = [
        "ca",
        "run-evaluate",
        map_path,
        map_reference_path,
        trajectory_path,
        trajectory_reference_path,
    ]
    if thresholds:
        args.extend(["--thresholds", ",".join(str(value) for value in thresholds)])
    if max_time_delta != 0.05:
        args.extend(["--max-time-delta", f"{max_time_delta}"])
    if align_origin:
        args.append("--align-origin")
    if align_rigid:
        args.append("--align-rigid")
    if min_auc is not None:
        args.extend(["--min-auc", f"{min_auc}"])
    if max_chamfer is not None:
        args.extend(["--max-chamfer", f"{max_chamfer}"])
    if max_ate is not None:
        args.extend(["--max-ate", f"{max_ate}"])
    if max_rpe is not None:
        args.extend(["--max-rpe", f"{max_rpe}"])
    if max_drift is not None:
        args.extend(["--max-drift", f"{max_drift}"])
    if min_coverage is not None:
        args.extend(["--min-coverage", f"{min_coverage}"])
    args.extend(["--report", report_name])
    return " ".join(shlex.quote(arg) for arg in args)


def _run_web_inspection_command(
    map_path: str,
    map_reference_path: str,
    trajectory_path: str,
    trajectory_reference_path: str,
    max_time_delta: float = 0.05,
    align_origin: bool = False,
    align_rigid: bool = False,
) -> str:
    """Build a combined web viewer command for one run."""
    args = [
        "ca",
        "web",
        map_path,
        map_reference_path,
        "--heatmap",
        "--trajectory",
        trajectory_path,
        "--trajectory-reference",
        trajectory_reference_path,
    ]
    if max_time_delta != 0.05:
        args.extend(["--trajectory-max-time-delta", f"{max_time_delta}"])
    if align_origin:
        args.append("--trajectory-align-origin")
    if align_rigid:
        args.append("--trajectory-align-rigid")
    return " ".join(shlex.quote(arg) for arg in args)


def _overall_quality_gate(
    map_gate: dict | None,
    trajectory_gate: dict | None,
) -> dict | None:
    """Build an overall pass/fail decision for combined map + trajectory evaluation."""
    if map_gate is None and trajectory_gate is None:
        return None

    reasons: list[str] = []
    if map_gate is not None and not map_gate["passed"]:
        reasons.extend(f"Map: {reason}" for reason in map_gate["reasons"])
    if trajectory_gate is not None and not trajectory_gate["passed"]:
        reasons.extend(f"Trajectory: {reason}" for reason in trajectory_gate["reasons"])

    return {
        "passed": not reasons,
        "map_passed": None if map_gate is None else map_gate["passed"],
        "trajectory_passed": None if trajectory_gate is None else trajectory_gate["passed"],
        "reasons": reasons,
    }


def _find_point_cloud_files(directory: str, recursive: bool = False) -> list[Path]:
    """Find supported point cloud files in a directory."""
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in directory_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _find_trajectory_files(directory: str, recursive: bool = False) -> list[Path]:
    """Find supported trajectory files in a directory."""
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in directory_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_TRAJECTORY_EXTENSIONS
    )


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


def evaluate_run(
    map_path: str,
    map_reference_path: str,
    trajectory_path: str,
    trajectory_reference_path: str,
    thresholds: list[float] | None = None,
    max_time_delta: float = 0.05,
    align_origin: bool = False,
    align_rigid: bool = False,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> dict:
    """Evaluate one map output and one trajectory output together."""
    map_result = evaluate(map_path, map_reference_path, thresholds=thresholds)
    map_result["best_f1"] = max(map_result["f1_scores"], key=lambda score: score["f1"])
    map_result["quality_gate"] = _map_quality_gate(
        map_result["auc"],
        map_result["chamfer_distance"],
        min_auc=min_auc,
        max_chamfer=max_chamfer,
    )

    trajectory_result = evaluate_trajectory(
        trajectory_path,
        trajectory_reference_path,
        max_time_delta=max_time_delta,
        align_origin=align_origin,
        align_rigid=align_rigid,
        max_ate=max_ate,
        max_rpe=max_rpe,
        max_drift=max_drift,
        min_coverage=min_coverage,
    )

    return {
        "map": map_result,
        "trajectory": trajectory_result,
        "overall_quality_gate": _overall_quality_gate(
            map_result["quality_gate"],
            trajectory_result["quality_gate"],
        ),
        "inspect": {
            "run_evaluate": _run_inspection_command(
                map_path,
                map_reference_path,
                trajectory_path,
                trajectory_reference_path,
                thresholds=thresholds,
                max_time_delta=max_time_delta,
                align_origin=align_origin,
                align_rigid=align_rigid,
                min_auc=min_auc,
                max_chamfer=max_chamfer,
                max_ate=max_ate,
                max_rpe=max_rpe,
                max_drift=max_drift,
                min_coverage=min_coverage,
            ),
            "run_web": _run_web_inspection_command(
                map_path,
                map_reference_path,
                trajectory_path,
                trajectory_reference_path,
                max_time_delta=max_time_delta,
                align_origin=align_origin,
                align_rigid=align_rigid,
            ),
            "map": _map_inspection_commands(map_path, map_reference_path),
            "trajectory": _trajectory_inspection_commands(
                trajectory_path,
                trajectory_reference_path,
            ),
        },
    }


def evaluate_run_batch(
    map_dir: str,
    map_reference_dir: str,
    trajectory_dir: str,
    trajectory_reference_dir: str,
    recursive: bool = False,
    thresholds: list[float] | None = None,
    max_time_delta: float = 0.05,
    align_origin: bool = False,
    align_rigid: bool = False,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> list[dict]:
    """Evaluate multiple runs by matching map and trajectory artifacts."""
    map_files = _find_point_cloud_files(map_dir, recursive=recursive)
    if not map_files:
        return []

    map_root = Path(map_dir)
    map_reference_root = Path(map_reference_dir)
    trajectory_root = Path(trajectory_dir)
    trajectory_reference_root = Path(trajectory_reference_dir)

    map_reference_files = _find_point_cloud_files(map_reference_dir, recursive=recursive)
    trajectory_files = _find_trajectory_files(trajectory_dir, recursive=recursive)
    trajectory_reference_files = _find_trajectory_files(
        trajectory_reference_dir,
        recursive=recursive,
    )

    results = []
    for map_path in map_files:
        map_reference_path = _match_by_relative_or_stem(
            map_path,
            map_root,
            map_reference_root,
            map_reference_files,
        )
        if map_reference_path is None:
            logger.warning("Skipped %s: no matched reference map in %s", map_path, map_reference_dir)
            continue

        trajectory_path = _match_by_relative_or_stem(
            map_path,
            map_root,
            trajectory_root,
            trajectory_files,
        )
        if trajectory_path is None:
            logger.warning("Skipped %s: no matched trajectory in %s", map_path, trajectory_dir)
            continue

        trajectory_reference_path = _match_by_relative_or_stem(
            trajectory_path,
            trajectory_root,
            trajectory_reference_root,
            trajectory_reference_files,
        )
        if trajectory_reference_path is None:
            logger.warning(
                "Skipped %s: no matched reference trajectory in %s",
                trajectory_path,
                trajectory_reference_dir,
            )
            continue

        logger.debug("Evaluating run: %s", map_path)
        try:
            result = evaluate_run(
                str(map_path),
                str(map_reference_path),
                str(trajectory_path),
                str(trajectory_reference_path),
                thresholds=thresholds,
                max_time_delta=max_time_delta,
                align_origin=align_origin,
                align_rigid=align_rigid,
                min_auc=min_auc,
                max_chamfer=max_chamfer,
                max_ate=max_ate,
                max_rpe=max_rpe,
                max_drift=max_drift,
                min_coverage=min_coverage,
            )
        except (FileNotFoundError, ValueError) as error:
            logger.warning("Skipped %s: %s", map_path, error)
            continue

        relative_id = map_path.relative_to(map_root).with_suffix("").as_posix()
        results.append(
            {
                "id": relative_id,
                "map_path": str(map_path),
                "map_reference_path": str(map_reference_path),
                "trajectory_path": str(trajectory_path),
                "trajectory_reference_path": str(trajectory_reference_path),
                "map": result["map"],
                "trajectory": result["trajectory"],
                "overall_quality_gate": result["overall_quality_gate"],
                "inspect": result["inspect"],
            }
        )

    return results
