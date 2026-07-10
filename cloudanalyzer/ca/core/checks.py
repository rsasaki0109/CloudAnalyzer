"""Config-driven QA checks for artifact, trajectory, and integrated run validation."""

from __future__ import annotations

import json
import math
import shlex
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from ca.core.check_triage import summarize_failed_checks
from ca.core.gate import (
    GateMode,
    GateSeverity,
    gate_status_for_check,
    normalize_gate_severity,
    summarize_gate_policy,
)
from ca.batch import batch_evaluate, trajectory_batch_evaluate
from ca.detection import evaluate_detection
from ca.evaluate import evaluate
from ca.loop_closure_report import LoopClosureGate, build_loop_closure_report
from ca.posegraph import discover_session_paths
from ca.report import (
    make_batch_summary,
    make_run_batch_summary,
    make_trajectory_batch_summary,
    save_batch_report,
    save_detection_report,
    save_ground_report,
    save_image_report,
    save_rendered_report,
    save_run_batch_report,
    save_run_report,
    save_tracking_report,
    save_trajectory_batch_report,
    save_trajectory_report,
)
from ca.run_evaluate import evaluate_run, evaluate_run_batch
from ca.tracking import evaluate_tracking
from ca.trajectory import evaluate_trajectory


CheckKind = Literal[
    "artifact",
    "artifact_batch",
    "trajectory",
    "trajectory_batch",
    "detection",
    "tracking",
    "run",
    "run_batch",
    "ground",
    "loop_closure",
    "image",
    "rendered",
    "structure",
    "uncertainty",
]
AlignmentMode = Literal["none", "origin", "rigid"]

_KIND_ALIASES: dict[str, CheckKind] = {
    "artifact": "artifact",
    "map": "artifact",
    "artifact_batch": "artifact_batch",
    "map_batch": "artifact_batch",
    "trajectory": "trajectory",
    "trajectory_batch": "trajectory_batch",
    "detection": "detection",
    "object_detection": "detection",
    "tracking": "tracking",
    "object_tracking": "tracking",
    "run": "run",
    "run_batch": "run_batch",
    "ground": "ground",
    "ground_segmentation": "ground",
    "loop_closure": "loop_closure",
    "loop-closure": "loop_closure",
    "manual_loop_closure": "loop_closure",
    "image": "image",
    "image_evaluate": "image",
    "photometric": "image",
    "rendered": "rendered",
    "rendered_evaluate": "rendered",
    "3dgs": "rendered",
    "structure": "structure",
    "plane_consistency": "structure",
    "uncertainty": "uncertainty",
    "slam_uncertainty": "uncertainty",
}

_VALID_GATE_KEYS = {
    "min_auc",
    "max_chamfer",
    "max_ate",
    "max_rpe",
    "max_drift",
    "min_coverage",
    "min_precision",
    "min_recall",
    "min_f1",
    "min_map",
    "min_mota",
    "min_iou",
    "max_id_switches",
    "max_lateral",
    "max_longitudinal",
    "min_auc_gain",
    "max_after_chamfer",
    "min_ate_gain",
    "max_after_ate",
    "require_posegraph_ok",
    "voxel_size",
    "min_psnr",
    "min_ssim",
    "max_lpips",
    "max_dreamsim_distance",
    "max_awd",
    "max_scs",
    "max_plane_normal_dispersion",
    "max_coplanar_offset_rmse",
    "max_mean_position_nees",
    "min_normalized_mean_position_nees",
    "min_coverage_95",
    "severity",
}

_ALLOWED_GATE_KEYS: dict[CheckKind, set[str]] = {
    "artifact": {"min_auc", "max_chamfer", "max_awd", "max_scs", "voxel_size"},
    "artifact_batch": {"min_auc", "max_chamfer", "max_awd", "max_scs", "voxel_size"},
    "trajectory": {"max_ate", "max_rpe", "max_drift", "min_coverage", "max_lateral", "max_longitudinal"},
    "trajectory_batch": {"max_ate", "max_rpe", "max_drift", "min_coverage", "max_lateral", "max_longitudinal"},
    "detection": {"min_map", "min_precision", "min_recall", "min_f1"},
    "tracking": {"min_mota", "min_recall", "max_id_switches"},
    "run": {
        "min_auc",
        "max_chamfer",
        "max_ate",
        "max_rpe",
        "max_drift",
        "min_coverage",
        "max_lateral",
        "max_longitudinal",
    },
    "run_batch": {
        "min_auc",
        "max_chamfer",
        "max_ate",
        "max_rpe",
        "max_drift",
        "min_coverage",
        "max_lateral",
        "max_longitudinal",
    },
    "ground": {"min_precision", "min_recall", "min_f1", "min_iou", "voxel_size"},
    "loop_closure": {
        "min_auc_gain",
        "max_after_chamfer",
        "min_ate_gain",
        "max_after_ate",
        "require_posegraph_ok",
    },
    "image": {"min_psnr", "min_ssim", "max_dreamsim_distance"},
    "rendered": {"min_psnr", "min_ssim", "max_lpips", "max_dreamsim_distance", "min_auc", "max_chamfer"},
    "structure": {"max_plane_normal_dispersion", "max_coplanar_offset_rmse", "voxel_size"},
    "uncertainty": {"max_mean_position_nees", "min_normalized_mean_position_nees", "min_coverage_95"},
}


@dataclass(frozen=True)
class CheckOutputs:
    """Optional file outputs for one check."""

    report_path: str | None = None
    json_path: str | None = None


@dataclass(frozen=True)
class CheckSpec:
    """Normalized spec for one config-driven QA check."""

    check_id: str
    kind: CheckKind
    inputs: dict[str, str]
    thresholds: tuple[float, ...] | None = None
    max_time_delta: float = 0.05
    recursive: bool = False
    alignment: AlignmentMode = "none"
    gate: dict[str, Any] = field(default_factory=dict)
    severity: GateSeverity = "fail"
    compressed_dir: str | None = None
    baseline_dir: str | None = None
    outputs: CheckOutputs = field(default_factory=CheckOutputs)


@dataclass(frozen=True)
class CheckSuite:
    """Normalized config file ready for execution."""

    config_path: str
    project: str | None
    checks: tuple[CheckSpec, ...]
    summary_output_json: str | None = None


def _as_mapping(value: object, context: str) -> dict[str, Any]:
    """Validate that a loaded config fragment is a mapping."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return cast(dict[str, Any], value)


def _as_list(value: object, context: str) -> list[Any]:
    """Validate that a loaded config fragment is a list."""
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return cast(list[Any], value)


def _require_string(value: object, context: str) -> str:
    """Validate a required string field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: object, context: str) -> str | None:
    """Validate an optional string field."""
    if value is None:
        return None
    return _require_string(value, context)


def _optional_bool(value: object, context: str, default: bool = False) -> bool:
    """Validate an optional boolean field."""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be true or false")
    return value


def _resolve_path(config_dir: Path, raw_path: str) -> str:
    """Resolve a config-relative path to an absolute path."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def _resolve_optional_path(config_dir: Path, raw_path: object, context: str) -> str | None:
    """Resolve an optional config-relative path."""
    path = _optional_string(raw_path, context)
    if path is None:
        return None
    return _resolve_path(config_dir, path)


def _normalize_thresholds(value: object, context: str) -> tuple[float, ...] | None:
    """Normalize optional threshold lists."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list of numbers")
    thresholds = tuple(sorted(float(item) for item in value))
    return thresholds


def _normalize_alignment(value: object, context: str, default: AlignmentMode = "none") -> AlignmentMode:
    """Normalize alignment mode."""
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{context} must be one of: none, origin, rigid")
    normalized = value.strip().lower()
    if normalized not in {"none", "origin", "rigid"}:
        raise ValueError(f"{context} must be one of: none, origin, rigid")
    return cast(AlignmentMode, normalized)


def _alignment_flags(mode: AlignmentMode) -> tuple[bool, bool]:
    """Convert alignment mode into evaluate_trajectory / evaluate_run flags."""
    return mode == "origin", mode == "rigid"


def _normalize_kind(raw_kind: object) -> CheckKind:
    """Normalize user-facing kind aliases into canonical kinds."""
    kind = _require_string(raw_kind, "check.kind").strip().lower()
    if kind not in _KIND_ALIASES:
        allowed = ", ".join(sorted(_KIND_ALIASES))
        raise ValueError(f"Unsupported check.kind '{kind}'. Use one of: {allowed}")
    return _KIND_ALIASES[kind]


def _normalize_gate(
    defaults: dict[str, Any],
    raw_check: dict[str, Any],
    kind: CheckKind,
) -> dict[str, Any]:
    """Merge top-level and per-check gate settings."""
    defaults_gate = _as_mapping(defaults.get("gate"), "defaults.gate")
    check_gate = _as_mapping(raw_check.get("gate"), "check.gate")
    merged = {**defaults_gate, **check_gate}
    unknown = set(merged) - _VALID_GATE_KEYS
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported gate key(s): {joined}")
    allowed = _ALLOWED_GATE_KEYS[kind]
    normalized: dict[str, Any] = {}
    for key, value in merged.items():
        if key not in allowed or value is None:
            continue
        if key == "require_posegraph_ok":
            normalized[key] = _optional_bool(value, f"gate.{key}")
        else:
            normalized[key] = float(value)
            if key == "voxel_size" and normalized[key] <= 0:
                raise ValueError("gate.voxel_size must be > 0")
    return normalized


def _normalize_severity(defaults: dict[str, Any], raw_check: dict[str, Any]) -> GateSeverity:
    """Normalize check severity from defaults/check/gate blocks."""
    defaults_gate = _as_mapping(defaults.get("gate"), "defaults.gate")
    check_gate = _as_mapping(raw_check.get("gate"), "check.gate")
    raw = raw_check.get(
        "severity",
        check_gate.get(
            "severity",
            defaults.get("severity", defaults_gate.get("severity")),
        ),
    )
    return normalize_gate_severity(raw, "check.severity")


def _default_output_paths(
    check_id: str,
    defaults: dict[str, Any],
    config_dir: Path,
) -> tuple[str | None, str | None]:
    """Resolve optional default output paths for reports and JSON dumps."""
    report_dir = _resolve_optional_path(config_dir, defaults.get("report_dir"), "defaults.report_dir")
    json_dir = _resolve_optional_path(config_dir, defaults.get("json_dir"), "defaults.json_dir")
    report_path = str((Path(report_dir) / f"{check_id}.html").resolve()) if report_dir else None
    json_path = str((Path(json_dir) / f"{check_id}.json").resolve()) if json_dir else None
    return report_path, json_path


def _normalize_outputs(
    raw_check: dict[str, Any],
    defaults: dict[str, Any],
    config_dir: Path,
    check_id: str,
) -> CheckOutputs:
    """Normalize optional report / JSON outputs for one check."""
    raw_outputs = _as_mapping(raw_check.get("outputs"), "check.outputs")
    default_report, default_json = _default_output_paths(check_id, defaults, config_dir)
    report_path = _resolve_optional_path(
        config_dir,
        raw_outputs.get("report", raw_check.get("report")),
        "check.report",
    )
    json_path = _resolve_optional_path(
        config_dir,
        raw_outputs.get("json", raw_outputs.get("output_json", raw_check.get("output_json"))),
        "check.output_json",
    )
    return CheckOutputs(
        report_path=report_path or default_report,
        json_path=json_path or default_json,
    )


def _normalize_artifact_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a single artifact check."""
    source = raw_check.get("source", raw_check.get("path", raw_check.get("artifact")))
    reference = raw_check.get("reference")
    return {
        "source": _resolve_path(config_dir, _require_string(source, "check.source")),
        "reference": _resolve_path(config_dir, _require_string(reference, "check.reference")),
    }


def _normalize_artifact_batch_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for an artifact batch check."""
    directory = raw_check.get("directory", raw_check.get("dir"))
    reference = raw_check.get("reference")
    return {
        "directory": _resolve_path(config_dir, _require_string(directory, "check.directory")),
        "reference": _resolve_path(config_dir, _require_string(reference, "check.reference")),
    }


def _normalize_trajectory_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a single trajectory check."""
    source = raw_check.get("estimated", raw_check.get("source"))
    reference = raw_check.get("reference")
    return {
        "source": _resolve_path(config_dir, _require_string(source, "check.estimated")),
        "reference": _resolve_path(config_dir, _require_string(reference, "check.reference")),
    }


def _normalize_trajectory_batch_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a trajectory batch check."""
    directory = raw_check.get("directory", raw_check.get("dir"))
    reference_dir = raw_check.get("reference_dir", raw_check.get("reference-directory"))
    return {
        "directory": _resolve_path(config_dir, _require_string(directory, "check.directory")),
        "reference_dir": _resolve_path(
            config_dir,
            _require_string(reference_dir, "check.reference_dir"),
        ),
    }


def _normalize_run_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a single integrated run check."""
    return {
        "map": _resolve_path(config_dir, _require_string(raw_check.get("map"), "check.map")),
        "map_reference": _resolve_path(
            config_dir,
            _require_string(raw_check.get("map_reference"), "check.map_reference"),
        ),
        "trajectory": _resolve_path(
            config_dir,
            _require_string(raw_check.get("trajectory"), "check.trajectory"),
        ),
        "trajectory_reference": _resolve_path(
            config_dir,
            _require_string(raw_check.get("trajectory_reference"), "check.trajectory_reference"),
        ),
    }


def _normalize_detection_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a single detection check."""
    source = raw_check.get("estimated", raw_check.get("source"))
    reference = raw_check.get("reference")
    return {
        "source": _resolve_path(config_dir, _require_string(source, "check.estimated")),
        "reference": _resolve_path(config_dir, _require_string(reference, "check.reference")),
    }


def _normalize_tracking_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a single tracking check."""
    source = raw_check.get("estimated", raw_check.get("source"))
    reference = raw_check.get("reference")
    return {
        "source": _resolve_path(config_dir, _require_string(source, "check.estimated")),
        "reference": _resolve_path(config_dir, _require_string(reference, "check.reference")),
    }


def _normalize_run_batch_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a run batch check."""
    return {
        "map_dir": _resolve_path(
            config_dir,
            _require_string(raw_check.get("map_dir"), "check.map_dir"),
        ),
        "map_reference_dir": _resolve_path(
            config_dir,
            _require_string(raw_check.get("map_reference_dir"), "check.map_reference_dir"),
        ),
        "trajectory_dir": _resolve_path(
            config_dir,
            _require_string(raw_check.get("trajectory_dir"), "check.trajectory_dir"),
        ),
        "trajectory_reference_dir": _resolve_path(
            config_dir,
            _require_string(
                raw_check.get("trajectory_reference_dir"),
                "check.trajectory_reference_dir",
            ),
        ),
    }


def _normalize_ground_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a ground segmentation check."""
    return {
        "estimated_ground": _resolve_path(
            config_dir,
            _require_string(raw_check.get("estimated_ground"), "check.estimated_ground"),
        ),
        "estimated_nonground": _resolve_path(
            config_dir,
            _require_string(raw_check.get("estimated_nonground"), "check.estimated_nonground"),
        ),
        "reference_ground": _resolve_path(
            config_dir,
            _require_string(raw_check.get("reference_ground"), "check.reference_ground"),
        ),
        "reference_nonground": _resolve_path(
            config_dir,
            _require_string(raw_check.get("reference_nonground"), "check.reference_nonground"),
        ),
    }


def _normalize_image_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a photometric (PSNR/SSIM) image check."""
    inputs = {
        "rendered_dir": _resolve_path(
            config_dir,
            _require_string(raw_check.get("rendered_dir"), "check.rendered_dir"),
        ),
        "reference_dir": _resolve_path(
            config_dir,
            _require_string(raw_check.get("reference_dir"), "check.reference_dir"),
        ),
    }
    metrics_raw = raw_check.get("metrics")
    if metrics_raw is not None:
        inputs["metrics"] = (
            ",".join(str(item).strip() for item in metrics_raw if str(item).strip())
            if isinstance(metrics_raw, list)
            else _require_string(metrics_raw, "check.metrics")
        )
    return inputs


def _normalize_rendered_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a 3DGS render-and-score check."""
    splat_raw = raw_check.get("splat", raw_check.get("splat_path"))
    cameras_raw = raw_check.get("cameras", raw_check.get("cameras_path"))
    inputs = {
        "splat": _resolve_path(
            config_dir,
            _require_string(splat_raw, "check.splat"),
        ),
        "cameras": _resolve_path(
            config_dir,
            _require_string(cameras_raw, "check.cameras"),
        ),
        "reference_dir": _resolve_path(
            config_dir,
            _require_string(raw_check.get("reference_dir"), "check.reference_dir"),
        ),
    }
    optional_paths = {
        "reference_pointcloud": raw_check.get(
            "reference_pointcloud",
            raw_check.get("reference", raw_check.get("reference_pcd")),
        ),
        "rendered_dir": raw_check.get("rendered_dir"),
    }
    for key, value in optional_paths.items():
        _add_optional_resolved_input(inputs, key, value, config_dir, f"check.{key}")

    metrics_raw = raw_check.get("metrics")
    if metrics_raw is not None:
        if isinstance(metrics_raw, list):
            metrics = ",".join(str(item).strip() for item in metrics_raw if str(item).strip())
        else:
            metrics = _require_string(metrics_raw, "check.metrics")
        inputs["metrics"] = metrics

    for key in ("opacity_threshold", "geometry_opacity_threshold", "geometry_voxel", "max_pairs"):
        value = raw_check.get(key)
        if value is not None:
            inputs[key] = str(value)

    geometry_splat_method = raw_check.get("geometry_splat_method")
    if geometry_splat_method is not None:
        inputs["geometry_splat_method"] = _require_string(
            geometry_splat_method,
            "check.geometry_splat_method",
        )
    geometry_splat_samples = raw_check.get("geometry_splat_samples")
    if geometry_splat_samples is not None:
        inputs["geometry_splat_samples"] = str(int(geometry_splat_samples))

    render_device = raw_check.get("render_device")
    if render_device is not None:
        inputs["render_device"] = _require_string(render_device, "check.render_device")
    skip_render = raw_check.get("skip_render")
    if skip_render is not None:
        inputs["skip_render"] = "true" if bool(skip_render) else "false"
    return inputs


def _add_optional_resolved_input(
    inputs: dict[str, str],
    key: str,
    value: object,
    config_dir: Path,
    context: str,
) -> None:
    resolved = _resolve_optional_path(config_dir, value, context)
    if resolved is not None:
        inputs[key] = resolved


def _normalize_loop_closure_inputs(raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize inputs for a manual loop-closure before/after check."""
    session_map_name = _optional_string(
        raw_check.get("session_map_name"),
        "check.session_map_name",
    ) or "map.pcd"
    before_session_root = _resolve_optional_path(
        config_dir,
        raw_check.get("before_session_root"),
        "check.before_session_root",
    )
    after_session_root = _resolve_optional_path(
        config_dir,
        raw_check.get("after_session_root"),
        "check.after_session_root",
    )

    before_map = raw_check.get("before_map", raw_check.get("before"))
    after_map = raw_check.get("after_map", raw_check.get("after"))
    reference_map = raw_check.get("reference_map", raw_check.get("reference"))
    inputs = {
        "before_map": (
            _resolve_path(config_dir, _require_string(before_map, "check.before_map"))
            if before_map is not None
            else str((Path(before_session_root or "") / session_map_name).resolve())
        ),
        "after_map": (
            _resolve_path(config_dir, _require_string(after_map, "check.after_map"))
            if after_map is not None
            else str((Path(after_session_root or "") / session_map_name).resolve())
        ),
        "reference_map": _resolve_path(
            config_dir,
            _require_string(reference_map, "check.reference_map"),
        ),
        "session_map_name": session_map_name,
    }
    if before_map is None and before_session_root is None:
        raise ValueError("check.before_map is required unless check.before_session_root is set")
    if after_map is None and after_session_root is None:
        raise ValueError("check.after_map is required unless check.after_session_root is set")
    if before_session_root is not None:
        inputs["before_session_root"] = before_session_root
    if after_session_root is not None:
        inputs["after_session_root"] = after_session_root

    optional_paths = {
        "before_trajectory": raw_check.get("before_traj", raw_check.get("before_trajectory")),
        "after_trajectory": raw_check.get("after_traj", raw_check.get("after_trajectory")),
        "reference_trajectory": raw_check.get(
            "ref_traj",
            raw_check.get("reference_traj", raw_check.get("reference_trajectory")),
        ),
        "before_g2o": raw_check.get("before_g2o"),
        "after_g2o": raw_check.get("after_g2o"),
        "before_tum": raw_check.get("before_tum"),
        "after_tum": raw_check.get("after_tum"),
        "before_key_point_frame": raw_check.get("before_key_point_frame"),
        "after_key_point_frame": raw_check.get("after_key_point_frame"),
    }
    for key, value in optional_paths.items():
        _add_optional_resolved_input(inputs, key, value, config_dir, f"check.{key}")
    return inputs


def _normalize_inputs(kind: CheckKind, raw_check: dict[str, Any], config_dir: Path) -> dict[str, str]:
    """Normalize kind-specific input paths."""
    if kind == "artifact":
        return _normalize_artifact_inputs(raw_check, config_dir)
    if kind == "artifact_batch":
        return _normalize_artifact_batch_inputs(raw_check, config_dir)
    if kind == "trajectory":
        return _normalize_trajectory_inputs(raw_check, config_dir)
    if kind == "trajectory_batch":
        return _normalize_trajectory_batch_inputs(raw_check, config_dir)
    if kind == "detection":
        return _normalize_detection_inputs(raw_check, config_dir)
    if kind == "tracking":
        return _normalize_tracking_inputs(raw_check, config_dir)
    if kind == "run":
        return _normalize_run_inputs(raw_check, config_dir)
    if kind == "ground":
        return _normalize_ground_inputs(raw_check, config_dir)
    if kind == "loop_closure":
        return _normalize_loop_closure_inputs(raw_check, config_dir)
    if kind == "image":
        return _normalize_image_inputs(raw_check, config_dir)
    if kind == "rendered":
        return _normalize_rendered_inputs(raw_check, config_dir)
    if kind == "structure":
        return {"source": _resolve_path(config_dir, _require_string(raw_check.get("source", raw_check.get("path")), "check.source"))}
    if kind == "uncertainty":
        return {
            "source": _resolve_path(config_dir, _require_string(raw_check.get("estimated", raw_check.get("source")), "check.estimated")),
            "reference": _resolve_path(config_dir, _require_string(raw_check.get("reference"), "check.reference")),
        }
    return _normalize_run_batch_inputs(raw_check, config_dir)


def _normalize_check(
    raw_check: object,
    defaults: dict[str, Any],
    config_dir: Path,
    index: int,
) -> CheckSpec:
    """Normalize one raw config entry."""
    check = _as_mapping(raw_check, f"checks[{index - 1}]")
    kind = _normalize_kind(check.get("kind"))
    raw_id = check.get("id", check.get("name"))
    check_id = (
        _require_string(raw_id, f"checks[{index - 1}].id")
        if raw_id is not None
        else f"{kind}-{index}"
    )
    # detection/tracking thresholds are IoU values (0–1), not distance thresholds
    # used by artifact checks — do not inherit from defaults for these kinds.
    if kind in {"detection", "tracking"}:
        raw_thresholds = check.get("thresholds")
    else:
        raw_thresholds = check.get("thresholds", defaults.get("thresholds"))
    thresholds = _normalize_thresholds(
        raw_thresholds,
        f"checks[{index - 1}].thresholds",
    )
    max_time_delta = float(check.get("max_time_delta", defaults.get("max_time_delta", 0.05)))
    recursive = _optional_bool(
        check.get("recursive", defaults.get("recursive")),
        f"checks[{index - 1}].recursive",
    )
    alignment = _normalize_alignment(
        check.get("alignment", defaults.get("alignment")),
        f"checks[{index - 1}].alignment",
    )
    outputs = _normalize_outputs(check, defaults, config_dir, check_id)
    return CheckSpec(
        check_id=check_id,
        kind=kind,
        inputs=_normalize_inputs(kind, check, config_dir),
        thresholds=thresholds,
        max_time_delta=max_time_delta,
        recursive=recursive,
        alignment=alignment,
        gate=_normalize_gate(defaults, check, kind),
        severity=_normalize_severity(defaults, check),
        compressed_dir=_resolve_optional_path(
            config_dir,
            check.get("compressed_dir"),
            f"checks[{index - 1}].compressed_dir",
        ),
        baseline_dir=_resolve_optional_path(
            config_dir,
            check.get("baseline_dir"),
            f"checks[{index - 1}].baseline_dir",
        ),
        outputs=outputs,
    )


def load_check_suite(config_path: str) -> CheckSuite:
    """Load a YAML or JSON check suite from disk."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError("Config file is empty")
    config = _as_mapping(raw, "config")
    version = config.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported config version: {version}")
    defaults = _as_mapping(config.get("defaults"), "defaults")
    raw_checks = _as_list(config.get("checks"), "checks")
    if not raw_checks:
        raise ValueError("checks must contain at least one item")
    project = _optional_string(config.get("project"), "project")
    summary_output_json = _resolve_optional_path(
        path.parent,
        config.get("summary_output_json"),
        "summary_output_json",
    )
    checks = tuple(
        _normalize_check(item, defaults, path.parent, index)
        for index, item in enumerate(raw_checks, start=1)
    )
    return CheckSuite(
        config_path=str(path),
        project=project,
        checks=checks,
        summary_output_json=summary_output_json,
    )


def _artifact_map_metrics(
    source_path: str,
    reference_path: str,
    *,
    voxel_size: float,
) -> dict[str, float]:
    """Compute MapEval-style AWD/SCS for artifact/map checks."""
    import numpy as np

    from ca.core.map_evaluate import compute_voxel_wasserstein_metrics
    from ca.io import load_point_cloud

    est = np.asarray(load_point_cloud(source_path).points, dtype=np.float64)
    ref = np.asarray(load_point_cloud(reference_path).points, dtype=np.float64)
    return compute_voxel_wasserstein_metrics(est, ref, voxel_size=float(voxel_size))


def _artifact_quality_gate(
    auc: float,
    chamfer_distance: float,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    awd_m: float | None = None,
    max_awd: float | None = None,
    scs: float | None = None,
    max_scs: float | None = None,
) -> dict[str, Any] | None:
    """Build optional pass/fail metadata for artifact-style checks."""
    if min_auc is None and max_chamfer is None and max_awd is None and max_scs is None:
        return None
    reasons: list[str] = []
    if min_auc is not None and auc < min_auc:
        reasons.append(f"AUC {auc:.4f} < min_auc {min_auc:.4f}")
    if max_chamfer is not None and chamfer_distance > max_chamfer:
        reasons.append(f"Chamfer {chamfer_distance:.4f} > max_chamfer {max_chamfer:.4f}")
    if max_awd is not None:
        if awd_m is None or not math.isfinite(awd_m):
            reasons.append("AWD unavailable: no voxel pair met the minimum point count")
        elif awd_m > max_awd:
            reasons.append(f"AWD {awd_m:.4f} m > max_awd {max_awd:.4f} m")
    if max_scs is not None:
        if scs is None or not math.isfinite(scs):
            reasons.append("SCS unavailable: no evaluated voxel had an evaluated neighbor")
        elif scs > max_scs:
            reasons.append(f"SCS {scs:.4f} > max_scs {max_scs:.4f}")
    return {
        "passed": not reasons,
        "min_auc": min_auc,
        "max_chamfer": max_chamfer,
        "max_awd": max_awd,
        "max_scs": max_scs,
        "reasons": reasons,
    }


def _artifact_inspection_commands(source_path: str, reference_path: str) -> dict[str, str]:
    """Build follow-up commands for interactive artifact inspection."""
    source_stem = Path(source_path).stem
    reference_stem = Path(reference_path).stem
    snapshot_name = f"{source_stem}_vs_{reference_stem}_heatmap.png"
    source = shlex.quote(source_path)
    reference = shlex.quote(reference_path)
    return {
        "web_heatmap": f"ca web {source} {reference} --heatmap",
        "heatmap3d": f"ca heatmap3d {source} {reference} -o {shlex.quote(snapshot_name)}",
    }


def _write_json(path: str, data: Any) -> None:
    """Write JSON output to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _maybe_artifact_map_metrics(spec: CheckSpec) -> dict[str, float] | None:
    """Compute AWD/SCS when map-style gates are configured."""
    if spec.gate.get("max_awd") is None and spec.gate.get("max_scs") is None:
        return None
    voxel_size = float(spec.gate.get("voxel_size", 0.5))
    return _artifact_map_metrics(
        spec.inputs["source"],
        spec.inputs["reference"],
        voxel_size=voxel_size,
    )


def _artifact_batch_item(
    spec: CheckSpec,
    result: dict[str, Any],
    map_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Convert a single evaluate() result into the batch/report shape."""
    best_f1 = max(result["f1_scores"], key=lambda score: score["f1"])
    awd_m = map_metrics.get("awd_m") if map_metrics is not None else None
    scs = map_metrics.get("scs") if map_metrics is not None else None
    gate = _artifact_quality_gate(
        result["auc"],
        result["chamfer_distance"],
        min_auc=cast(float | None, spec.gate.get("min_auc")),
        max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
        awd_m=cast(float | None, awd_m),
        max_awd=cast(float | None, spec.gate.get("max_awd")),
        scs=cast(float | None, scs),
        max_scs=cast(float | None, spec.gate.get("max_scs")),
    )
    item: dict[str, Any] = {
        "path": spec.inputs["source"],
        "num_points": result["source_points"],
        "reference_path": spec.inputs["reference"],
        "reference_points": result["target_points"],
        "chamfer_distance": result["chamfer_distance"],
        "hausdorff_distance": result["hausdorff_distance"],
        "auc": result["auc"],
        "best_f1": best_f1,
        "f1_scores": result["f1_scores"],
        "quality_gate": gate,
        "inspect": _artifact_inspection_commands(spec.inputs["source"], spec.inputs["reference"]),
        "compression": None,
    }
    if map_metrics is not None:
        item["awd_m"] = map_metrics["awd_m"]
        item["scs"] = map_metrics["scs"]
        item["n_awd_voxels"] = map_metrics["n_awd_voxels"]
        item["n_scs_voxels"] = map_metrics["n_scs_voxels"]
    return item


def _run_artifact_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a single artifact QA check."""
    result = evaluate(
        spec.inputs["source"],
        spec.inputs["reference"],
        thresholds=list(spec.thresholds) if spec.thresholds else None,
    )
    map_metrics = _maybe_artifact_map_metrics(spec)
    batch_item = _artifact_batch_item(spec, result, map_metrics=map_metrics)
    result_with_summary = {
        **result,
        "best_f1": batch_item["best_f1"],
        "quality_gate": batch_item["quality_gate"],
        "inspect": batch_item["inspect"],
    }
    if map_metrics is not None:
        result_with_summary.update(map_metrics)
    if spec.outputs.report_path:
        save_batch_report(
            [batch_item],
            spec.inputs["reference"],
            spec.outputs.report_path,
            min_auc=cast(float | None, spec.gate.get("min_auc")),
            max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
        )
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, result_with_summary)
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None
        if batch_item["quality_gate"] is None
        else batch_item["quality_gate"]["passed"],
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "auc": result["auc"],
            "chamfer_distance": result["chamfer_distance"],
            "hausdorff_distance": result["hausdorff_distance"],
            "best_f1": batch_item["best_f1"],
            **(
                {
                    "awd_m": map_metrics["awd_m"],
                    "scs": map_metrics["scs"],
                }
                if map_metrics is not None
                else {}
            ),
            "passed": None
            if batch_item["quality_gate"] is None
            else batch_item["quality_gate"]["passed"],
        },
        "result": result_with_summary,
    }


def _run_artifact_batch_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a batch artifact QA check."""
    results = batch_evaluate(
        spec.inputs["directory"],
        spec.inputs["reference"],
        recursive=spec.recursive,
        thresholds=list(spec.thresholds) if spec.thresholds else None,
        min_auc=cast(float | None, spec.gate.get("min_auc")),
        max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
        compressed_dir=spec.compressed_dir,
        baseline_dir=spec.baseline_dir,
    )
    if spec.gate.get("max_awd") is not None or spec.gate.get("max_scs") is not None:
        for item in results:
            map_metrics = _artifact_map_metrics(
                str(item["path"]),
                spec.inputs["reference"],
                voxel_size=float(spec.gate.get("voxel_size", 0.5)),
            )
            item.update(map_metrics)
            item["quality_gate"] = _artifact_quality_gate(
                float(item["auc"]),
                float(item["chamfer_distance"]),
                min_auc=cast(float | None, spec.gate.get("min_auc")),
                max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
                awd_m=map_metrics["awd_m"],
                max_awd=cast(float | None, spec.gate.get("max_awd")),
                scs=map_metrics["scs"],
                max_scs=cast(float | None, spec.gate.get("max_scs")),
            )
    summary = make_batch_summary(
        results,
        spec.inputs["reference"],
        min_auc=cast(float | None, spec.gate.get("min_auc")),
        max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
    )
    if spec.outputs.report_path:
        save_batch_report(
            results,
            spec.inputs["reference"],
            spec.outputs.report_path,
            min_auc=cast(float | None, spec.gate.get("min_auc")),
            max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
        )
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, results)
    gate = cast(dict[str, Any] | None, summary.get("quality_gate"))
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if gate is None else gate["fail_count"] == 0,
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "total_files": summary["total_files"],
            "mean_auc": summary["mean_auc"],
            "mean_chamfer_distance": summary["mean_chamfer_distance"],
            "passed": None if gate is None else gate["fail_count"] == 0,
            "quality_gate": gate,
        },
        "result": results,
    }


def _run_trajectory_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a single trajectory QA check."""
    align_origin, align_rigid = _alignment_flags(spec.alignment)
    result = evaluate_trajectory(
        spec.inputs["source"],
        spec.inputs["reference"],
        max_time_delta=spec.max_time_delta,
        align_origin=align_origin,
        align_rigid=align_rigid,
        max_ate=cast(float | None, spec.gate.get("max_ate")),
        max_rpe=cast(float | None, spec.gate.get("max_rpe")),
        max_drift=cast(float | None, spec.gate.get("max_drift")),
        min_coverage=cast(float | None, spec.gate.get("min_coverage")),
        max_lateral=cast(float | None, spec.gate.get("max_lateral")),
        max_longitudinal=cast(float | None, spec.gate.get("max_longitudinal")),
    )
    if spec.outputs.report_path:
        save_trajectory_report(result, spec.outputs.report_path)
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, result)
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if gate is None else gate["passed"],
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "matched_poses": result["matching"]["matched_poses"],
            "coverage_ratio": result["matching"]["coverage_ratio"],
            "ate_rmse": result["ate"]["rmse"],
            "rpe_rmse": result["rpe_translation"]["rmse"],
            "drift_endpoint": result["drift"]["endpoint"],
            "lateral_rmse": result["lateral"]["rmse"],
            "longitudinal_rmse": result["longitudinal"]["rmse"],
            "alignment_mode": result["alignment"]["mode"],
            "passed": None if gate is None else gate["passed"],
        },
        "result": result,
    }


def _run_trajectory_batch_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a batch trajectory QA check."""
    align_origin, align_rigid = _alignment_flags(spec.alignment)
    results = trajectory_batch_evaluate(
        spec.inputs["directory"],
        spec.inputs["reference_dir"],
        recursive=spec.recursive,
        max_time_delta=spec.max_time_delta,
        align_origin=align_origin,
        align_rigid=align_rigid,
        max_ate=cast(float | None, spec.gate.get("max_ate")),
        max_rpe=cast(float | None, spec.gate.get("max_rpe")),
        max_drift=cast(float | None, spec.gate.get("max_drift")),
        min_coverage=cast(float | None, spec.gate.get("min_coverage")),
    )
    summary = make_trajectory_batch_summary(
        results,
        spec.inputs["reference_dir"],
        max_ate=cast(float | None, spec.gate.get("max_ate")),
        max_rpe=cast(float | None, spec.gate.get("max_rpe")),
        max_drift=cast(float | None, spec.gate.get("max_drift")),
        min_coverage=cast(float | None, spec.gate.get("min_coverage")),
    )
    if spec.outputs.report_path:
        save_trajectory_batch_report(
            results,
            spec.inputs["reference_dir"],
            spec.outputs.report_path,
            max_ate=cast(float | None, spec.gate.get("max_ate")),
            max_rpe=cast(float | None, spec.gate.get("max_rpe")),
            max_drift=cast(float | None, spec.gate.get("max_drift")),
            min_coverage=cast(float | None, spec.gate.get("min_coverage")),
        )
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, results)
    gate = cast(dict[str, Any] | None, summary.get("quality_gate"))
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if gate is None else gate["fail_count"] == 0,
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "total_files": summary["total_files"],
            "mean_ate_rmse": summary["mean_ate_rmse"],
            "mean_rpe_rmse": summary["mean_rpe_rmse"],
            "mean_coverage_ratio": summary["mean_coverage_ratio"],
            "passed": None if gate is None else gate["fail_count"] == 0,
            "quality_gate": gate,
        },
        "result": results,
    }


def _run_detection_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a single detection QA check."""
    result = evaluate_detection(
        spec.inputs["source"],
        spec.inputs["reference"],
        iou_thresholds=list(spec.thresholds) if spec.thresholds else None,
        min_map=cast(float | None, spec.gate.get("min_map")),
        min_precision=cast(float | None, spec.gate.get("min_precision")),
        min_recall=cast(float | None, spec.gate.get("min_recall")),
        min_f1=cast(float | None, spec.gate.get("min_f1")),
    )
    if spec.outputs.report_path:
        save_detection_report(result, spec.outputs.report_path)
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, result)
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))
    primary = result["primary_threshold_result"]
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if gate is None else gate["passed"],
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "map": result["mAP"],
            "precision": primary["precision"],
            "recall": primary["recall"],
            "f1": primary["f1"],
            "mean_iou": primary["mean_iou"],
            "passed": None if gate is None else gate["passed"],
        },
        "result": result,
    }


def _run_tracking_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a single tracking QA check."""
    if spec.thresholds is not None and len(spec.thresholds) > 1:
        raise ValueError(
            f"tracking check '{spec.check_id}' accepts exactly one threshold, "
            f"got {len(spec.thresholds)}: {spec.thresholds}"
        )
    iou_threshold = float(spec.thresholds[0]) if spec.thresholds else 0.5
    result = evaluate_tracking(
        spec.inputs["source"],
        spec.inputs["reference"],
        iou_threshold=iou_threshold,
        min_mota=cast(float | None, spec.gate.get("min_mota")),
        min_recall=cast(float | None, spec.gate.get("min_recall")),
        max_id_switches=(
            int(spec.gate["max_id_switches"])
            if "max_id_switches" in spec.gate
            else None
        ),
    )
    if spec.outputs.report_path:
        save_tracking_report(result, spec.outputs.report_path)
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, result)
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if gate is None else gate["passed"],
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "mota": result["tracking"]["mota"],
            "recall": result["detection"]["recall"],
            "id_switches": result["tracking"]["id_switches"],
            "mean_iou": result["tracking"]["mean_iou"],
            "passed": None if gate is None else gate["passed"],
        },
        "result": result,
    }


def _run_run_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a single integrated map + trajectory QA check."""
    align_origin, align_rigid = _alignment_flags(spec.alignment)
    result = evaluate_run(
        spec.inputs["map"],
        spec.inputs["map_reference"],
        spec.inputs["trajectory"],
        spec.inputs["trajectory_reference"],
        thresholds=list(spec.thresholds) if spec.thresholds else None,
        max_time_delta=spec.max_time_delta,
        align_origin=align_origin,
        align_rigid=align_rigid,
        min_auc=cast(float | None, spec.gate.get("min_auc")),
        max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
        max_ate=cast(float | None, spec.gate.get("max_ate")),
        max_rpe=cast(float | None, spec.gate.get("max_rpe")),
        max_drift=cast(float | None, spec.gate.get("max_drift")),
        min_coverage=cast(float | None, spec.gate.get("min_coverage")),
    )
    if spec.outputs.report_path:
        save_run_report(result, spec.outputs.report_path)
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, result)
    gate = cast(dict[str, Any] | None, result.get("overall_quality_gate"))
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if gate is None else gate["passed"],
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "map_auc": result["map"]["auc"],
            "map_chamfer_distance": result["map"]["chamfer_distance"],
            "trajectory_ate_rmse": result["trajectory"]["ate"]["rmse"],
            "trajectory_rpe_rmse": result["trajectory"]["rpe_translation"]["rmse"],
            "trajectory_drift_endpoint": result["trajectory"]["drift"]["endpoint"],
            "coverage_ratio": result["trajectory"]["matching"]["coverage_ratio"],
            "passed": None if gate is None else gate["passed"],
        },
        "result": result,
    }


def _run_run_batch_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a batch integrated map + trajectory QA check."""
    align_origin, align_rigid = _alignment_flags(spec.alignment)
    results = evaluate_run_batch(
        spec.inputs["map_dir"],
        spec.inputs["map_reference_dir"],
        spec.inputs["trajectory_dir"],
        spec.inputs["trajectory_reference_dir"],
        recursive=spec.recursive,
        thresholds=list(spec.thresholds) if spec.thresholds else None,
        max_time_delta=spec.max_time_delta,
        align_origin=align_origin,
        align_rigid=align_rigid,
        min_auc=cast(float | None, spec.gate.get("min_auc")),
        max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
        max_ate=cast(float | None, spec.gate.get("max_ate")),
        max_rpe=cast(float | None, spec.gate.get("max_rpe")),
        max_drift=cast(float | None, spec.gate.get("max_drift")),
        min_coverage=cast(float | None, spec.gate.get("min_coverage")),
    )
    summary = make_run_batch_summary(
        results,
        spec.inputs["map_reference_dir"],
        spec.inputs["trajectory_reference_dir"],
        min_auc=cast(float | None, spec.gate.get("min_auc")),
        max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
        max_ate=cast(float | None, spec.gate.get("max_ate")),
        max_rpe=cast(float | None, spec.gate.get("max_rpe")),
        max_drift=cast(float | None, spec.gate.get("max_drift")),
        min_coverage=cast(float | None, spec.gate.get("min_coverage")),
    )
    if spec.outputs.report_path:
        save_run_batch_report(
            results,
            spec.inputs["map_reference_dir"],
            spec.inputs["trajectory_reference_dir"],
            spec.outputs.report_path,
            min_auc=cast(float | None, spec.gate.get("min_auc")),
            max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
            max_ate=cast(float | None, spec.gate.get("max_ate")),
            max_rpe=cast(float | None, spec.gate.get("max_rpe")),
            max_drift=cast(float | None, spec.gate.get("max_drift")),
            min_coverage=cast(float | None, spec.gate.get("min_coverage")),
        )
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, results)
    gate = cast(dict[str, Any] | None, summary.get("quality_gate"))
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if gate is None else gate["fail_count"] == 0,
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "total_runs": summary["total_runs"],
            "mean_map_auc": summary["mean_map_auc"],
            "mean_map_chamfer": summary["mean_map_chamfer"],
            "mean_traj_ate_rmse": summary["mean_traj_ate_rmse"],
            "mean_traj_coverage": summary["mean_traj_coverage"],
            "passed": None if gate is None else gate["fail_count"] == 0,
            "quality_gate": gate,
        },
        "result": results,
    }


def _run_ground_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a single ground segmentation QA check."""
    from ca.ground_evaluate import evaluate_ground_segmentation  # lazy to avoid circular import

    voxel_size = float(spec.gate.get("voxel_size", 0.2))  # type: ignore[arg-type]
    result = evaluate_ground_segmentation(
        spec.inputs["estimated_ground"],
        spec.inputs["estimated_nonground"],
        spec.inputs["reference_ground"],
        spec.inputs["reference_nonground"],
        voxel_size=voxel_size,
        min_precision=cast(float | None, spec.gate.get("min_precision")),
        min_recall=cast(float | None, spec.gate.get("min_recall")),
        min_f1=cast(float | None, spec.gate.get("min_f1")),
        min_iou=cast(float | None, spec.gate.get("min_iou")),
    )
    if spec.outputs.report_path:
        save_ground_report(result, spec.outputs.report_path)
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, result)
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if gate is None else gate["passed"],
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "iou": result["iou"],
            "accuracy": result["accuracy"],
            "passed": None if gate is None else gate["passed"],
        },
        "result": result,
    }


def _loop_closure_session_ok(report: dict[str, Any], label: str) -> bool | None:
    sessions = report.get("posegraph_session")
    if not isinstance(sessions, dict):
        return None
    session = sessions.get(label)
    if not isinstance(session, dict):
        return None
    summary = session.get("summary")
    if not isinstance(summary, dict):
        return None
    ok = summary.get("ok")
    return bool(ok) if isinstance(ok, bool) else None


def _write_loop_closure_report(path: str, report: dict[str, Any]) -> None:
    """Write a compact Markdown/HTML report for a loop-closure check."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m = report["map"]
    gate = report.get("quality_gate")
    gate_status = "INFO"
    if isinstance(gate, dict):
        gate_status = "PASS" if gate.get("passed") else "FAIL"
    lines = [
        "# Loop Closure QA Report",
        "",
        f"- Reference: `{m['reference']}`",
        f"- Before map: `{m['before']['path']}`",
        f"- After map: `{m['after']['path']}`",
        f"- Gate: {gate_status}",
        "",
        "| Metric | Before | After | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| AUC | {m['before']['auc']:.6f} | {m['after']['auc']:.6f} | "
            f"{m['delta']['auc']:.6f} |"
        ),
        (
            f"| Chamfer | {m['before']['chamfer_distance']:.6f} | "
            f"{m['after']['chamfer_distance']:.6f} | "
            f"{m['delta']['chamfer_distance']:.6f} |"
        ),
    ]
    if "trajectory" in report:
        t = report["trajectory"]
        lines.extend(
            [
                "",
                "## Trajectory",
                "",
                "| Metric | Before | After | Delta |",
                "|---|---:|---:|---:|",
                (
                    f"| ATE RMSE | {t['before']['ate_rmse']:.6f} | "
                    f"{t['after']['ate_rmse']:.6f} | {t['delta']['ate_rmse']:.6f} |"
                ),
                (
                    f"| Coverage | {t['before']['coverage']:.6f} | "
                    f"{t['after']['coverage']:.6f} | {t['delta']['coverage']:.6f} |"
                ),
            ]
        )
    if isinstance(gate, dict) and gate.get("reasons"):
        lines.extend(["", "## Gate Reasons", ""])
        lines.extend(f"- {reason}" for reason in gate["reasons"])

    text = "\n".join(lines) + "\n"
    if output_path.suffix.lower() in {".html", ".htm"}:
        body = escape(text)
        output_path.write_text(f"<pre>{body}</pre>\n", encoding="utf-8")
    else:
        output_path.write_text(text, encoding="utf-8")


def _run_loop_closure_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a manual loop-closure before/after QA check."""
    before_map = spec.inputs["before_map"]
    after_map = spec.inputs["after_map"]
    before_g2o = spec.inputs.get("before_g2o")
    after_g2o = spec.inputs.get("after_g2o")
    before_tum = spec.inputs.get("before_tum")
    after_tum = spec.inputs.get("after_tum")
    before_key_point_frame = spec.inputs.get("before_key_point_frame")
    after_key_point_frame = spec.inputs.get("after_key_point_frame")
    before_discovery = None
    after_discovery = None

    if "before_session_root" in spec.inputs:
        before_discovery = discover_session_paths(
            spec.inputs["before_session_root"],
            map_name=spec.inputs["session_map_name"],
        )
        before_map = before_discovery["map_path"] or before_map
        before_g2o = before_discovery["g2o_path"] or before_g2o
        before_tum = before_discovery["tum_path"] or before_tum
        before_key_point_frame = before_discovery["key_point_frame_dir"] or before_key_point_frame
        if before_discovery["map_path"] is None and not Path(before_map).exists():
            raise ValueError(
                "Before session discovery did not find a map file.\n"
                f"Looked for: {before_discovery['expected']['map_path']}\n"
                f"Also received before_map: {before_map}"
            )
    if "after_session_root" in spec.inputs:
        after_discovery = discover_session_paths(
            spec.inputs["after_session_root"],
            map_name=spec.inputs["session_map_name"],
        )
        after_map = after_discovery["map_path"] or after_map
        after_g2o = after_discovery["g2o_path"] or after_g2o
        after_tum = after_discovery["tum_path"] or after_tum
        after_key_point_frame = after_discovery["key_point_frame_dir"] or after_key_point_frame
        if after_discovery["map_path"] is None and not Path(after_map).exists():
            raise ValueError(
                "After session discovery did not find a map file.\n"
                f"Looked for: {after_discovery['expected']['map_path']}\n"
                f"Also received after_map: {after_map}"
            )

    align_origin, align_rigid = _alignment_flags(spec.alignment)
    result = build_loop_closure_report(
        before_map=before_map,
        after_map=after_map,
        reference_map=spec.inputs["reference_map"],
        thresholds=list(spec.thresholds) if spec.thresholds else None,
        before_trajectory=spec.inputs.get("before_trajectory"),
        after_trajectory=spec.inputs.get("after_trajectory"),
        reference_trajectory=spec.inputs.get("reference_trajectory"),
        trajectory_max_time_delta=spec.max_time_delta,
        trajectory_align_origin=align_origin,
        trajectory_align_rigid=align_rigid,
        before_g2o=before_g2o,
        after_g2o=after_g2o,
        before_tum=before_tum,
        after_tum=after_tum,
        before_key_point_frame_dir=before_key_point_frame,
        after_key_point_frame_dir=after_key_point_frame,
        gate=LoopClosureGate(
            min_auc_gain=cast(float | None, spec.gate.get("min_auc_gain")),
            max_after_chamfer=cast(float | None, spec.gate.get("max_after_chamfer")),
            min_ate_gain=cast(float | None, spec.gate.get("min_ate_gain")),
            max_after_ate=cast(float | None, spec.gate.get("max_after_ate")),
            require_posegraph_ok=bool(spec.gate.get("require_posegraph_ok", False)),
        ),
    )
    if before_discovery is not None or after_discovery is not None:
        result["discovery"] = {"before": before_discovery, "after": after_discovery}
    if spec.outputs.report_path:
        _write_loop_closure_report(spec.outputs.report_path, result)
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, result)

    gate = cast(dict[str, Any] | None, result.get("quality_gate"))
    trajectory = result.get("trajectory")
    before_ate = trajectory["before"]["ate_rmse"] if isinstance(trajectory, dict) else None
    after_ate = trajectory["after"]["ate_rmse"] if isinstance(trajectory, dict) else None
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if gate is None else gate["passed"],
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "map_auc_gain": result["map"]["delta"]["auc"],
            "after_map_auc": result["map"]["after"]["auc"],
            "after_chamfer_distance": result["map"]["after"]["chamfer_distance"],
            "trajectory_ate_gain": (before_ate - after_ate) if before_ate is not None and after_ate is not None else None,
            "after_trajectory_ate_rmse": after_ate,
            "posegraph_before_ok": _loop_closure_session_ok(result, "before"),
            "posegraph_after_ok": _loop_closure_session_ok(result, "after"),
            "passed": None if gate is None else gate["passed"],
            "quality_gate": gate,
        },
        "result": result,
    }


def _run_image_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a single photometric (PSNR/SSIM) image-set QA check."""
    from ca.core.image_evaluate import ImageEvalRequest, image_evaluate  # lazy import

    metrics_raw = spec.inputs.get("metrics", "psnr,ssim")
    metrics = tuple(item.strip() for item in metrics_raw.split(",") if item.strip())
    if spec.gate.get("max_dreamsim_distance") is not None and "dreamsim_distance" not in metrics:
        metrics = (*metrics, "dreamsim_distance")
    result = image_evaluate(
        ImageEvalRequest(
            rendered_dir=Path(spec.inputs["rendered_dir"]),
            reference_dir=Path(spec.inputs["reference_dir"]),
            metrics=metrics,
        )
    )
    summary = result.summary
    psnr_mean = cast(float | None, summary.get("psnr_mean"))
    ssim_mean = cast(float | None, summary.get("ssim_mean"))
    dreamsim_mean = cast(float | None, summary.get("dreamsim_distance_mean"))
    pairs_evaluated = int(summary["pairs_evaluated"])

    min_psnr = cast(float | None, spec.gate.get("min_psnr"))
    min_ssim = cast(float | None, spec.gate.get("min_ssim"))
    max_dreamsim = cast(float | None, spec.gate.get("max_dreamsim_distance"))
    has_gate = min_psnr is not None or min_ssim is not None or max_dreamsim is not None

    gate_reasons: list[str] = []
    if has_gate and pairs_evaluated == 0:
        gate_reasons.append(
            "0 pairs evaluated (no rendered/reference filename matches)"
        )
    else:
        # A None mean with pairs_evaluated > 0 means every scored pair was
        # bit-identical (PSNR=+inf, filtered from the aggregate), which
        # trivially satisfies any min_psnr floor — so only fail on a finite
        # mean that falls short.
        if min_psnr is not None and psnr_mean is not None and psnr_mean < min_psnr:
            gate_reasons.append(
                f"PSNR mean {psnr_mean:.4f} dB < min_psnr {min_psnr:.4f} dB"
            )
        if min_ssim is not None and ssim_mean is not None and ssim_mean < min_ssim:
            gate_reasons.append(
                f"SSIM mean {ssim_mean:.4f} < min_ssim {min_ssim:.4f}"
            )
        if max_dreamsim is not None and (
            dreamsim_mean is None or dreamsim_mean > max_dreamsim
        ):
            gate_reasons.append(
                "DreamSim distance unavailable"
                if dreamsim_mean is None
                else f"DreamSim distance mean {dreamsim_mean:.4f} > max_dreamsim_distance {max_dreamsim:.4f}"
            )

    quality_gate = (
        {
            "passed": not gate_reasons,
            "min_psnr": min_psnr,
            "min_ssim": min_ssim,
            "max_dreamsim_distance": max_dreamsim,
            "reasons": gate_reasons,
        }
        if has_gate
        else None
    )

    result_dict: dict[str, Any] = {
        "summary": summary,
        "pairs": result.pairs,
        "metadata": result.metadata,
        "quality_gate": quality_gate,
    }
    if spec.outputs.report_path:
        save_image_report(result_dict, spec.outputs.report_path)
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, result_dict)
    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if quality_gate is None else quality_gate["passed"],
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": {
            "psnr_mean": psnr_mean,
            "ssim_mean": ssim_mean,
            "dreamsim_distance_mean": dreamsim_mean,
            "pairs_evaluated": pairs_evaluated,
            "pairs_missing_in_reference": int(summary["pairs_missing_in_reference"]),
            "pairs_size_mismatch": int(summary["pairs_size_mismatch"]),
            "passed": None if quality_gate is None else quality_gate["passed"],
        },
        "result": result_dict,
    }


def _run_rendered_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a 3DGS render → photometric (+ optional geometry) QA check."""
    from ca.core.rendered_evaluate import (
        RenderedEvalRequest,
        rendered_evaluate,
        rendered_evaluate_to_dict,
    )

    metrics_raw = spec.inputs.get("metrics", "psnr,ssim")
    metrics = tuple(item.strip() for item in metrics_raw.split(",") if item.strip()) or (
        "psnr",
        "ssim",
    )
    if spec.gate.get("max_dreamsim_distance") is not None and "dreamsim_distance" not in metrics:
        metrics = (*metrics, "dreamsim_distance")

    opacity_threshold = (
        float(spec.inputs["opacity_threshold"])
        if "opacity_threshold" in spec.inputs
        else None
    )
    geometry_opacity_threshold = (
        float(spec.inputs["geometry_opacity_threshold"])
        if "geometry_opacity_threshold" in spec.inputs
        else opacity_threshold
    )
    geometry_voxel = (
        float(spec.inputs["geometry_voxel"]) if "geometry_voxel" in spec.inputs else None
    )
    max_pairs = int(spec.inputs["max_pairs"]) if "max_pairs" in spec.inputs else None
    geometry_splat_method = spec.inputs.get("geometry_splat_method", "centers")
    geometry_splat_samples = int(spec.inputs.get("geometry_splat_samples", "8"))
    render_device = spec.inputs.get("render_device")
    skip_render = spec.inputs.get("skip_render", "false").lower() in {"1", "true", "yes"}

    reference_pointcloud = spec.inputs.get("reference_pointcloud")
    keep_rendered_dir = (
        Path(spec.inputs["rendered_dir"]) if "rendered_dir" in spec.inputs else None
    )

    result = rendered_evaluate(
        RenderedEvalRequest(
            splat_path=Path(spec.inputs["splat"]),
            cameras_path=Path(spec.inputs["cameras"]),
            reference_dir=Path(spec.inputs["reference_dir"]),
            metrics=metrics,
            reference_pointcloud=(
                Path(reference_pointcloud) if reference_pointcloud is not None else None
            ),
            opacity_threshold=opacity_threshold,
            geometry_opacity_threshold=geometry_opacity_threshold,
            geometry_voxel=geometry_voxel,
            geometry_splat_method=geometry_splat_method,
            geometry_splat_samples=geometry_splat_samples,
            geometry_thresholds=list(spec.thresholds) if spec.thresholds else None,
            render_device=render_device,
            keep_rendered_dir=keep_rendered_dir,
            skip_render=skip_render,
            max_pairs=max_pairs,
        )
    )
    result_dict = rendered_evaluate_to_dict(result)

    summary = cast(dict[str, Any], result.photometric["summary"])
    psnr_mean = cast(float | None, summary.get("psnr_mean"))
    ssim_mean = cast(float | None, summary.get("ssim_mean"))
    lpips_mean = cast(float | None, summary.get("lpips_mean"))
    dreamsim_mean = cast(float | None, summary.get("dreamsim_distance_mean"))
    pairs_evaluated = int(summary["pairs_evaluated"])

    min_psnr = cast(float | None, spec.gate.get("min_psnr"))
    min_ssim = cast(float | None, spec.gate.get("min_ssim"))
    max_lpips = cast(float | None, spec.gate.get("max_lpips"))
    max_dreamsim = cast(float | None, spec.gate.get("max_dreamsim_distance"))
    min_auc = cast(float | None, spec.gate.get("min_auc"))
    max_chamfer = cast(float | None, spec.gate.get("max_chamfer"))
    has_photometric_gate = (
        min_psnr is not None or min_ssim is not None or max_lpips is not None or max_dreamsim is not None
    )
    has_geometry_gate = min_auc is not None or max_chamfer is not None
    has_gate = has_photometric_gate or has_geometry_gate

    gate_reasons: list[str] = []
    if has_photometric_gate and pairs_evaluated == 0:
        gate_reasons.append(
            "0 pairs evaluated (no rendered/reference filename matches)"
        )
    else:
        if min_psnr is not None and psnr_mean is not None and psnr_mean < min_psnr:
            gate_reasons.append(
                f"PSNR mean {psnr_mean:.4f} dB < min_psnr {min_psnr:.4f} dB"
            )
        if min_ssim is not None and ssim_mean is not None and ssim_mean < min_ssim:
            gate_reasons.append(
                f"SSIM mean {ssim_mean:.4f} < min_ssim {min_ssim:.4f}"
            )
        if max_lpips is not None and lpips_mean is not None and lpips_mean > max_lpips:
            gate_reasons.append(
                f"LPIPS mean {lpips_mean:.4f} > max_lpips {max_lpips:.4f}"
            )
        if max_dreamsim is not None and (
            dreamsim_mean is None or dreamsim_mean > max_dreamsim
        ):
            gate_reasons.append(
                "DreamSim distance unavailable"
                if dreamsim_mean is None
                else f"DreamSim distance mean {dreamsim_mean:.4f} > max_dreamsim_distance {max_dreamsim:.4f}"
            )

    geometry_summary: dict[str, Any] | None = None
    if result.geometry is not None:
        geometry_summary = {
            "auc": float(result.geometry["auc"]),
            "chamfer_distance": float(result.geometry["chamfer_distance"]),
        }
        geom_gate = _artifact_quality_gate(
            geometry_summary["auc"],
            geometry_summary["chamfer_distance"],
            min_auc=min_auc,
            max_chamfer=max_chamfer,
        )
        if geom_gate is not None:
            gate_reasons.extend(cast(list[str], geom_gate["reasons"]))
    elif has_geometry_gate:
        gate_reasons.append(
            "geometry gate requested but reference_pointcloud was not configured"
        )

    quality_gate = (
        {
            "passed": not gate_reasons,
            "min_psnr": min_psnr,
            "min_ssim": min_ssim,
            "max_lpips": max_lpips,
            "max_dreamsim_distance": max_dreamsim,
            "min_auc": min_auc,
            "max_chamfer": max_chamfer,
            "reasons": gate_reasons,
        }
        if has_gate
        else None
    )
    result_dict["quality_gate"] = quality_gate

    if spec.outputs.report_path:
        save_rendered_report(result_dict, spec.outputs.report_path)
    if spec.outputs.json_path:
        _write_json(spec.outputs.json_path, result_dict)

    check_summary: dict[str, Any] = {
        "psnr_mean": psnr_mean,
        "ssim_mean": ssim_mean,
        "lpips_mean": lpips_mean,
        "dreamsim_distance_mean": dreamsim_mean,
        "pairs_evaluated": pairs_evaluated,
        "pairs_missing_in_reference": int(summary["pairs_missing_in_reference"]),
        "pairs_size_mismatch": int(summary["pairs_size_mismatch"]),
        "passed": None if quality_gate is None else quality_gate["passed"],
    }
    if geometry_summary is not None:
        check_summary["auc"] = geometry_summary["auc"]
        check_summary["chamfer_distance"] = geometry_summary["chamfer_distance"]

    return {
        "id": spec.check_id,
        "kind": spec.kind,
        "passed": None if quality_gate is None else quality_gate["passed"],
        "report_path": spec.outputs.report_path,
        "json_path": spec.outputs.json_path,
        "summary": check_summary,
        "result": result_dict,
    }


def _run_check(spec: CheckSpec) -> dict[str, Any]:
    """Dispatch one normalized check spec."""
    result: dict[str, Any]
    if spec.severity in {"skip", "not_applicable"}:
        result = {
            "quality_gate": {
                "passed": None,
                "severity": spec.severity,
                "reasons": [],
            }
        }
        if spec.outputs.json_path:
            _write_json(spec.outputs.json_path, result)
        return {
            "id": spec.check_id,
            "kind": spec.kind,
            "severity": spec.severity,
            "gate_status": spec.severity,
            "passed": None,
            "report_path": spec.outputs.report_path,
            "json_path": spec.outputs.json_path,
            "summary": {
                "passed": None,
                "reason": spec.severity,
            },
            "result": result,
        }
    if spec.kind == "artifact":
        result = _run_artifact_check(spec)
    elif spec.kind == "artifact_batch":
        result = _run_artifact_batch_check(spec)
    elif spec.kind == "trajectory":
        result = _run_trajectory_check(spec)
    elif spec.kind == "trajectory_batch":
        result = _run_trajectory_batch_check(spec)
    elif spec.kind == "detection":
        result = _run_detection_check(spec)
    elif spec.kind == "tracking":
        result = _run_tracking_check(spec)
    elif spec.kind == "run":
        result = _run_run_check(spec)
    elif spec.kind == "ground":
        result = _run_ground_check(spec)
    elif spec.kind == "loop_closure":
        result = _run_loop_closure_check(spec)
    elif spec.kind == "image":
        result = _run_image_check(spec)
    elif spec.kind == "rendered":
        result = _run_rendered_check(spec)
    elif spec.kind == "structure":
        from ca.core.plane_consistency import evaluate_plane_consistency

        metrics = evaluate_plane_consistency(
            spec.inputs["source"], voxel_size=float(spec.gate.get("voxel_size", 1.0))
        )
        reasons: list[str] = []
        for metric, gate_key in (
            ("plane_normal_dispersion", "max_plane_normal_dispersion"),
            ("coplanar_offset_rmse", "max_coplanar_offset_rmse"),
        ):
            threshold = spec.gate.get(gate_key)
            value = float(metrics[metric])
            if threshold is not None and (not math.isfinite(value) or value > float(threshold)):
                reasons.append(f"{metric} {value:.4f} > {gate_key} {float(threshold):.4f}")
        gate = None if not any(k.startswith("max_") for k in spec.gate) else {
            "passed": not reasons,
            "max_plane_normal_dispersion": spec.gate.get("max_plane_normal_dispersion"),
            "max_coplanar_offset_rmse": spec.gate.get("max_coplanar_offset_rmse"),
            "reasons": reasons,
        }
        metrics["quality_gate"] = gate
        if spec.outputs.report_path:
            report = (
                "# Experimental plane consistency\n\n"
                "> PNE/CPV-inspired proxy; not a reproduction of the published metrics.\n\n"
                f"- Plane normal dispersion: {metrics['plane_normal_dispersion']:.6f}\n"
                f"- Coplanar offset RMSE: {metrics['coplanar_offset_rmse']:.6f} m\n"
                f"- Plane patches: {metrics['num_plane_patches']}\n"
            )
            output = Path(spec.outputs.report_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report, encoding="utf-8")
        if spec.outputs.json_path:
            _write_json(spec.outputs.json_path, metrics)
        result = {
            "id": spec.check_id, "kind": spec.kind,
            "passed": None if gate is None else gate["passed"],
            "report_path": spec.outputs.report_path, "json_path": spec.outputs.json_path,
            "summary": {k: metrics[k] for k in ("plane_normal_dispersion", "coplanar_offset_rmse", "num_plane_patches")} | {"passed": None if gate is None else gate["passed"]},
            "result": metrics,
        }
    elif spec.kind == "uncertainty":
        from ca.core.uncertainty_evaluate import evaluate_uncertainty

        metrics = evaluate_uncertainty(
            spec.inputs["source"], spec.inputs["reference"],
            max_time_delta=spec.max_time_delta, align_mode=spec.alignment,
        )
        max_mean_nees = spec.gate.get("max_mean_position_nees")
        min_normalized_nees = spec.gate.get("min_normalized_mean_position_nees")
        min_coverage_95 = spec.gate.get("min_coverage_95")
        reasons: list[str] = []
        if max_mean_nees is not None and metrics["mean_position_nees"] > float(max_mean_nees):
            reasons.append(f"Mean position NEES {metrics['mean_position_nees']:.4f} > max_mean_position_nees {float(max_mean_nees):.4f}")
        if min_normalized_nees is not None and metrics["normalized_mean_position_nees"] < float(min_normalized_nees):
            reasons.append(f"Normalized mean position NEES {metrics['normalized_mean_position_nees']:.4f} < min_normalized_mean_position_nees {float(min_normalized_nees):.4f}")
        if min_coverage_95 is not None and metrics["coverage_95"] < float(min_coverage_95):
            reasons.append(f"95% coverage {metrics['coverage_95']:.1%} < min_coverage_95 {float(min_coverage_95):.1%}")
        gate = None if max_mean_nees is None and min_normalized_nees is None and min_coverage_95 is None else {
            "passed": not reasons, "max_mean_position_nees": max_mean_nees,
            "min_normalized_mean_position_nees": min_normalized_nees,
            "min_coverage_95": min_coverage_95, "reasons": reasons,
        }
        metrics["quality_gate"] = gate
        if spec.outputs.json_path:
            _write_json(spec.outputs.json_path, metrics)
        if spec.outputs.report_path:
            output = Path(spec.outputs.report_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                "# SLAM uncertainty consistency\n\n"
                f"- Mean position NEES: {metrics['mean_position_nees']:.6f}\n"
                f"- Normalized mean position NEES: {metrics['normalized_mean_position_nees']:.6f}\n"
                f"- 95% coverage: {metrics['coverage_95']:.1%}\n",
                encoding="utf-8",
            )
        result = {
            "id": spec.check_id, "kind": spec.kind,
            "passed": None if gate is None else gate["passed"],
            "report_path": spec.outputs.report_path, "json_path": spec.outputs.json_path,
            "summary": {key: metrics[key] for key in ("mean_position_nees", "normalized_mean_position_nees", "coverage_95", "num_matched_states")} | {"passed": None if gate is None else gate["passed"]},
            "result": metrics,
        }
    else:
        result = _run_run_batch_check(spec)
    result["severity"] = spec.severity
    result["gate_status"] = gate_status_for_check(result)
    return result


def run_check_suite(suite: CheckSuite, *, gate_mode: GateMode = "default") -> dict[str, Any]:
    """Execute every check in a normalized suite and aggregate pass/fail state."""
    executed_checks = [_run_check(spec) for spec in suite.checks]
    gate_summary = summarize_gate_policy(executed_checks, mode=gate_mode)
    gated_checks = [item for item in executed_checks if item["passed"] is not None]
    failed_checks = [item for item in gated_checks if item["passed"] is False]
    passed_checks = [item for item in gated_checks if item["passed"] is True]
    triage = summarize_failed_checks(executed_checks, project=suite.project)
    summary = {
        "total_checks": len(executed_checks),
        "gated_checks": len(gated_checks),
        "passed_checks": len(passed_checks),
        "failed_checks": len(failed_checks),
        "blocking_failed_checks": len(gate_summary["blocking_failed_ids"]),
        "unchecked_checks": len(executed_checks) - len(gated_checks),
        "failed_check_ids": [item["id"] for item in failed_checks],
        "blocking_failed_check_ids": gate_summary["blocking_failed_ids"],
        "passed": gate_summary["passed"],
        "gate_summary": gate_summary,
        "triage": triage,
    }
    result = {
        "config_path": suite.config_path,
        "project": suite.project,
        "gate_summary": gate_summary,
        "summary": summary,
        "checks": executed_checks,
    }
    if suite.summary_output_json:
        _write_json(suite.summary_output_json, result)
    return result
