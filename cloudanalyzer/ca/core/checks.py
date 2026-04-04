"""Config-driven QA checks for artifact, trajectory, and integrated run validation."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from ca.core.check_triage import summarize_failed_checks
from ca.batch import batch_evaluate, trajectory_batch_evaluate
from ca.evaluate import evaluate
from ca.report import (
    make_batch_summary,
    make_run_batch_summary,
    make_trajectory_batch_summary,
    save_batch_report,
    save_run_batch_report,
    save_run_report,
    save_trajectory_batch_report,
    save_trajectory_report,
)
from ca.run_evaluate import evaluate_run, evaluate_run_batch
from ca.trajectory import evaluate_trajectory


CheckKind = Literal[
    "artifact",
    "artifact_batch",
    "trajectory",
    "trajectory_batch",
    "run",
    "run_batch",
]
AlignmentMode = Literal["none", "origin", "rigid"]

_KIND_ALIASES: dict[str, CheckKind] = {
    "artifact": "artifact",
    "map": "artifact",
    "artifact_batch": "artifact_batch",
    "map_batch": "artifact_batch",
    "trajectory": "trajectory",
    "trajectory_batch": "trajectory_batch",
    "run": "run",
    "run_batch": "run_batch",
}

_VALID_GATE_KEYS = {
    "min_auc",
    "max_chamfer",
    "max_ate",
    "max_rpe",
    "max_drift",
    "min_coverage",
}

_ALLOWED_GATE_KEYS: dict[CheckKind, set[str]] = {
    "artifact": {"min_auc", "max_chamfer"},
    "artifact_batch": {"min_auc", "max_chamfer"},
    "trajectory": {"max_ate", "max_rpe", "max_drift", "min_coverage"},
    "trajectory_batch": {"max_ate", "max_rpe", "max_drift", "min_coverage"},
    "run": {
        "min_auc",
        "max_chamfer",
        "max_ate",
        "max_rpe",
        "max_drift",
        "min_coverage",
    },
    "run_batch": {
        "min_auc",
        "max_chamfer",
        "max_ate",
        "max_rpe",
        "max_drift",
        "min_coverage",
    },
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
    gate: dict[str, float] = field(default_factory=dict)
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
) -> dict[str, float]:
    """Merge top-level and per-check gate settings."""
    defaults_gate = _as_mapping(defaults.get("gate"), "defaults.gate")
    check_gate = _as_mapping(raw_check.get("gate"), "check.gate")
    merged = {**defaults_gate, **check_gate}
    unknown = set(merged) - _VALID_GATE_KEYS
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported gate key(s): {joined}")
    allowed = _ALLOWED_GATE_KEYS[kind]
    return {
        key: float(value)
        for key, value in merged.items()
        if key in allowed and value is not None
    }


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
    if kind == "run":
        return _normalize_run_inputs(raw_check, config_dir)
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
    thresholds = _normalize_thresholds(
        check.get("thresholds", defaults.get("thresholds")),
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


def _artifact_quality_gate(
    auc: float,
    chamfer_distance: float,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
) -> dict[str, Any] | None:
    """Build optional pass/fail metadata for artifact-style checks."""
    if min_auc is None and max_chamfer is None:
        return None
    reasons: list[str] = []
    if min_auc is not None and auc < min_auc:
        reasons.append(f"AUC {auc:.4f} < min_auc {min_auc:.4f}")
    if max_chamfer is not None and chamfer_distance > max_chamfer:
        reasons.append(f"Chamfer {chamfer_distance:.4f} > max_chamfer {max_chamfer:.4f}")
    return {
        "passed": not reasons,
        "min_auc": min_auc,
        "max_chamfer": max_chamfer,
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


def _artifact_batch_item(spec: CheckSpec, result: dict[str, Any]) -> dict[str, Any]:
    """Convert a single evaluate() result into the batch/report shape."""
    best_f1 = max(result["f1_scores"], key=lambda score: score["f1"])
    gate = _artifact_quality_gate(
        result["auc"],
        result["chamfer_distance"],
        min_auc=cast(float | None, spec.gate.get("min_auc")),
        max_chamfer=cast(float | None, spec.gate.get("max_chamfer")),
    )
    return {
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


def _run_artifact_check(spec: CheckSpec) -> dict[str, Any]:
    """Run a single artifact QA check."""
    result = evaluate(
        spec.inputs["source"],
        spec.inputs["reference"],
        thresholds=list(spec.thresholds) if spec.thresholds else None,
    )
    batch_item = _artifact_batch_item(spec, result)
    result_with_summary = {
        **result,
        "best_f1": batch_item["best_f1"],
        "quality_gate": batch_item["quality_gate"],
        "inspect": batch_item["inspect"],
    }
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


def _run_check(spec: CheckSpec) -> dict[str, Any]:
    """Dispatch one normalized check spec."""
    if spec.kind == "artifact":
        return _run_artifact_check(spec)
    if spec.kind == "artifact_batch":
        return _run_artifact_batch_check(spec)
    if spec.kind == "trajectory":
        return _run_trajectory_check(spec)
    if spec.kind == "trajectory_batch":
        return _run_trajectory_batch_check(spec)
    if spec.kind == "run":
        return _run_run_check(spec)
    return _run_run_batch_check(spec)


def run_check_suite(suite: CheckSuite) -> dict[str, Any]:
    """Execute every check in a normalized suite and aggregate pass/fail state."""
    executed_checks = [_run_check(spec) for spec in suite.checks]
    gated_checks = [item for item in executed_checks if item["passed"] is not None]
    failed_checks = [item for item in gated_checks if item["passed"] is False]
    passed_checks = [item for item in gated_checks if item["passed"] is True]
    triage = summarize_failed_checks(executed_checks, project=suite.project)
    summary = {
        "total_checks": len(executed_checks),
        "gated_checks": len(gated_checks),
        "passed_checks": len(passed_checks),
        "failed_checks": len(failed_checks),
        "unchecked_checks": len(executed_checks) - len(gated_checks),
        "failed_check_ids": [item["id"] for item in failed_checks],
        "passed": len(failed_checks) == 0,
        "triage": triage,
    }
    result = {
        "config_path": suite.config_path,
        "project": suite.project,
        "summary": summary,
        "checks": executed_checks,
    }
    if suite.summary_output_json:
        _write_json(suite.summary_output_json, result)
    return result
