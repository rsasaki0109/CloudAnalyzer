"""SLAM benchmark suite runner.

A *benchmark suite* freezes a reference map, reference trajectory, and a
quality gate so users can plug their own SLAM output in for a one-command
quality check. ``evaluate_benchmark_run`` delegates to
:func:`ca.run_evaluate.evaluate_run` with the suite's references and gate
values applied.

The suite manifest schema (YAML) lives next to the data:

.. code-block:: yaml

   version: 1
   name: synthetic-figure8
   description: ...
   license: MIT (synthetic)
   sequences:
     default:
       description: ...
       reference_map: reference/map.pcd
       reference_trajectory: reference/trajectory.tum
   sample_outputs:        # optional — used by docs / tests
     default:
       map: sample_outputs/map_pass.pcd
       trajectory: sample_outputs/trajectory_pass.tum
   gate:
     min_auc: 0.90
     max_chamfer: 0.05
     max_ate: 0.30
     max_rpe: 0.10
     max_drift: 0.50
     min_coverage: 0.90

All file paths in the manifest are resolved relative to the manifest file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]

from ca.run_evaluate import evaluate_run


GATE_KEYS: tuple[str, ...] = (
    "min_auc",
    "max_chamfer",
    "max_ate",
    "max_rpe",
    "max_drift",
    "min_coverage",
)


@dataclass(frozen=True, slots=True)
class BenchmarkSequence:
    name: str
    description: str
    reference_map_path: Path
    reference_trajectory_path: Path
    sample_map_path: Path | None = None
    sample_trajectory_path: Path | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    name: str
    version: int
    description: str
    license: str | None
    source_path: Path
    sequences: dict[str, BenchmarkSequence] = field(default_factory=dict)
    gate: dict[str, float] = field(default_factory=dict)

    def resolve_sequence(self, sequence: str | None) -> BenchmarkSequence:
        if not self.sequences:
            raise ValueError(f"Benchmark suite {self.name!r} has no sequences.")
        if sequence is None:
            return next(iter(self.sequences.values()))
        if sequence not in self.sequences:
            available = ", ".join(sorted(self.sequences))
            raise ValueError(
                f"Sequence {sequence!r} not in benchmark suite {self.name!r}. "
                f"Available: {available}"
            )
        return self.sequences[sequence]


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping; got {type(value).__name__}")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _resolve_path(base_dir: Path, raw: Any, name: str) -> Path:
    rel = _require_string(raw, name)
    resolved = (base_dir / rel).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    return resolved


def _optional_path(base_dir: Path, raw: Any, name: str) -> Path | None:
    if raw is None:
        return None
    return _resolve_path(base_dir, raw, name)


def _parse_gate(raw: Any) -> dict[str, float]:
    mapping = _as_mapping(raw, "gate")
    gate: dict[str, float] = {}
    for key, value in mapping.items():
        if key not in GATE_KEYS:
            raise ValueError(
                f"Unsupported gate key {key!r}. Allowed: {', '.join(GATE_KEYS)}"
            )
        if value is None:
            continue
        try:
            gate[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"gate.{key} must be numeric; got {value!r}") from exc
    return gate


def _parse_sequences(
    raw_sequences: Any,
    raw_sample_outputs: Any,
    base_dir: Path,
) -> dict[str, BenchmarkSequence]:
    sequences_map = _as_mapping(raw_sequences, "sequences")
    if not sequences_map:
        raise ValueError("Benchmark suite must declare at least one sequence.")
    sample_map = _as_mapping(raw_sample_outputs, "sample_outputs")

    out: dict[str, BenchmarkSequence] = {}
    for name, raw in sequences_map.items():
        entry = _as_mapping(raw, f"sequences.{name}")
        description = _require_string(
            entry.get("description", ""),
            f"sequences.{name}.description",
        )
        reference_map = _resolve_path(
            base_dir,
            entry.get("reference_map"),
            f"sequences.{name}.reference_map",
        )
        reference_trajectory = _resolve_path(
            base_dir,
            entry.get("reference_trajectory"),
            f"sequences.{name}.reference_trajectory",
        )
        sample_entry = _as_mapping(sample_map.get(name, {}), f"sample_outputs.{name}")
        sample_map_path = _optional_path(
            base_dir,
            sample_entry.get("map"),
            f"sample_outputs.{name}.map",
        )
        sample_traj_path = _optional_path(
            base_dir,
            sample_entry.get("trajectory"),
            f"sample_outputs.{name}.trajectory",
        )
        out[name] = BenchmarkSequence(
            name=name,
            description=description,
            reference_map_path=reference_map,
            reference_trajectory_path=reference_trajectory,
            sample_map_path=sample_map_path,
            sample_trajectory_path=sample_traj_path,
        )
    return out


def load_benchmark_suite(path: str | Path) -> BenchmarkSuite:
    """Load a benchmark suite manifest from a YAML file."""
    manifest_path = Path(path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError("Benchmark suite file is empty")
    config = _as_mapping(raw, "config")

    version = config.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported benchmark suite version: {version!r}")

    name = _require_string(config.get("name"), "name")
    description = _require_string(config.get("description"), "description")
    license_value = config.get("license")
    license_str = (
        _require_string(license_value, "license") if license_value is not None else None
    )

    base_dir = manifest_path.parent
    sequences = _parse_sequences(
        config.get("sequences"),
        config.get("sample_outputs"),
        base_dir,
    )
    gate = _parse_gate(config.get("gate"))

    return BenchmarkSuite(
        name=name,
        version=int(version),
        description=description,
        license=license_str,
        sequences=sequences,
        gate=gate,
        source_path=manifest_path,
    )


def _merge_gate(
    suite_gate: Mapping[str, float],
    overrides: Mapping[str, float | None] | None,
) -> dict[str, float]:
    merged = dict(suite_gate)
    if not overrides:
        return merged
    for key, value in overrides.items():
        if key not in GATE_KEYS:
            raise ValueError(
                f"Unsupported gate override {key!r}. Allowed: {', '.join(GATE_KEYS)}"
            )
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = float(value)
    return merged


def evaluate_benchmark_run(
    suite: BenchmarkSuite,
    map_path: str,
    trajectory_path: str,
    *,
    sequence: str | None = None,
    gate_overrides: Mapping[str, float | None] | None = None,
    thresholds: list[float] | None = None,
    max_time_delta: float = 0.05,
    align_origin: bool = False,
    align_rigid: bool = False,
) -> dict[str, Any]:
    """Evaluate a SLAM run against a benchmark suite's reference and gate.

    The returned dict matches ``evaluate_run`` and carries an extra
    ``benchmark`` block with the suite identity, sequence, and applied gate
    so reports and CI summaries can recover the calibration source.
    """
    seq = suite.resolve_sequence(sequence)
    gate = _merge_gate(suite.gate, gate_overrides)
    result = evaluate_run(
        map_path,
        str(seq.reference_map_path),
        trajectory_path,
        str(seq.reference_trajectory_path),
        thresholds=thresholds,
        max_time_delta=max_time_delta,
        align_origin=align_origin,
        align_rigid=align_rigid,
        **gate,
    )
    result["benchmark"] = {
        "suite": suite.name,
        "version": suite.version,
        "sequence": seq.name,
        "source_path": str(suite.source_path),
        "gate": gate,
        "reference": {
            "map": str(seq.reference_map_path),
            "trajectory": str(seq.reference_trajectory_path),
        },
    }
    return result
