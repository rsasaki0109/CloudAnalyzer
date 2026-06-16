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

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]

import ca
from ca.report_paths import make_paths_portable
from ca.run_evaluate import evaluate_run


GATE_KEYS: tuple[str, ...] = (
    "min_auc",
    "max_chamfer",
    "max_ate",
    "max_rpe",
    "max_drift",
    "min_coverage",
)

REPORT_BUNDLE_SCHEMA_VERSION = "cloudanalyzer.benchmark_report_bundle.v0.1"


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


# ----------------------------------------------------------- report bundle


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: str | Path, roots: tuple[Path, ...]) -> dict[str, Any]:
    resolved = Path(path).resolve()
    portable_path = make_paths_portable(str(resolved), roots)
    return {
        "path": portable_path,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _bundle_roots(
    suite: BenchmarkSuite,
    output_dir: Path,
    extra_roots: tuple[Path | str, ...] = (),
) -> tuple[Path, ...]:
    roots: list[Path] = []
    for candidate in (
        Path.cwd(),
        suite.source_path.parent,
        output_dir.parent,
        *extra_roots,
    ):
        path = Path(candidate).resolve()
        if path not in roots:
            roots.append(path)
    return tuple(roots)


def _bundle_lock_payload(
    *,
    suite: BenchmarkSuite,
    sequence: BenchmarkSequence,
    result: Mapping[str, Any],
    output_dir: Path,
    map_path: str | Path,
    trajectory_path: str | Path,
    report_assets: list[str],
    roots: tuple[Path, ...],
) -> dict[str, Any]:
    benchmark = result.get("benchmark")
    benchmark_map = benchmark if isinstance(benchmark, Mapping) else {}
    return {
        "schema_version": REPORT_BUNDLE_SCHEMA_VERSION,
        "suite": {
            "name": suite.name,
            "version": suite.version,
            "description": suite.description,
            "license": suite.license,
            "sequence": sequence.name,
            "source_path": make_paths_portable(str(suite.source_path), roots),
        },
        "gate": dict(benchmark_map.get("gate", suite.gate)),
        "inputs": {
            "candidate_map": _file_provenance(map_path, roots),
            "candidate_trajectory": _file_provenance(trajectory_path, roots),
            "reference_map": _file_provenance(sequence.reference_map_path, roots),
            "reference_trajectory": _file_provenance(
                sequence.reference_trajectory_path,
                roots,
            ),
            "suite_manifest": _file_provenance(suite.source_path, roots),
        },
        "outputs": {
            "metrics": "metrics.json",
            "summary": "summary.md",
            "report": "report.html",
            "report_assets": report_assets,
            "provenance": "provenance.json",
            "manifest_lock": "manifest.lock.yaml",
        },
        "bundle_path": make_paths_portable(str(output_dir), roots),
    }


def write_benchmark_report_bundle(
    result: Mapping[str, Any],
    suite: BenchmarkSuite,
    output_dir: str | Path,
    *,
    map_path: str | Path,
    trajectory_path: str | Path,
    sequence: str | None = None,
    extra_roots: tuple[Path | str, ...] = (),
) -> dict[str, str]:
    """Write the standard ``ca benchmark eval --out`` report bundle.

    The bundle is a directory contract intended for CI artifacts and static
    publishing. It is intentionally simple and inspectable:

    - ``metrics.json``: portable benchmark result JSON
    - ``summary.md``: PR-comment-ready Markdown summary
    - ``report.html``: human report
    - ``manifest.lock.yaml``: suite/gate/input hash lock
    - ``provenance.json``: machine-readable bundle metadata
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    roots = _bundle_roots(suite, out.resolve(), extra_roots=extra_roots)
    sequence_name = sequence
    benchmark = result.get("benchmark")
    if sequence_name is None and isinstance(benchmark, Mapping):
        raw_sequence = benchmark.get("sequence")
        if isinstance(raw_sequence, str):
            sequence_name = raw_sequence
    seq = suite.resolve_sequence(sequence_name)

    portable_result = make_paths_portable(dict(result), roots)
    metrics_path = out / "metrics.json"
    metrics_path.write_text(
        json.dumps(portable_result, indent=2),
        encoding="utf-8",
    )

    from ca.pr_comment import build_pr_comment  # local import avoids cold CLI cycles
    from ca.report import save_run_report

    summary_path = out / "summary.md"
    summary_path.write_text(build_pr_comment(portable_result), encoding="utf-8")

    report_path = out / "report.html"
    save_run_report(portable_result, report_path)
    report_assets = sorted(
        path.name
        for path in out.iterdir()
        if path.is_file()
        and path.name
        not in {
            "manifest.lock.yaml",
            "metrics.json",
            "provenance.json",
            "report.html",
            "summary.md",
        }
    )

    lock_payload = _bundle_lock_payload(
        suite=suite,
        sequence=seq,
        result=portable_result,
        output_dir=out.resolve(),
        map_path=map_path,
        trajectory_path=trajectory_path,
        report_assets=report_assets,
        roots=roots,
    )
    manifest_lock_path = out / "manifest.lock.yaml"
    manifest_lock_path.write_text(
        yaml.safe_dump(lock_payload, sort_keys=False),
        encoding="utf-8",
    )

    provenance = {
        "schema_version": REPORT_BUNDLE_SCHEMA_VERSION,
        "cloudanalyzer_version": getattr(ca, "__version__", "0.0.0"),
        "summary_kind": "benchmark_run",
        "benchmark": portable_result.get("benchmark"),
        "overall_quality_gate": portable_result.get("overall_quality_gate"),
        "artifacts": lock_payload["outputs"],
        "inputs": lock_payload["inputs"],
    }
    provenance_path = out / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    return {
        "bundle_dir": str(out),
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "report": str(report_path),
        "manifest_lock": str(manifest_lock_path),
        "provenance": str(provenance_path),
    }


# ----------------------------------------------------------- materialize


def _voxel_downsample_pcd(source: Path, target: Path, voxel_size: float) -> None:
    """Voxel-downsample a point cloud file using Open3D. No-op if voxel<=0."""
    import open3d as o3d  # local import: avoid open3d on cold benchmark path

    target.parent.mkdir(parents=True, exist_ok=True)
    if voxel_size <= 0:
        # Plain copy when no downsampling requested.
        target.write_bytes(source.read_bytes())
        return
    pcd = o3d.io.read_point_cloud(str(source))
    if len(pcd.points) == 0:
        raise ValueError(f"{source} contains no points")
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    if not o3d.io.write_point_cloud(str(target), downsampled, write_ascii=False):
        raise RuntimeError(f"Failed to write downsampled point cloud: {target}")


def _subsample_tum(source: Path, target: Path, max_poses: int | None) -> None:
    """Copy a TUM trajectory; if max_poses is set, keep an evenly-spaced subset."""
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    # Drop blank lines and full-line comments; preserve trailing whitespace on data lines.
    lines = [
        line for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"{source} contains no TUM poses")
    if max_poses is not None and max_poses > 0 and len(lines) > max_poses:
        # Evenly-spaced indices; always keep first and last.
        import numpy as np

        raw_indices = np.linspace(0, len(lines) - 1, num=max_poses).round().astype(int)
        keep_indices = sorted({int(i) for i in raw_indices})
        kept = [lines[i] for i in keep_indices]
    else:
        kept = lines
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")


def materialize_suite(
    suite_dir: str | Path,
    *,
    name: str,
    description: str,
    reference_map: str | Path,
    reference_trajectory: str | Path,
    sequence_name: str = "default",
    sequence_description: str | None = None,
    license: str | None = None,
    voxel_size: float = 0.0,
    max_poses: int | None = None,
    gate: Mapping[str, float] | None = None,
    sample_map: str | Path | None = None,
    sample_trajectory: str | Path | None = None,
) -> BenchmarkSuite:
    """Build a benchmark suite from raw reference data on disk.

    Materializes the supplied map / trajectory (and optionally a sample
    output pair) under ``<suite_dir>/data/`` and writes a ``suite.yaml``
    that ``load_benchmark_suite`` can read. The output layout is::

        <suite_dir>/
        ├── suite.yaml
        └── data/
            ├── <sequence_name>/map.pcd
            ├── <sequence_name>/trajectory.tum
            └── sample_outputs/<sequence_name>/{map.pcd,trajectory.tum}  # optional

    Use this to convert a public SLAM dataset (e.g. Newer College Dataset
    ground-truth bundle) into a ``ca benchmark eval``-ready suite without
    hand-writing the manifest. ``voxel_size > 0`` downsamples the map,
    ``max_poses`` keeps an evenly-spaced subset of the trajectory.
    """
    suite_dir_path = Path(suite_dir).resolve()
    suite_dir_path.mkdir(parents=True, exist_ok=True)

    data_dir = suite_dir_path / "data" / sequence_name
    map_target = data_dir / "map.pcd"
    traj_target = data_dir / "trajectory.tum"

    _voxel_downsample_pcd(Path(reference_map), map_target, voxel_size)
    _subsample_tum(Path(reference_trajectory), traj_target, max_poses)

    sample_section: dict[str, dict[str, str]] | None = None
    if sample_map is not None and sample_trajectory is not None:
        sample_target_dir = suite_dir_path / "data" / "sample_outputs" / sequence_name
        sample_map_target = sample_target_dir / "map.pcd"
        sample_traj_target = sample_target_dir / "trajectory.tum"
        _voxel_downsample_pcd(Path(sample_map), sample_map_target, voxel_size)
        _subsample_tum(Path(sample_trajectory), sample_traj_target, max_poses)
        sample_section = {
            sequence_name: {
                "map": str(sample_map_target.relative_to(suite_dir_path)),
                "trajectory": str(sample_traj_target.relative_to(suite_dir_path)),
            }
        }

    manifest: dict[str, Any] = {
        "version": 1,
        "name": name,
        "description": description,
        "sequences": {
            sequence_name: {
                "description": sequence_description or description,
                "reference_map": str(map_target.relative_to(suite_dir_path)),
                "reference_trajectory": str(
                    traj_target.relative_to(suite_dir_path)
                ),
            }
        },
    }
    if license:
        manifest["license"] = license
    if sample_section is not None:
        manifest["sample_outputs"] = sample_section
    if gate:
        manifest["gate"] = {
            key: float(value) for key, value in gate.items() if key in GATE_KEYS
        }

    suite_yaml_path = suite_dir_path / "suite.yaml"
    suite_yaml_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    return load_benchmark_suite(suite_yaml_path)


# Add to __all__
__all__: list[str] = [
    "BenchmarkSequence",
    "BenchmarkSuite",
    "GATE_KEYS",
    "REPORT_BUNDLE_SCHEMA_VERSION",
    "evaluate_benchmark_run",
    "load_benchmark_suite",
    "materialize_suite",
    "write_benchmark_report_bundle",
]
