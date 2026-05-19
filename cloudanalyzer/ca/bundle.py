"""Pack a CloudAnalyzer QA result into a reproducible ZIP bundle.

A *bundle* freezes one QA run — the summary JSON, the per-check reports
the summary points at, an optional baseline summary, and a metadata
header (commit / PR / runner / project / dataset notes) — into a single
``qa_bundle.zip`` artifact. Downstream consumers (long-term retention,
PR-comment formatters, future dashboards) can reopen it without going
back to the original CI workspace.

Two summary shapes are supported, mirroring :mod:`ca.pr_comment`:

- *check_suite*: ``summary["checks"]`` with ``report_path`` / ``json_path``
  per check. Each path the summary points at is copied into the bundle,
  preserving its file name under ``reports/<check_id>/``.
- *single_run*: ``ca run-evaluate`` / ``ca benchmark eval`` JSON. No
  associated artifact list, so the bundle just carries the summary and
  optional baseline.

Bundle layout::

    qa_bundle.zip
    ├── metadata.json              # versioned BundleMetadata
    ├── summary.json               # the input summary, untouched
    ├── baseline-summary.json      # only if --baseline was supplied
    └── reports/
        └── <check_id>/<file>      # per-check artifacts the summary referenced
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import ca


BUNDLE_FILENAME = "qa_bundle.zip"
METADATA_FILENAME = "metadata.json"
SUMMARY_FILENAME = "summary.json"
BASELINE_FILENAME = "baseline-summary.json"
REPORTS_DIR = "reports"

BUNDLE_VERSION = 1


@dataclass(slots=True)
class BundleArtifact:
    """One file referenced from the summary and copied into the bundle."""

    check_id: str
    field_name: str           # e.g. "report_path" or "json_path"
    archive_path: str         # path inside the zip
    source_path: str          # absolute path on the runner that produced the bundle
    size_bytes: int


@dataclass(slots=True)
class BundleMetadata:
    """Single-source-of-truth header inside the bundle."""

    bundle_version: int
    created_at: str  # ISO-8601 UTC
    cloudanalyzer_version: str
    summary_kind: str  # "check_suite" | "single_run"
    project: str | None = None
    git_commit: str | None = None
    pr_number: str | None = None
    runner_id: str | None = None
    notes: dict[str, str] = field(default_factory=dict)
    artifacts: list[BundleArtifact] = field(default_factory=list)
    has_baseline: bool = False


# --------------------------------------------------------------------- detection


def _detect_summary_kind(summary: Mapping[str, Any]) -> str:
    if isinstance(summary.get("summary"), Mapping) and isinstance(
        summary.get("checks"), Sequence
    ):
        return "check_suite"
    if "overall_quality_gate" in summary and (
        isinstance(summary.get("map"), Mapping)
        or isinstance(summary.get("trajectory"), Mapping)
    ):
        return "single_run"
    raise ValueError(
        "Unrecognized summary JSON shape. Expected `ca check` suite "
        "(`summary` + `checks`) or `ca run-evaluate` / `ca benchmark eval` "
        "single-run (`overall_quality_gate` + `map`/`trajectory`)."
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in: {path}")
    return data


def _safe_archive_name(check_id: str, source: Path) -> str:
    """Compose a stable path inside the archive.

    Uses ``reports/<check_id>/<filename>``. ``check_id`` is sanitized so a
    malicious / unusual check id can't break out of the reports tree.
    """
    safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in check_id)
    return f"{REPORTS_DIR}/{safe_id}/{source.name}"


# --------------------------------------------------------------------- pack


def _collect_artifacts(summary: Mapping[str, Any]) -> list[BundleArtifact]:
    """Walk a check-suite summary and collect referenced report/json files."""
    artifacts: list[BundleArtifact] = []
    checks = summary.get("checks")
    if not isinstance(checks, Sequence):
        return artifacts
    seen_archive_names: set[str] = set()
    for entry in checks:
        if not isinstance(entry, Mapping):
            continue
        check_id = str(entry.get("id", ""))
        for field_name in ("report_path", "json_path"):
            raw = entry.get(field_name)
            if not isinstance(raw, str) or not raw:
                continue
            source = Path(raw)
            if not source.is_file():
                # Skip silently — the original artifact may have been on a
                # different runner. metadata.notes can carry the omission
                # if a caller cares.
                continue
            archive = _safe_archive_name(check_id or "unknown", source)
            # Disambiguate collisions (same file name in two checks).
            base_archive = archive
            counter = 1
            while archive in seen_archive_names:
                stem = source.stem
                suffix = source.suffix
                archive = f"{REPORTS_DIR}/{check_id}/{stem}_{counter}{suffix}"
                counter += 1
            seen_archive_names.add(archive)
            try:
                size = source.stat().st_size
            except OSError:
                size = 0
            artifacts.append(
                BundleArtifact(
                    check_id=check_id,
                    field_name=field_name,
                    archive_path=archive,
                    source_path=str(source.resolve()),
                    size_bytes=size,
                )
            )
    return artifacts


def pack_bundle(
    summary_path: str | Path,
    output_path: str | Path,
    *,
    baseline_path: str | Path | None = None,
    project: str | None = None,
    git_commit: str | None = None,
    pr_number: str | None = None,
    runner_id: str | None = None,
    notes: Mapping[str, str] | None = None,
) -> BundleMetadata:
    """Pack a QA summary and its referenced artifacts into a single ZIP.

    Returns the metadata block written into the bundle so callers can
    log / display it without re-opening the archive.
    """
    summary_file = Path(summary_path).resolve()
    if not summary_file.is_file():
        raise FileNotFoundError(summary_file)
    summary = _load_json(summary_file)
    kind = _detect_summary_kind(summary)

    baseline_data: dict[str, Any] | None = None
    baseline_file: Path | None = None
    if baseline_path is not None:
        baseline_file = Path(baseline_path).resolve()
        if not baseline_file.is_file():
            raise FileNotFoundError(baseline_file)
        baseline_data = _load_json(baseline_file)
        # The baseline shape doesn't have to match exactly, but it's a
        # gotcha worth warning about loudly: detection should still succeed.
        _detect_summary_kind(baseline_data)

    artifacts = _collect_artifacts(summary) if kind == "check_suite" else []

    metadata = BundleMetadata(
        bundle_version=BUNDLE_VERSION,
        created_at=_now_iso(),
        cloudanalyzer_version=getattr(ca, "__version__", "0.0.0"),
        summary_kind=kind,
        project=project if project else (summary.get("project") if isinstance(summary.get("project"), str) else None),
        git_commit=git_commit,
        pr_number=pr_number,
        runner_id=runner_id,
        notes=dict(notes) if notes else {},
        artifacts=artifacts,
        has_baseline=baseline_data is not None,
    )

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            METADATA_FILENAME,
            json.dumps(_metadata_to_json(metadata), indent=2, sort_keys=False),
        )
        zf.writestr(SUMMARY_FILENAME, json.dumps(summary, indent=2))
        if baseline_data is not None:
            zf.writestr(BASELINE_FILENAME, json.dumps(baseline_data, indent=2))
        for artifact in artifacts:
            zf.write(artifact.source_path, artifact.archive_path)

    return metadata


def _metadata_to_json(metadata: BundleMetadata) -> dict[str, Any]:
    """Render a BundleMetadata to a stable JSON-friendly dict."""
    return {
        "bundle_version": metadata.bundle_version,
        "created_at": metadata.created_at,
        "cloudanalyzer_version": metadata.cloudanalyzer_version,
        "summary_kind": metadata.summary_kind,
        "project": metadata.project,
        "git_commit": metadata.git_commit,
        "pr_number": metadata.pr_number,
        "runner_id": metadata.runner_id,
        "notes": dict(metadata.notes),
        "has_baseline": metadata.has_baseline,
        "artifacts": [asdict(a) for a in metadata.artifacts],
    }


# --------------------------------------------------------------------- unpack


def unpack_bundle(
    bundle_path: str | Path,
    output_dir: str | Path,
) -> BundleMetadata:
    """Extract a bundle to ``output_dir``. Returns the embedded metadata."""
    bundle = Path(bundle_path).resolve()
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle, mode="r") as zf:
        # Validate before extraction so a malformed bundle errors cleanly.
        names = set(zf.namelist())
        if METADATA_FILENAME not in names:
            raise ValueError(f"{bundle}: missing {METADATA_FILENAME}")
        if SUMMARY_FILENAME not in names:
            raise ValueError(f"{bundle}: missing {SUMMARY_FILENAME}")

        # Reject absolute or parent-traversal paths up front.
        for name in names:
            normalized = name.replace("\\", "/")
            if normalized.startswith("/") or ".." in normalized.split("/"):
                raise ValueError(f"{bundle}: unsafe archive entry {name!r}")

        zf.extractall(out_dir)
        metadata_raw = json.loads((out_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
        return _metadata_from_json(metadata_raw)


def _metadata_from_json(raw: Mapping[str, Any]) -> BundleMetadata:
    artifacts = [
        BundleArtifact(
            check_id=str(a.get("check_id", "")),
            field_name=str(a.get("field_name", "")),
            archive_path=str(a.get("archive_path", "")),
            source_path=str(a.get("source_path", "")),
            size_bytes=int(a.get("size_bytes", 0) or 0),
        )
        for a in raw.get("artifacts", [])
        if isinstance(a, Mapping)
    ]
    return BundleMetadata(
        bundle_version=int(raw.get("bundle_version", 0) or 0),
        created_at=str(raw.get("created_at", "")),
        cloudanalyzer_version=str(raw.get("cloudanalyzer_version", "")),
        summary_kind=str(raw.get("summary_kind", "unknown")),
        project=raw.get("project") if isinstance(raw.get("project"), str) else None,
        git_commit=raw.get("git_commit") if isinstance(raw.get("git_commit"), str) else None,
        pr_number=raw.get("pr_number") if isinstance(raw.get("pr_number"), str) else None,
        runner_id=raw.get("runner_id") if isinstance(raw.get("runner_id"), str) else None,
        notes={
            str(k): str(v)
            for k, v in (raw.get("notes") or {}).items()
            if isinstance(k, str)
        },
        artifacts=artifacts,
        has_baseline=bool(raw.get("has_baseline", False)),
    )


# --------------------------------------------------------------------- show


def show_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Return the metadata + table of contents for a bundle without extracting."""
    bundle = Path(bundle_path).resolve()
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    with zipfile.ZipFile(bundle, mode="r") as zf:
        names = zf.namelist()
        if METADATA_FILENAME not in names:
            raise ValueError(f"{bundle}: missing {METADATA_FILENAME}")
        with zf.open(METADATA_FILENAME) as fp:
            metadata = json.loads(io.TextIOWrapper(fp, encoding="utf-8").read())
        infos = [zf.getinfo(name) for name in names]
    toc = [
        {
            "path": info.filename,
            "size_bytes": info.file_size,
            "compressed_bytes": info.compress_size,
        }
        for info in infos
    ]
    return {
        "bundle_path": str(bundle),
        "metadata": metadata,
        "contents": toc,
    }


__all__ = [
    "BUNDLE_FILENAME",
    "BUNDLE_VERSION",
    "BundleArtifact",
    "BundleMetadata",
    "pack_bundle",
    "show_bundle",
    "unpack_bundle",
]
