"""Baseline history management: save, discover, and rotate QA summaries."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _timestamp_label() -> str:
    """Generate a UTC timestamp label for baseline naming."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def discover_history(history_dir: str, pattern: str = "*.json") -> list[str]:
    """Find and sort baseline summary JSONs in a directory, oldest first.

    Files are sorted by name (which embeds a timestamp when saved via
    ``save_baseline``).  Non-JSON files and files that fail to parse are
    silently skipped.
    """
    dir_path = Path(history_dir)
    if not dir_path.is_dir():
        return []
    paths: list[Path] = sorted(dir_path.glob(pattern))
    valid: list[str] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                valid.append(str(path))
        except (json.JSONDecodeError, OSError):
            continue
    return valid


def save_baseline(
    summary_path: str,
    history_dir: str,
    label: str | None = None,
) -> str:
    """Copy a QA summary JSON into the history directory with a timestamped name.

    Returns the destination path.
    """
    source = Path(summary_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Summary file not found: {source}")

    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in: {source}")

    dest_dir = Path(history_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stem = label or _timestamp_label()
    dest = dest_dir / f"baseline-{stem}.json"
    shutil.copy2(str(source), str(dest))
    return str(dest)


def rotate_history(history_dir: str, keep: int = 10) -> list[str]:
    """Remove oldest baselines beyond the keep limit.

    Returns the list of removed file paths.
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")
    all_paths = discover_history(history_dir)
    if len(all_paths) <= keep:
        return []
    to_remove = all_paths[: len(all_paths) - keep]
    for path in to_remove:
        Path(path).unlink(missing_ok=True)
    return to_remove


def list_baselines(history_dir: str) -> list[dict]:
    """Return metadata for each baseline in the history directory."""
    paths = discover_history(history_dir)
    entries: list[dict] = []
    for path in paths:
        file_path = Path(path)
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summary = data.get("summary", {})
        entries.append({
            "path": str(file_path),
            "name": file_path.name,
            "passed": summary.get("passed"),
            "project": data.get("project"),
            "failed_check_ids": summary.get("failed_check_ids", []),
        })
    return entries
