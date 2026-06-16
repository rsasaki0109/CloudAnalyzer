"""Helpers for making generated public reports portable."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _replace_root_prefix(value: str, root: Path) -> str:
    """Replace absolute occurrences of ``root`` with repo-relative text."""
    root_text = root.resolve().as_posix()
    if value == root_text:
        return "."
    prefix = f"{root_text}/"
    return value.replace(prefix, "")


def make_paths_portable(data: Any, roots: Iterable[Path | str]) -> Any:
    """Return ``data`` with absolute paths under ``roots`` made relative.

    Public demo and leaderboard artifacts should be reproducible on any machine.
    Evaluation results still need to carry useful paths and copyable commands, so
    this recursively rewrites only known local roots instead of dropping path
    fields outright.
    """
    resolved_roots = tuple(Path(root).resolve() for root in roots)

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        if isinstance(value, str):
            portable = value
            for root in resolved_roots:
                portable = _replace_root_prefix(portable, root)
            return portable
        return value

    return convert(data)
