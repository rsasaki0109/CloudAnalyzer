"""Regression checks for checked-in public demo/report artifacts."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_TEXT_EXTENSIONS = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
LOCAL_PATH_PATTERNS = (
    "",
    "$HOME/",
    "old_~2026/CloudAnalyzer",
)
BROKEN_COMMAND_ONCLICK = re.compile(r"onclick='copyCommand\(\"[^\"]*'")


def _public_text_files() -> list[Path]:
    roots = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in PUBLIC_TEXT_EXTENSIONS
        )
    return sorted(files)


def test_public_text_artifacts_do_not_contain_local_absolute_paths():
    offenders: list[str] = []
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = [pattern for pattern in LOCAL_PATH_PATTERNS if pattern in text]
        if matches:
            rel = path.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{rel}: {', '.join(matches)}")

    assert offenders == []


def test_public_html_command_buttons_escape_onclick_payloads():
    offenders: list[str] = []
    for path in _public_text_files():
        if path.suffix.lower() != ".html":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if BROKEN_COMMAND_ONCLICK.search(text):
            rel = path.relative_to(REPO_ROOT).as_posix()
            offenders.append(rel)

    assert offenders == []
