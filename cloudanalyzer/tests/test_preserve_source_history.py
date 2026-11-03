"""Tests for scripts/preserve_source_history.py."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import preserve_source_history as history  # noqa: E402

MEASURED = b"fn measured() {}\r\n"
OTHER = b"fn other() {}\n"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
         "-c", "core.autocrlf=false", *args],
        cwd=root, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _run(root: Path, *args: str) -> int:
    return history.main([*args, "--root", str(root)])


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is required")
    root = tmp_path / "repo"
    (root / "rust/src").mkdir(parents=True)
    (root / "benchmarks/packet").mkdir(parents=True)
    (root / "rust/src/core.rs").write_bytes(MEASURED)
    (root / "rust/src/other.rs").write_bytes(OTHER)
    (root / "rust/src/free.rs").write_bytes(b"fn free() {}\n")
    packet = {
        "normalizedSourceFootprint": {
            "rust/src/core.rs": history.normalized_sha256(MEASURED),
        },
        "inputs": [
            {"path": "rust/src/other.rs", "normalizedSha256": history.normalized_sha256(OTHER)},
        ],
    }
    (root / "benchmarks/packet/manifest.json").write_text(json.dumps(packet), encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "measured")
    return root


def test_unchanged_tree_needs_nothing(repo: Path) -> None:
    assert history.find_missing(repo, "HEAD") == []
    assert _run(repo, "--check", "--base", "HEAD") == 0


def test_changed_referenced_file_fails_check_until_retained(repo: Path, capsys) -> None:
    (repo / "rust/src/core.rs").write_bytes(b"fn faster() {}\n")
    digest = history.normalized_sha256(MEASURED)

    assert _run(repo, "--check", "--base", "HEAD") == 1
    assert f"rust/src/core.rs  {digest}" in capsys.readouterr().out

    assert _run(repo, "--base", "HEAD") == 0
    manifest = json.loads((repo / "benchmarks/source-history/manifest.json").read_text())
    entry = manifest["sources"][digest]
    assert entry == {
        "original": "rust/src/core.rs",
        "archived": f"benchmarks/source-history/{digest}/core.rs",
        "originCommit": _git(repo, "rev-parse", "HEAD"),
        "originGitBlob": _git(repo, "rev-parse", "HEAD:rust/src/core.rs"),
    }
    # The retained bytes are the measured ones, CRLF included.
    assert (repo / entry["archived"]).read_bytes() == MEASURED
    assert _run(repo, "--check", "--base", "HEAD") == 0


def test_committed_change_is_checked_against_its_parent(repo: Path) -> None:
    (repo / "rust/src/core.rs").write_bytes(b"fn faster() {}\n")
    _git(repo, "commit", "-q", "-am", "optimize")

    missing = history.find_missing(repo, "HEAD^1")
    assert [item["path"] for item in missing] == ["rust/src/core.rs"]


def test_object_references_and_deleted_files_are_detected(repo: Path) -> None:
    (repo / "rust/src/other.rs").unlink()

    assert history.find_missing(repo, "HEAD") == [
        {"path": "rust/src/other.rs", "sha256": history.normalized_sha256(OTHER)},
    ]


def test_unreferenced_and_new_files_are_ignored(repo: Path) -> None:
    (repo / "rust/src/free.rs").write_bytes(b"fn changed() {}\n")
    (repo / "rust/src/new.rs").write_bytes(b"fn new() {}\n")
    _git(repo, "add", "rust/src/new.rs")

    assert history.find_missing(repo, "HEAD") == []


def test_existing_archived_copy_counts_as_retained(repo: Path) -> None:
    archive = repo / "benchmarks/packet/executed-source/core.rs"
    archive.parent.mkdir()
    archive.write_bytes(MEASURED.replace(b"\r\n", b"\n"))
    (repo / "rust/src/core.rs").write_bytes(b"fn faster() {}\n")

    assert history.find_missing(repo, "HEAD") == []


def test_existing_manifest_entries_are_kept(repo: Path) -> None:
    manifest_path = repo / "benchmarks/source-history/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    kept = {"original": "web/src/app.ts", "archived": "benchmarks/source-history/x/app.ts",
            "originCommit": "0" * 40, "originGitBlob": "1" * 40}
    manifest_path.write_text(json.dumps({"schema": history.MANIFEST_SCHEMA,
                                         "purpose": "kept", "sources": {"a" * 64: kept}}))
    (repo / "rust/src/core.rs").write_bytes(b"fn faster() {}\n")

    assert _run(repo, "--base", "HEAD") == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["purpose"] == "kept"
    assert list(manifest["sources"]) == ["a" * 64, history.normalized_sha256(MEASURED)]
    assert manifest["sources"]["a" * 64] == kept
