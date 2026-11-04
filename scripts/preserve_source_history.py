#!/usr/bin/env python3
"""Keep source versions that frozen benchmark packets still reference.

Benchmark packets under ``benchmarks/`` record the normalized SHA-256 of the
source files they measured. When one of those files changes, the measured
version must stay resolvable or the packet tests fail. Packet tests look up
retained versions in ``benchmarks/source-history/manifest.json``.

This script compares a base revision with the working tree. For every changed
or deleted file whose *base* version is referenced by a packet and is not yet
retained, it copies the base bytes into ``benchmarks/source-history``.

    # before or after editing sources on a branch
    python scripts/preserve_source_history.py --base origin/main

    # CI: fail when a change drops a referenced version
    python scripts/preserve_source_history.py --check --base HEAD^1

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

HISTORY_DIR = "benchmarks/source-history"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = "cloudanalyzer.preserved_source.v1"
MANIFEST_PURPOSE = (
    "Retain exact source for historical benchmark verification while current "
    "application code evolves. Existing measured packets and hashes are unchanged."
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_REPO_PATH = re.compile(r"[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)+")
_PATH_FIELDS = ("path", "source", "file")
_HASH_FIELDS = ("normalizedSha256", "normalized_sha256", "sha256", "source_sha256")


def normalized_sha256(data: bytes) -> str:
    """Hash text with CRLF folded to LF, matching the packet tests."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True
    ).stdout


def changed_paths(root: Path, base: str) -> list[str]:
    """Tracked files that differ between ``base`` and the working tree."""
    out = _git(root, "diff", "--name-only", "--no-renames", "-z", base, "--")
    paths = [p for p in out.decode("utf-8").split("\0") if p]
    return sorted(p for p in paths if not p.startswith("benchmarks/"))


def base_bytes(root: Path, base: str, path: str) -> bytes | None:
    """Return the file at ``base``, or None when it did not exist there."""
    try:
        return _git(root, "show", f"{base}:{path}")
    except subprocess.CalledProcessError:
        return None


def _is_repo_path(value: str) -> bool:
    return (
        _REPO_PATH.fullmatch(value) is not None
        and ".." not in value.split("/")
        and not value.startswith("benchmarks/")
    )


def _pairs(node: Any) -> Iterator[tuple[str, str]]:
    """Yield (repo path, sha256) references found anywhere in a JSON value."""
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for field in _PATH_FIELDS:
                path = item.get(field)
                if isinstance(path, str) and _is_repo_path(path):
                    for hash_field in _HASH_FIELDS:
                        digest = item.get(hash_field)
                        if isinstance(digest, str) and _HEX64.fullmatch(digest):
                            yield path, digest
            for key, value in item.items():
                if isinstance(value, str):
                    if _HEX64.fullmatch(value) and _is_repo_path(key):
                        yield key, value
                else:
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)


def referenced_hashes(root: Path, candidates: Mapping[str, str]) -> dict[str, set[str]]:
    """Map each candidate path to the hashes benchmark JSON records for it.

    ``candidates`` maps a path to the digest of its base version. A hash counts
    when JSON pairs it with the path, or when the base digest appears anywhere:
    some packets name a measured file by its role (``"core": <sha256>``) rather
    than its path, and an exact SHA-256 match can only mean that version.
    """
    wanted = set(candidates)
    path_needles = [p.encode("utf-8") for p in wanted]
    digest_needles = {path: digest.encode("ascii") for path, digest in candidates.items()}
    found: dict[str, set[str]] = {p: set() for p in wanted}
    if not wanted:
        return found
    manifest = root / HISTORY_DIR / MANIFEST_NAME
    for json_path in sorted((root / "benchmarks").rglob("*.json")):
        if json_path == manifest:
            continue
        raw = json_path.read_bytes()
        for path, needle in digest_needles.items():
            if needle in raw:
                found[path].add(candidates[path])
        if not any(needle in raw for needle in path_needles):
            continue
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        for path, digest in _pairs(document):
            if path in wanted:
                found[path].add(digest)
    return found


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / HISTORY_DIR / MANIFEST_NAME
    if not path.exists():
        return {"schema": MANIFEST_SCHEMA, "purpose": MANIFEST_PURPOSE, "sources": {}}
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def _archived_hashes(root: Path, suffixes: set[str]) -> set[str]:
    """Normalized hashes of every retained file under ``benchmarks/``."""
    hashes: set[str] = set()
    for path in (root / "benchmarks").rglob("*"):
        if path.suffix in suffixes and path.is_file() and not path.is_symlink():
            hashes.add(normalized_sha256(path.read_bytes()))
    return hashes


def find_missing(root: Path, base: str) -> list[dict[str, str]]:
    """Referenced base versions of changed files that nothing retains."""
    candidates: dict[str, tuple[str, bytes]] = {}
    for path in changed_paths(root, base):
        data = base_bytes(root, base, path)
        if data is not None:
            candidates[path] = (normalized_sha256(data), data)
    references = referenced_hashes(root, {p: c[0] for p, c in candidates.items()})
    needed = {p: c for p, c in candidates.items() if c[0] in references[p]}
    if not needed:
        return []
    sources = load_manifest(root)["sources"]
    archived = _archived_hashes(root, {Path(p).suffix for p in needed})
    missing = []
    for path, (digest, _) in sorted(needed.items()):
        retained = sources.get(digest, {}).get("original") == path
        if not retained and digest not in archived:
            missing.append({"path": path, "sha256": digest})
    return missing


def preserve(root: Path, base: str, missing: list[dict[str, str]]) -> list[dict[str, str]]:
    """Copy each missing base version into the source history."""
    if not missing:
        return []
    manifest = load_manifest(root)
    commit = _git(root, "rev-parse", "--verify", f"{base}^{{commit}}").decode().strip()
    entries = []
    for item in missing:
        path, digest = item["path"], item["sha256"]
        data = base_bytes(root, base, path)
        if data is None or normalized_sha256(data) != digest:
            raise RuntimeError(f"{base}:{path} no longer has SHA-256 {digest}")
        archived = f"{HISTORY_DIR}/{digest}/{Path(path).name}"
        target = root / archived
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        blob = _git(root, "rev-parse", f"{base}:{path}").decode().strip()
        entry = {"original": path, "archived": archived,
                 "originCommit": commit, "originGitBlob": blob}
        manifest["sources"][digest] = entry
        entries.append(entry)
    manifest_path = root / HISTORY_DIR / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Keep source versions that frozen benchmark packets still reference."
    )
    parser.add_argument("--base", default="origin/main",
                        help="Revision to compare the working tree with (default: origin/main)")
    parser.add_argument("--check", action="store_true",
                        help="Report missing versions and exit 1 instead of archiving them")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()

    missing = find_missing(root, args.base)
    if not missing:
        print("Every changed source version referenced by a benchmark packet is retained.")
        return 0
    if args.check:
        print("These changed files had versions that benchmark packets still reference,")
        print("but no retained copy exists, so packet tests will fail:")
        for item in missing:
            print(f"  {item['path']}  {item['sha256']}")
        print()
        print("Retain them and commit the result:")
        print(f"  python scripts/preserve_source_history.py --base {args.base}")
        return 1
    for entry in preserve(root, args.base, missing):
        print(f"Retained {entry['original']} -> {entry['archived']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
