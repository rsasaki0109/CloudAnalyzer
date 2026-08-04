"""Versioned evaluation protocol loading and provenance helpers.

The metric implementations in CloudAnalyzer already expose a few local
protocol descriptions (for example MapEval and rendered 3DGS evaluation).
This module provides the small common layer that lets a caller declare the
conventions once and carry that declaration through a JSON result.

The protocol file is deliberately data-only.  It describes conventions and
parameters; the ``run`` block added to an evaluation result records the
actual command, input manifests, and runtime without changing the protocol's
stable identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ca import __version__


PROTOCOL_SCHEMA_VERSION = "cloudanalyzer.protocol.v1"

_DEFAULT_CONVENTIONS: dict[str, Any] = {
    "coordinate_frame": "unspecified",
    "length_unit": "m",
    "time_unit": "s",
    "alignment": "none",
    "association": "unspecified",
}


class ProtocolValidationError(ValueError):
    """Raised when a protocol document cannot be compared safely."""


def _canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in a stable form for hashing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"{name} must be a mapping/object")
    return dict(value)


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _protocol_identity_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(document))
    payload.pop("protocol_sha256", None)
    payload.pop("protocol_id", None)
    return payload


def _protocol_digest(document: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(_protocol_identity_payload(document)).encode("utf-8"))


def validate_protocol(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a ``cloudanalyzer.protocol.v1`` document.

    Unknown fields are preserved so a future protocol revision can add
    metadata without making older readers discard it.  The required fields
    and the comparison-critical convention fields are intentionally strict.
    """

    raw = _require_mapping(document, "protocol")
    schema_version = _require_text(raw.get("schema_version"), "schema_version")
    if schema_version != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolValidationError(
            f"Unsupported schema_version {schema_version!r}; "
            f"expected {PROTOCOL_SCHEMA_VERSION!r}"
        )

    normalized = copy.deepcopy(raw)
    normalized["schema_version"] = PROTOCOL_SCHEMA_VERSION
    normalized["name"] = _require_text(raw.get("name"), "name")
    normalized["kind"] = _require_text(raw.get("kind"), "kind")

    conventions = dict(_DEFAULT_CONVENTIONS)
    conventions.update(_require_mapping(raw.get("conventions", {}), "conventions"))
    for key in ("coordinate_frame", "length_unit", "time_unit", "alignment", "association"):
        conventions[key] = _require_text(conventions.get(key), f"conventions.{key}")
    normalized["conventions"] = conventions

    normalized["parameters"] = _require_mapping(raw.get("parameters", {}), "parameters")
    provenance = _require_mapping(raw.get("provenance", {}), "provenance")
    if "hash_inputs" in provenance and not isinstance(provenance["hash_inputs"], bool):
        raise ProtocolValidationError("provenance.hash_inputs must be a boolean")
    normalized["provenance"] = provenance

    if "inputs" in raw:
        inputs = raw["inputs"]
        if not isinstance(inputs, (Mapping, list, tuple)):
            raise ProtocolValidationError("inputs must be a mapping or list")
        normalized["inputs"] = copy.deepcopy(inputs)
    else:
        normalized["inputs"] = {}

    digest = _protocol_digest(normalized)
    normalized["protocol_sha256"] = digest
    normalized["protocol_id"] = f"sha256:{digest}"
    return normalized


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON or YAML protocol document."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Protocol file not found: {resolved}")

    try:
        if resolved.suffix.lower() == ".json":
            document = json.loads(resolved.read_text(encoding="utf-8"))
        else:
            document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ProtocolValidationError(f"Could not parse protocol {resolved}: {exc}") from exc

    return validate_protocol(document)


def _manifest_for_path(path: str | Path, *, hash_inputs: bool) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Evaluation input not found: {resolved}")

    if resolved.is_file():
        stat = resolved.stat()
        entry: dict[str, Any] = {
            "path": str(resolved),
            "kind": "file",
            "size_bytes": int(stat.st_size),
        }
        entry["sha256"] = _sha256_file(resolved) if hash_inputs else None
        return entry

    if not resolved.is_dir():
        raise ProtocolValidationError(f"Evaluation input is not a regular file/directory: {resolved}")

    files: list[dict[str, Any]] = []
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = child.relative_to(resolved).as_posix()
        stat = child.stat()
        item_manifest: dict[str, Any] = {
            "path": relative,
            "size_bytes": int(stat.st_size),
        }
        item_manifest["sha256"] = _sha256_file(child) if hash_inputs else None
        files.append(item_manifest)

    return {
        "path": str(resolved),
        "kind": "directory",
        "file_count": len(files),
        "files": files,
        "sha256": _sha256_bytes(_canonical_json(files).encode("utf-8")),
    }


def build_evaluation_protocol(
    protocol_path: str | Path,
    *,
    command: str,
    kind: str,
    inputs: Iterable[str | Path | None],
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated protocol plus run-specific provenance."""

    document = load_protocol(protocol_path)
    declared_kind = document["kind"]
    if declared_kind not in {"generic", kind}:
        raise ProtocolValidationError(
            f"Protocol kind {declared_kind!r} does not match evaluation kind {kind!r}"
        )
    hash_inputs = bool(document["provenance"].get("hash_inputs", True))
    input_manifest = [
        _manifest_for_path(path, hash_inputs=hash_inputs)
        for path in inputs
        if path is not None
    ]

    result = copy.deepcopy(document)
    result["run"] = {
        "command": command,
        "kind": kind,
        "cloudanalyzer_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "protocol_source": str(Path(protocol_path).expanduser().resolve()),
        "inputs": input_manifest,
        "options": copy.deepcopy(dict(options or {})),
    }
    return result


def attach_evaluation_protocol(
    result: dict[str, Any],
    protocol_path: str | Path | None,
    *,
    command: str,
    kind: str,
    inputs: Iterable[str | Path | None],
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a common protocol block when ``--protocol`` was supplied."""

    if protocol_path is None:
        return result
    result["evaluation_protocol"] = build_evaluation_protocol(
        protocol_path,
        command=command,
        kind=kind,
        inputs=inputs,
        options=options,
    )
    return result


__all__ = [
    "PROTOCOL_SCHEMA_VERSION",
    "ProtocolValidationError",
    "attach_evaluation_protocol",
    "build_evaluation_protocol",
    "load_protocol",
    "validate_protocol",
]
