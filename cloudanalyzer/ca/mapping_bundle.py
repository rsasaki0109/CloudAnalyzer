"""Portable review of an immutable delivered point/HD pair, without processing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "cloudanalyzer.mapping_review_bundle.v1"
DEFAULT_LIMIT = 1024**3
MANIFEST_LIMIT = 10 * 1024**2
MAX_FILES = 128


def _limit(value: int) -> None:
    if type(value) is not int or not 1024 <= value <= 4 * DEFAULT_LIMIT:
        raise ValueError("max_bundle_bytes must be an integer from 1024 to 4294967296")


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _rewrite(value: Any, paths: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {k: _rewrite(v, paths) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite(v, paths) for v in value]
    return paths.get(value, value) if isinstance(value, str) else value


def _name(value: str) -> bool:
    return (
        bool(value)
        and "\\" not in value
        and not value.startswith("/")
        and all(part not in {"", ".", ".."} for part in value.split("/"))
        and str(PurePosixPath(value)) == value
    )


def export_mapping_run(
    finished_job_dir: str,
    bundle_path: str,
    attribution: str,
    max_bundle_bytes: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Export a finished point/HD pair and final evidence to a NEW review ZIP.

    Copies the exact point map, graph, trajectory, Lanelet2, editable IR, projector,
    final four source audits, retained layout, source proposals and decision records.
    attribution must record the source-data license/credit supplied by the operator.
    The limit bounds total uncompressed bytes. No mapping attempt or run revision is
    spent. Original histories retain provenance paths; manifest output paths resolve
    inside the ZIP. Raw logs, native binaries and prior-run dependencies are excluded.
    This is a portable review/export, not a resumable mapping job or accuracy claim.
    inspect_mapping_bundle checks all member hashes without native processing.
    """
    from ca import mapping_job as jobs, mapping_run as runs, mapping_retry as retries

    _limit(max_bundle_bytes)
    if not isinstance(attribution, str) or not 1 <= len(attribution.strip()) <= 16384:
        raise ValueError(
            "supply source-data attribution/license text (1..16384 characters)"
        )
    root = Path(finished_job_dir).resolve()
    target = Path(bundle_path).absolute()
    if target.suffix.lower() != ".zip":
        raise ValueError("bundle_path must end in .zip")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"review bundle already exists: {target}")
    if not target.parent.is_dir():
        raise ValueError("bundle parent directory must already exist")
    with runs._locked(root):
        run = runs._load(root)
        if (
            run["status"] != "finished"
            or not run.get("output")
            or run["output"]["candidate_id"] is None
        ):
            raise ValueError(
                "export only a finished run with a delivered audited HD map"
            )
        runs.inspect_mapping_run(str(root))
        output = run["output"]
        owner = Path(output.get("candidate_job_dir", str(root))).resolve()
        job = jobs._load(owner)
        jobs._inputs(job)
        owner_run = runs._load(owner)
        if (
            owner_run["status"] != "finished"
            or owner_run["layout_file"]["sha256"] != run["layout_file"]["sha256"]
        ):
            raise ValueError(
                "delivered owner must be finished with the same frozen layout"
            )
        candidate = retries._parent(job, output["candidate_id"])
        for key in ("map", "graph", "trajectory"):
            if output["artifacts"].get(key) != job["pointcloud"]["files"][key]:
                raise ValueError("output is not the exact delivered point map")
        for key, artifact in candidate["files"].items():
            if output["artifacts"].get(f"hd_{key}") != artifact:
                raise ValueError("output is not the exact delivered HD map")
        if output["artifacts"].get("hd_source_audits") != candidate["quality_report"]:
            raise ValueError("output audit report does not match delivered HD map")
        diagnosis = jobs.diagnose_mapping_candidate(str(owner), candidate["id"])
        audits = [
            diagnosis["editable"],
            diagnosis["reopened_osm"],
            *(diagnosis["ground_consensus"] or {}).values(),
        ]
        if (
            len(audits) != 4
            or any(not a["complete"] or a["errors"] for a in audits)
            or any(i["severity"] == "error" for i in diagnosis["export_issues"])
        ):
            raise ValueError(
                "review export needs four complete audits without structural errors"
            )
        artifacts = {
            **output["artifacts"],
            "layout_hypothesis": run["layout_file"],
            "source_proposal": candidate["corridor_proposal"],
            "decision_history": jobs._artifact(root / "run.json"),
            "owner_decision_history": jobs._artifact(owner / "run.json"),
        }
        paths: dict[str, str] = {}
        files: list[dict[str, Any]] = []
        roles: dict[str, str] = {}
        for role, artifact in artifacts.items():
            jobs._verify(artifact)
            path = artifact["path"]
            if path not in paths:
                member = f"files/{len(files):03d}-{re.sub('[^a-zA-Z0-9_-]', '_', role)}{Path(path).suffix.lower()}"
                paths[path] = member
                files.append(
                    {
                        "path": member,
                        "sha256": artifact["sha256"],
                        "bytes": artifact["bytes"],
                        "original_path": path,
                    }
                )
            roles[role] = paths[path]
        manifest = {
            "schema": SCHEMA,
            "purpose": "portable_review",
            "attribution": attribution.strip(),
            "files": files,
            "roles": roles,
            "review": _rewrite(output, paths),
            "source": job["source"],
            "runtime": job["runtime"],
            "pointcloud_summary": {
                k: v
                for k, v in job["pointcloud"].items()
                if k not in {"files", "reports", "source_motion"}
            },
            "raw_logs_included": False,
            "resumable_mapping_job": False,
            "independent_accuracy_established": False,
            "deployment_ready": False,
            "integrity_meaning": "member hashes detect changed bytes; they are not an authenticity signature",
        }
        header = _json(manifest)
        if len(files) > MAX_FILES or len(header) > MANIFEST_LIMIT:
            raise ValueError("review manifest exceeds bounded file/metadata limits")
        total = sum(f["bytes"] for f in files) + len(header)
        if total > max_bundle_bytes:
            raise ValueError("review bundle exceeds max_bundle_bytes before copying")
        with tempfile.NamedTemporaryFile(
            prefix=".mapping-review-", suffix=".zip", dir=target.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("manifest.json", header)
                for file in files:
                    digest = hashlib.sha256()
                    size = 0
                    with Path(file["original_path"]).open("rb") as source, archive.open(
                        file["path"], "w", force_zip64=True
                    ) as destination:
                        for block in iter(lambda: source.read(1024**2), b""):
                            size += len(block)
                            if size > file["bytes"]:
                                raise ValueError(
                                    "recorded artifact changed while copying"
                                )
                            digest.update(block)
                            destination.write(block)
                    if size != file["bytes"] or digest.hexdigest() != file["sha256"]:
                        raise ValueError("recorded artifact changed while copying")
            result = inspect_mapping_bundle(str(temporary), max_bundle_bytes)
            # A competing export or existing target must never be overwritten.
            os.link(temporary, target)
        finally:
            temporary.unlink()
    result.update(bundle=jobs._artifact(target), bundle_path=str(target))
    return result


def inspect_mapping_bundle(
    bundle_path: str, max_bundle_bytes: int = DEFAULT_LIMIT
) -> dict[str, Any]:
    """Verify every review ZIP member and return its portable map/evidence paths.

    Needs only Python's standard library, not the native core, raw recording or
    original run directories. No extraction, processing or mapping attempts occur.
    Rejects changed bytes, unlisted/duplicate members, unsafe paths and excessive
    uncompressed sizes before streaming files. A hash match establishes integrity
    against this manifest, not authenticity, accuracy or road-use permission.
    Final audit/extent holds remain in review.diagnosis and full audit members.
    """
    _limit(max_bundle_bytes)
    with zipfile.ZipFile(bundle_path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if (
            len(infos) > MAX_FILES + 1
            or len(set(names)) != len(names)
            or any(not _name(n) for n in names)
        ):
            raise ValueError("review ZIP has duplicate, unsafe or excessive members")
        if any(
            i.is_dir()
            or i.flag_bits & 1
            or (i.external_attr >> 16) & 0o170000 not in {0, 0o100000}
            for i in infos
        ):
            raise ValueError("review ZIP must contain regular unencrypted files")
        if "manifest.json" not in names:
            raise ValueError("review ZIP has no manifest.json")
        if (
            archive.getinfo("manifest.json").file_size > MANIFEST_LIMIT
            or sum(i.file_size for i in infos) > max_bundle_bytes
        ):
            raise ValueError("review ZIP exceeds metadata or max_bundle_bytes limits")
        manifest = json.loads(archive.read("manifest.json"))
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported mapping review bundle schema")
        files = manifest.get("files")
        roles = manifest.get("roles")
        if not isinstance(files, list) or not isinstance(roles, dict):
            raise ValueError("review manifest needs files and roles")
        members: list[str] = []
        for file in files:
            if (
                not isinstance(file, dict)
                or not isinstance(file.get("path"), str)
                or not _name(file["path"])
                or not file["path"].startswith("files/")
            ):
                raise ValueError("unsafe review artifact path")
            if (
                type(file.get("bytes")) is not int
                or file["bytes"] < 0
                or not isinstance(file.get("sha256"), str)
                or not re.fullmatch("[0-9a-f]{64}", file["sha256"])
            ):
                raise ValueError("invalid review artifact descriptor")
            members.append(file["path"])
        if len(set(members)) != len(members) or set(names) != {
            "manifest.json",
            *members,
        }:
            raise ValueError("review members differ from manifest")
        required = {
            "map",
            "graph",
            "trajectory",
            "hd_map",
            "hd_editable_map",
            "hd_projector",
            "hd_source_audits",
            "layout_hypothesis",
            "source_proposal",
            "decision_history",
        }
        if not required <= roles.keys() or any(
            not isinstance(v, str) or v not in members for v in roles.values()
        ):
            raise ValueError("review bundle has missing map/evidence roles")
        for file in files:
            if archive.getinfo(file["path"]).file_size != file["bytes"]:
                raise ValueError("review artifact size differs from manifest")
            digest = hashlib.sha256()
            with archive.open(file["path"]) as source:
                for block in iter(lambda: source.read(1024**2), b""):
                    digest.update(block)
            if digest.hexdigest() != file["sha256"]:
                raise ValueError("review artifact hash differs from manifest")
        review = manifest.get("review")
        if not isinstance(review, dict) or not isinstance(
            review.get("artifacts"), dict
        ):
            raise ValueError("review manifest needs delivered output")
        if (
            not (
                required - {"layout_hypothesis", "source_proposal", "decision_history"}
            )
            <= review["artifacts"].keys()
        ):
            raise ValueError("delivered output is missing map/evidence roles")
        by_path = {file["path"]: file for file in files}
        for role, artifact in review["artifacts"].items():
            if (
                not isinstance(artifact, dict)
                or roles.get(role) != artifact.get("path")
                or artifact.get("path") not in by_path
            ):
                raise ValueError("delivered output differs from review roles")
            file = by_path[artifact["path"]]
            if any(artifact.get(k) != file[k] for k in ("sha256", "bytes")):
                raise ValueError("delivered output differs from member identity")
    return {
        "schema": SCHEMA,
        "bundle_path": str(Path(bundle_path).absolute()),
        "integrity_verified": True,
        "authenticity_verified": False,
        "verified_files": len(files),
        "uncompressed_bytes": sum(i.file_size for i in infos),
        "roles": roles,
        "attribution": manifest.get("attribution"),
        "pointcloud_summary": manifest.get("pointcloud_summary"),
        "review": review,
        "resumable_mapping_job": False,
        "independent_accuracy_established": False,
        "deployment_ready": False,
    }
