"""Portable review of an immutable delivered point/HD pair, without processing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "cloudanalyzer.mapping_review_bundle.v1"
PREVIEW_SCHEMA = "cloudanalyzer.mapping_review_bundle.v2"
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


def _check_preview(
    value: dict[str, Any],
    output: dict[str, Any],
    file: dict[str, Any],
    source_count: int,
    archive: zipfile.ZipFile,
) -> None:
    try:
        count, kept, stride, limit = (
            value[k]
            for k in (
                "source_count",
                "preview_count",
                "every_nth_record",
                "max_preview_points",
            )
        )
        if (
            any(type(n) is not int for n in (count, kept, stride, limit))
            or not 1 <= count <= 10**9
            or not 1 <= limit <= 10**6
            or count != source_count
            or limit >= count
            or stride != math.ceil(count / limit)
            or kept != math.ceil(count / stride)
        ):
            raise ValueError("invalid preview sampling counts")
        if value["source"] != output["artifacts"]["map"] or value["file"] != {
            k: file[k] for k in ("path", "sha256", "bytes")
        }:
            raise ValueError("preview differs from delivered source or packaged file")
        source = value["source"]
        if (
            not isinstance(source.get("path"), str)
            or not re.fullmatch("[0-9a-f]{64}", source.get("sha256", ""))
            or type(source.get("bytes")) is not int
            or source["bytes"] < file["bytes"]
        ):
            raise ValueError("invalid original full point-map identity")
        if (
            value["purpose"] != "display_only"
            or value["first_record"] != 0
            or value["source_for_saved_audits"] != "original_full_point_map"
            or value["full_point_map_included"] is not False
            or value["coordinate_frame_changed"] is not False
            or value["coordinate_or_attribute_quantization"] is not False
            or value["original_record_bytes_preserved"] is not True
        ):
            raise ValueError("invalid preview purpose/provenance flags")
        with archive.open(file["path"]) as stream:
            lines: list[bytes] = []
            size = 0
            while size < 16384:
                line = stream.readline(16384)
                lines.append(line)
                size += len(line)
                if line == b"end_header\n":
                    break
                if not line:
                    raise ValueError("incomplete preview PLY header")
            else:
                raise ValueError("preview PLY header exceeds its limit")
        if lines[:3] != [
            b"ply\n",
            b"format binary_little_endian 1.0\n",
            f"element vertex {kept}\n".encode(),
        ] or lines[3:6] != [
            b"property double x\n",
            b"property double y\n",
            b"property double z\n",
        ]:
            raise ValueError("preview PLY differs from its sampling metadata")
        fields = [
            re.fullmatch(rb"property float ([A-Za-z_][A-Za-z_0-9]*)\n", line)
            for line in lines[6:-1]
        ]
        if len(fields) > 16 or any(f is None for f in fields):
            raise ValueError("invalid preview PLY attribute fields")
        names = [f[1] for f in fields if f is not None]
        if len(set(names + [b"x", b"y", b"z"])) != len(names) + 3:
            raise ValueError("duplicate preview PLY attribute fields")
        width = 24 + len(fields) * 4
        if value["record_size_bytes"] != width or file["bytes"] != size + kept * width:
            raise ValueError("preview PLY record size differs from manifest")
    except (KeyError, TypeError, ZeroDivisionError) as error:
        raise ValueError("incomplete display preview provenance") from error


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
    return _export(finished_job_dir, bundle_path, attribution, max_bundle_bytes, 0)


def export_mapping_preview(
    finished_job_dir: str,
    bundle_path: str,
    attribution: str,
    max_preview_points: int = 200000,
    max_bundle_bytes: int = 64 * 1024**2,
) -> dict[str, Any]:
    """Package a display-only subset of a finished point map with its exact HD map.

    Streams every kth original canonical PLY record with all double coordinates and
    attributes byte-identical. The full point map stays unchanged outside this ZIP;
    its descriptor and all four saved audits remain bound to the original full map.
    This subset is for viewing, not for new source audits or mapping continuation.
    A new v2 manifest distinguishes preview_map from the original delivered map.
    No mapping attempts, odometry, fusion or HD generation are performed. Existing
    ZIPs are never overwritten. Point budget is 1..1000000; uncompressed bytes are
    bounded before packaging. Source-data attribution is required.
    """
    if type(max_preview_points) is not int or not 1 <= max_preview_points <= 10**6:
        raise ValueError("max_preview_points must be an integer from 1 to 1000000")
    return _export(
        finished_job_dir, bundle_path, attribution, max_bundle_bytes, max_preview_points
    )


def _export(
    finished_job_dir: str,
    bundle_path: str,
    attribution: str,
    max_bundle_bytes: int,
    preview_points: int,
) -> dict[str, Any]:
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
    with runs._locked(root), ExitStack() as cleanup:
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
        preview = None
        if preview_points and (
            type(job["pointcloud"]["map_points"]) is not int
            or job["pointcloud"]["map_points"] < 1
        ):
            raise ValueError("preview requires a positive original point count")
        if preview_points and job["pointcloud"]["map_points"] > preview_points:
            from ca import mapping_preview

            directory = Path(
                cleanup.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix=".mapping-preview-", dir=target.parent
                    )
                )
            )
            preview_path = directory / "display-preview.ply"
            preview = mapping_preview.write(
                artifacts.pop("map"),
                preview_path,
                job["pointcloud"]["map_points"],
                preview_points,
            )
            artifacts["preview_map"] = jobs._artifact(preview_path)
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
            "schema": PREVIEW_SCHEMA if preview else SCHEMA,
            "purpose": "portable_preview_review" if preview else "portable_review",
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
        if preview:
            preview["file"] = _rewrite(artifacts["preview_map"], paths)
            manifest["preview_pointcloud"] = preview
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
    V1 contains the exact full point map. V2 contains a display-only preview_map;
    preview_pointcloud identifies its external original map and full-source audits.
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
        if not isinstance(manifest, dict) or manifest.get("schema") not in {
            SCHEMA,
            PREVIEW_SCHEMA,
        }:
            raise ValueError("unsupported mapping review bundle schema")
        preview = (
            manifest.get("preview_pointcloud")
            if manifest["schema"] == PREVIEW_SCHEMA
            else None
        )
        if manifest["schema"] == PREVIEW_SCHEMA and not isinstance(preview, dict):
            raise ValueError("preview bundle needs display-only point provenance")
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
        required_roles = (
            (required - {"map"}) | {"preview_map"} if preview is not None else required
        )
        if not required_roles <= roles.keys() or any(
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
        if preview is not None:
            if "map" in roles:
                raise ValueError("preview bundle must distinguish its original map")
            summary = manifest.get("pointcloud_summary")
            if (
                not isinstance(summary, dict)
                or type(summary.get("map_points")) is not int
            ):
                raise ValueError("preview bundle needs original full point count")
            _check_preview(
                preview,
                review,
                by_path[roles["preview_map"]],
                summary["map_points"],
                archive,
            )
        for role, artifact in review["artifacts"].items():
            if preview is not None and role == "map":
                continue
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
        "schema": manifest["schema"],
        "bundle_path": str(Path(bundle_path).absolute()),
        "integrity_verified": True,
        "authenticity_verified": False,
        "verified_files": len(files),
        "uncompressed_bytes": sum(i.file_size for i in infos),
        "roles": roles,
        "attribution": manifest.get("attribution"),
        "pointcloud_summary": manifest.get("pointcloud_summary"),
        "preview_pointcloud": preview,
        "review": review,
        "resumable_mapping_job": False,
        "independent_accuracy_established": False,
        "deployment_ready": False,
    }
