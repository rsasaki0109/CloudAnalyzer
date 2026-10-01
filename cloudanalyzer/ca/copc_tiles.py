"""Bounded COPC tile packs with disk journals and explicit core/halo ownership."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sqlite3
import struct
import sys
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from ca._rust import core

_MAGIC = b"CATILE01"
_PACK = struct.Struct("<8s16sQH")
_FRAME_BYTES = 8 << 20
_MAX_INDEX = (1 << 52) - 8


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _path(root: Path, name: str) -> Path:
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("job artifact path leaves its output directory")
    return result


def _digest(path: Path, check: Callable[[], None] | None = None) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        while True:
            if check is not None:
                check()
            data = file.read(1 << 20)
            if not data:
                break
            size += len(data)
            digest.update(data)
    return size, digest.hexdigest()


def _sync_parent(path: Path) -> None:
    # POSIX directory synchronization persists the published name before the
    # SQLite node commit. Windows uses file fsync + its filesystem journal.
    if sys.platform != "win32":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _database(path: Path, *, create: bool = False) -> sqlite3.Connection:
    if not create and not path.is_file():
        raise ValueError("output is not an initialized COPC tile job")
    db = sqlite3.connect(path)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA cache_size=-8192")
        db.execute("PRAGMA temp_store=FILE")
    except BaseException:
        db.close()
        raise
    return db


def _lock_job(root: Path) -> BinaryIO:
    """OS lock releases automatically on close/crash; never delete a PID lock."""
    handle = _path(root, ".job.lock").open("a+b", buffering=0)
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return cast(BinaryIO, handle)


def _meta(db: sqlite3.Connection, key: str) -> str:
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        raise ValueError(f"job metadata missing: {key}")
    return str(row[0])


def _status(db: sqlite3.Connection, status: str) -> None:
    db.execute("INSERT OR REPLACE INTO meta VALUES ('status', ?)", (status,))
    db.commit()


def _summary(db: sqlite3.Connection, root: Path) -> dict[str, Any]:
    nodes, source, owned, halo = db.execute(
        "SELECT COUNT(*),COALESCE(SUM(source_points),0),COALESCE(SUM(core_points),0),COALESCE(SUM(halo_points),0) FROM nodes"
    ).fetchone()
    tiles = db.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
    return {"output_dir": str(root), "manifest": str(root / "manifest.sqlite3"),
            "status": _meta(db, "status"), "nodes": int(nodes),
            "decoded_source_points": int(source), "core_points": int(owned),
            "halo_copies": int(halo), "tiles": int(tiles)}


def _pack_header(job_id: str, offset: int, record_length: int) -> bytes:
    return _PACK.pack(_MAGIC, uuid.UUID(job_id).bytes, offset, record_length)


def _remove_orphan(path: Path, expected_header: bytes) -> None:
    if not path.exists():
        return
    with path.open("rb") as file:
        if file.read(_PACK.size) != expected_header:
            raise ValueError("existing uncommitted file does not belong to this job/node")
    # Exact job-owned path, already resolved/checked by _path.
    path.unlink()


def _initialize(db: sqlite3.Connection, root: Path, stream: Any, options: dict[str, Any], identity: dict[str, Any]) -> None:
    db.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE nodes (offset INTEGER PRIMARY KEY,source_points INTEGER NOT NULL,
          core_points INTEGER NOT NULL,halo_points INTEGER NOT NULL,pack_bytes INTEGER NOT NULL,sha256 TEXT NOT NULL);
        CREATE TABLE fragments (node INTEGER NOT NULL,fragment INTEGER NOT NULL,i INTEGER NOT NULL,j INTEGER NOT NULL,
          offset INTEGER NOT NULL,size INTEGER NOT NULL,points INTEGER NOT NULL,core_points INTEGER NOT NULL,
          PRIMARY KEY(node,fragment));
        CREATE INDEX fragments_tile ON fragments(i,j,node,fragment);
        CREATE TABLE tiles (i INTEGER NOT NULL,j INTEGER NOT NULL,core_points INTEGER NOT NULL,
          points INTEGER NOT NULL,PRIMARY KEY(i,j));
    """)
    job_id = str(uuid.uuid4())
    for name, data in (("source-header.bin", stream.raw_header), ("source-metadata-evlrs.bin", stream.raw_metadata_evlrs)):
        with _path(root, name).open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
    metadata = {name: _digest(_path(root, name)) for name in ("source-header.bin", "source-metadata-evlrs.bin")}
    values = {"version": "1", "job_id": job_id, "options": _json(options), "identity": _json(identity),
              "metadata": _json(metadata), "status": "running"}
    db.executemany("INSERT INTO meta VALUES (?,?)", values.items())
    db.commit()


def _verify_metadata(db: sqlite3.Connection, root: Path) -> None:
    if _meta(db, "version") != "1":
        raise ValueError("unsupported tile job version")
    metadata = json.loads(_meta(db, "metadata"))
    if not isinstance(metadata, dict) or set(metadata) != {"source-header.bin", "source-metadata-evlrs.bin"}:
        raise ValueError("source metadata manifest is incomplete")
    for name, expected in metadata.items():
        if not isinstance(expected, list) or len(expected) != 2 or type(expected[0]) is not int or not 0 <= expected[0] <= 8 << 20:
            raise ValueError("invalid source metadata byte limit")
        path = _path(root, name)
        if path.stat().st_size != expected[0] or list(_digest(path)) != expected:
            raise ValueError("committed source metadata is corrupt")


def _frame_dtype(header: Any) -> np.dtype:
    return np.dtype([("record", header.point_format.dtype()), ("ordinal", "<u8"), ("halo", "u1")])


def _tile_frames(batch: Any, options: dict[str, Any], domain: np.ndarray, dtype: np.dtype) -> Iterator[tuple[int, int, np.ndarray]]:
    xyz = batch.positions
    grid = options["grid_size"]
    origin = np.asarray(options["origin"])
    cells = (xyz[:, :2] - origin) / grid
    if not np.isfinite(cells).all() or np.any(np.abs(cells) > _MAX_INDEX):
        raise ValueError("grid indices exceed exact supported coordinate range")
    owners = np.floor(cells).astype(np.int64)
    inside = np.all((xyz >= domain[:3]) & (xyz <= domain[3:]), axis=1)
    halo = options["halo"]
    radius = math.ceil(halo / grid) + 1 if halo else 0
    domain_cells = (domain[[0, 1, 3, 4]].reshape(2, 2) - origin) / grid
    if not np.isfinite(domain_cells).all():
        raise ValueError("grid/domain arithmetic overflow")
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            targets = owners + np.array([di, dj], dtype=np.int64)
            # Only target tiles containing part of the original inclusive ROI.
            eligible = np.all((targets + 1 > domain_cells[0]) & (targets <= domain_cells[1]), axis=1)
            membership = np.all((cells >= targets - halo / grid) & (cells <= targets + 1 + halo / grid), axis=1)
            mask = eligible & membership
            if not mask.any():
                continue
            ids = np.flatnonzero(mask)
            keys, inverse = np.unique(targets[mask], axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            order = np.argsort(inverse, kind="stable")
            cuts = np.flatnonzero(np.diff(inverse[order])) + 1
            starts = np.concatenate(([0], cuts))
            ends = np.concatenate((cuts, [len(order)]))
            for key, start, end in zip(keys, starts, ends, strict=True):
                indices = ids[order[start:end]]
                frame = np.empty(len(indices), dtype=dtype)
                frame["record"] = batch.records.array[indices]
                frame["ordinal"] = batch.ordinals[indices]
                frame["halo"] = (~inside[indices]) | bool(di or dj)
                yield int(key[0]), int(key[1]), frame


def _commit_node(db: sqlite3.Connection, node: Any, owned: int, halo: int, size: int, digest: str) -> None:
    db.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?)", (node.offset, node.count, owned, halo, size, digest))
    db.commit()


def _write_node(db: sqlite3.Connection, root: Path, stream: Any, node: Any, options: dict[str, Any], domain: np.ndarray) -> int:
    header = _pack_header(_meta(db, "job_id"), node.offset, stream.header.point_format.size)
    name = f"nodes/{node.offset:016x}.pack"
    target, staged = _path(root, name), _path(root, name + ".part")
    seed = _path(root, name + f".{uuid.uuid4().hex}.seed")
    digest = hashlib.sha256(header)
    owned = halo = fragment = 0
    db.execute("BEGIN IMMEDIATE")
    try:
        if db.execute("SELECT 1 FROM nodes WHERE offset=?", (node.offset,)).fetchone() is not None:
            raise ValueError("node was already committed by another writer")
        _remove_orphan(staged, header)
        _remove_orphan(target, header)
        # Publish a complete ownership header before the known .part name is
        # visible. A crash while seeding leaves only an unreferenced tiny file.
        with seed.open("xb") as file:
            file.write(header)
            file.flush()
            os.fsync(file.fileno())
        os.replace(seed, staged)
        with staged.open("ab") as file:
            for batch in stream.read_batches(node):
                for i, j, frame in _tile_frames(batch, options, domain, _frame_dtype(stream.header)):
                    stream._check()
                    data = frame.tobytes()
                    offset = file.tell()
                    if offset + len(data) > options["max_node_output_bytes"] or fragment >= options["max_fragments_per_node"]:
                        raise ValueError("node output/fragment limit exceeded; use a larger grid, smaller halo or retiled source")
                    core_count = int(np.count_nonzero(frame["halo"] == 0))
                    owned += core_count
                    halo += len(frame) - core_count
                    file.write(data)
                    digest.update(data)
                    db.execute("INSERT INTO fragments VALUES (?,?,?,?,?,?,?,?)",
                               (node.offset, fragment, i, j, offset, len(data), len(frame), core_count))
                    db.execute("INSERT INTO tiles VALUES (?,?,?,?) ON CONFLICT(i,j) DO UPDATE SET "
                               "core_points=core_points+excluded.core_points,points=points+excluded.points",
                               (i, j, core_count, len(frame)))
                    fragment += 1
            file.flush()
            os.fsync(file.fileno())
            size = file.tell()
        os.replace(staged, target)
        _sync_parent(target)
        _commit_node(db, node, owned, halo, size, digest.hexdigest())
        return size
    except BaseException:
        db.rollback()
        if seed.exists():
            seed.unlink()
        # A published but uncommitted pack is validated/recovered on resume.
        _remove_orphan(staged, header)
        raise


def tile_copc(
    source: str,
    out_dir: str,
    grid_size: float,
    *,
    halo: float = 0,
    bounds: list[float] | None = None,
    origin: list[float] | None = None,
    chunk_size: int = 10_000,
    resume: bool = False,
    stop_after_nodes: int | None = None,
    limits: dict[str, int] | None = None,
    max_node_output_bytes: int = 256 << 20,
    max_fragments_per_node: int = 65_536,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Stream COPC into durable XY tiles in a new directory; resume compatible jobs.

    Coordinates/grid/halo use source units. Core ownership is half-open and
    unique; halo copies retain identities and do not increase core totals.
    Per-item/frame/output limits fail explicitly. Use stop_after_nodes for
    resumable bounded calls; cancellation checkpoints only completed nodes.
    Output is an internal pack + SQLite manifest, with bounded tile readers.
    """
    rust = core()
    if rust is None or not hasattr(rust, "CopcStream"):
        raise ValueError("COPC tile jobs require an updated Rust core")
    if not math.isfinite(grid_size) or grid_size <= 0 or not math.isfinite(halo) or halo < 0:
        raise ValueError("grid_size must be finite/positive and halo finite/nonnegative")
    grid_size, halo = float(grid_size), float(halo)
    ratio = halo / grid_size
    if not math.isfinite(ratio) or (halo > 0 and (2 * (math.ceil(ratio) + 1) + 1) ** 2 > 64):
        raise ValueError("halo/grid fan-out exceeds 64 candidates per point")
    if type(chunk_size) is not int or not 1 <= chunk_size <= 1_000_000:
        raise ValueError("chunk_size must be an integer in 1..1000000")
    if stop_after_nodes is not None and (type(stop_after_nodes) is not int or stop_after_nodes < 1):
        raise ValueError("stop_after_nodes must be a positive integer")
    if type(max_node_output_bytes) is not int or not _PACK.size <= max_node_output_bytes <= 2 << 30:
        raise ValueError("max_node_output_bytes must be in header size..2GiB")
    if type(max_fragments_per_node) is not int or not 1 <= max_fragments_per_node <= 1_000_000:
        raise ValueError("max_fragments_per_node must be in 1..1000000")
    origin_array = np.asarray(origin if origin is not None else [0., 0.], dtype=np.float64)
    if origin_array.shape != (2,) or not np.isfinite(origin_array).all():
        raise ValueError("origin must be two finite coordinates")
    box = None
    if bounds is not None:
        domain = np.asarray(bounds, dtype=np.float64)
        if domain.shape != (6,) or not np.isfinite(domain).all() or np.any(domain[:3] > domain[3:]):
            raise ValueError("bounds must be finite min XYZ <= max XYZ")
        expanded = domain + np.array([-halo, -halo, 0, halo, halo, 0])
        if not np.isfinite(expanded).all():
            raise ValueError("halo-expanded bounds overflow")
        box = tuple(float(value) for value in expanded)
    try:
        query_limits = rust.CopcLimits(**(limits or {}))
    except TypeError as exc:
        raise ValueError("invalid COPC limits") from exc
    root = Path(out_dir).resolve()
    with rust.CopcStream(source, bounds=box, chunk_size=chunk_size, limits=query_limits, cancel=cancel) as stream:
        if not hasattr(stream, "raw_metadata_evlrs"):
            raise ValueError("COPC tile jobs require a core with raw metadata snapshots")
        if stream.file_size > (1 << 63) - 1:
            raise ValueError("source offsets exceed the SQLite signed 64bit range")
        fanout = (2 * (math.ceil(ratio) + 1) + 1) ** 2 if halo else 1
        if stream.total_points > ((1 << 63) - 1) // fanout:
            raise ValueError("source/copy totals exceed the SQLite signed 64bit range")
        identity = stream.identity
        if urlsplit(source).scheme in {"http", "https"}:
            if not identity.get("etag"):
                raise ValueError("resumable HTTP jobs require an exposed strong ETag")
            parsed = urlsplit(source)
            # Allow refreshed presigned credentials for the same pinned object.
            identity["source_sha256"] = hashlib.sha256(urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")).encode()).hexdigest()
        if bounds is None:
            domain = np.asarray(stream.bounds)
        options = {"grid_size": grid_size, "halo": halo, "bounds": domain.tolist() if bounds is not None else None, "origin": origin_array.tolist(),
                   "chunk_size": chunk_size, "limits": vars(query_limits), "frame_bytes": _FRAME_BYTES,
                   "max_node_output_bytes": max_node_output_bytes, "max_fragments_per_node": max_fragments_per_node}
        # Bound point+identity frames independently of unusually wide records.
        stream.chunk_size = min(chunk_size, _FRAME_BYTES // _frame_dtype(stream.header).itemsize)
        if not resume:
            root.mkdir(parents=True, exist_ok=False)
            _path(root, "nodes").mkdir()
        lock_handle = _lock_job(root)
        try:
            db = _database(_path(root, "manifest.sqlite3"), create=not resume)
        except BaseException:
            lock_handle.close()
            raise
        try:
            if resume:
                _verify_metadata(db, root)
                stored_options = json.loads(_meta(db, "options"))
                for quota in ("max_node_output_bytes", "max_fragments_per_node"):
                    if options[quota] < stored_options[quota]:
                        raise ValueError("resume cannot reduce output/fragment quotas")
                    stored_options[quota] = options[quota]
                if _meta(db, "identity") != _json(identity) or _json(stored_options) != _json(options):
                    raise ValueError("resume source identity/options do not match the original job")
                db.execute("UPDATE meta SET value=? WHERE key='options'", (_json(options),))
                db.commit()
            else:
                _initialize(db, root, stream, options, identity)
            _status(db, "running")
            committed_this_call = 0
            verified_pack_bytes = written_pack_bytes = 0
            try:
                for node in stream.nodes():
                    row = db.execute("SELECT source_points,pack_bytes,sha256 FROM nodes WHERE offset=?", (node.offset,)).fetchone()
                    if row is not None:
                        pack = _path(root, f"nodes/{node.offset:016x}.pack")
                        if (int(row[0]) != node.count or not _PACK.size <= int(row[1]) <= options["max_node_output_bytes"] or
                                pack.stat().st_size != int(row[1]) or _digest(pack, stream._check) != (int(row[1]), str(row[2]))):
                            raise ValueError("committed node artifact is missing/corrupt or disagrees with the source")
                        verified_pack_bytes += int(row[1])
                        continue
                    written_pack_bytes += _write_node(db, root, stream, node, options, domain)
                    committed_this_call += 1
                    if stop_after_nodes is not None and committed_this_call >= stop_after_nodes:
                        _status(db, "paused")
                        break
                else:
                    if bounds is None:
                        owned_total = db.execute("SELECT COALESCE(SUM(core_points),0) FROM nodes").fetchone()[0]
                        if owned_total != stream.total_points:
                            raise ValueError("full-density ownership total disagrees with source point count")
                    _status(db, "complete")
            except (KeyboardInterrupt, rust.CopcCancelled):
                db.rollback()
                _status(db, "cancelled")
            except BaseException:
                db.rollback()
                _status(db, "failed")
                raise
            result = _summary(db, root)
            result.update(source_read_bytes=stream.ranges.bytes_read, requests=stream.ranges.requests,
                          verified_pack_bytes=verified_pack_bytes, written_pack_bytes=written_pack_bytes,
                          committed_nodes_this_call=committed_this_call, total_source_points=stream.total_points)
            return result
        finally:
            db.close()
            lock_handle.close()


def iter_tile_batches(out_dir: str, i: int, j: int, *, core_only: bool = False) -> Iterator[dict[str, Any]]:
    """Read one tile's committed bounded frames; return records/XYZ/source identities."""
    import laspy

    root = Path(out_dir).resolve()
    db_path = _path(root, "manifest.sqlite3")
    if not db_path.is_file():
        raise ValueError("output is not a COPC tile job")
    db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        _verify_metadata(db, root)
        head = _path(root, "source-header.bin").read_bytes()
        header = laspy.LasHeader.read_from(io.BytesIO(head), read_evlrs=False)
        dtype = _frame_dtype(header)
        job_id = _meta(db, "job_id")
        options = json.loads(_meta(db, "options"))
        cursor = db.execute("SELECT node,offset,size,points FROM fragments WHERE i=? AND j=? ORDER BY node,fragment", (i, j))
        last_node = None
        for node, offset, size, count in cursor:
            path = _path(root, f"nodes/{node:016x}.pack")
            if last_node != node:
                row = db.execute("SELECT pack_bytes,sha256 FROM nodes WHERE offset=?", (node,)).fetchone()
                if (row is None or not _PACK.size <= int(row[0]) <= options["max_node_output_bytes"] or
                        path.stat().st_size != int(row[0]) or _digest(path) != (int(row[0]), str(row[1]))):
                    raise ValueError("committed node artifact is corrupt")
                last_node = node
            if count <= 0 or size != count * dtype.itemsize or size > options["frame_bytes"] or offset < _PACK.size:
                raise ValueError("invalid committed fragment span")
            with path.open("rb") as file:
                if file.read(_PACK.size) != _pack_header(job_id, node, header.point_format.size):
                    raise ValueError("pack identity does not match the job")
                file.seek(offset)
                data = file.read(size)
            if len(data) != size:
                raise ValueError("committed fragment is truncated")
            frame = np.frombuffer(data, dtype=dtype)
            if core_only:
                frame = frame[frame["halo"] == 0]
            if len(frame):
                records = laspy.ScaleAwarePointRecord(frame["record"].copy(), header.point_format, header.scales, header.offsets)
                yield {"records": records, "positions": np.column_stack((records.x, records.y, records.z)),
                       "node_offset": int(node), "ordinals": frame["ordinal"].copy(), "halo": frame["halo"].copy()}
    finally:
        db.close()


def _export_header(root: Path) -> Any:
    import laspy
    from laspy.vlrs.vlrlist import VLRList

    header = laspy.LasHeader.read_from(io.BytesIO(_path(root, "source-header.bin").read_bytes()), read_evlrs=False)
    header.vlrs = VLRList(v for v in header.vlrs if v.user_id not in {"copc", "laszip encoded"})
    data = _path(root, "source-metadata-evlrs.bin").read_bytes()
    offset = count = 0
    while offset < len(data):
        if offset + 60 > len(data):
            raise ValueError("source EVLR snapshot is truncated")
        size = int.from_bytes(data[offset + 20:offset + 28], "little")
        offset += 60 + size
        count += 1
        if offset > len(data) or count > 16_384:
            raise ValueError("invalid source EVLR snapshot")
    header.evlrs = VLRList.read_from(io.BytesIO(data), count, extended=True)
    header.are_points_compressed = False
    header.start_of_first_evlr = 0
    header.number_of_evlrs = 0
    return header


def export_copc_tile(out_dir: str, i: int, j: int, output: str, *, include_halo: bool = False) -> dict[str, Any]:
    """Stream a committed tile to new LAS/LAZ, preserving source schema/metadata.

    By default export only uniquely owned core records. include_halo adds
    explicit ca_halo/ca_source_node_offset/ca_source_ordinal extra dimensions.
    Output is atomically published without replacing an existing artifact.
    """
    import laspy

    root = Path(out_dir).resolve()
    db = _database(_path(root, "manifest.sqlite3"))
    try:
        _verify_metadata(db, root)
        row = db.execute("SELECT core_points,points FROM tiles WHERE i=? AND j=?", (i, j)).fetchone()
        if _meta(db, "status") != "complete":
            raise ValueError("tile job is not complete; resume before full-density export")
        if row is None:
            raise ValueError("requested tile is not in the committed manifest")
    finally:
        db.close()
    target = Path(output).resolve()
    if target.suffix.lower() not in {".las", ".laz"}:
        raise ValueError("output must have .las or .laz extension")
    if target.exists():
        raise FileExistsError(target)
    header = _export_header(root)
    if include_halo:
        for name, dtype in (("ca_halo", "u1"), ("ca_source_node_offset", "u8"), ("ca_source_ordinal", "u8")):
            if name in header.point_format.dimension_names:
                raise ValueError(f"source already has reserved export dimension {name}")
            header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.dtype(dtype)))
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    count = 0
    try:
        with staged.open("xb") as file:
            with laspy.open(file, mode="w", header=header, do_compress=target.suffix.lower() == ".laz", closefd=False) as writer:
                for batch in iter_tile_batches(out_dir, i, j, core_only=not include_halo):
                    records = batch["records"]
                    if include_halo:
                        extended = laspy.ScaleAwarePointRecord.zeros(len(records), header=header)
                        for name in records.array.dtype.names:
                            extended.array[name] = records.array[name]
                        extended.array["ca_halo"] = batch["halo"]
                        extended.array["ca_source_node_offset"] = batch["node_offset"]
                        extended.array["ca_source_ordinal"] = batch["ordinals"]
                        records = extended
                    writer.write_points(records)
                    count += len(records)
                writer.write_evlrs(header.evlrs)
            file.flush()
            os.fsync(file.fileno())
        expected = int(row[1] if include_halo else row[0])
        if count != expected:
            raise ValueError("tile changed during export; retry after the writer has stopped")
        # Same-directory hard link publishes atomically and refuses overwrite.
        os.link(staged, target)
        _sync_parent(target)
    finally:
        if staged.exists():
            staged.unlink()
    return {"output": str(target), "tile": [i, j], "points": count, "include_halo": include_halo}
