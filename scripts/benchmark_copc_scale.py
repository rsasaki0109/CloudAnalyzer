#!/usr/bin/env python3
"""Measure real local COPC streams and resumable tiles in separate processes.

Install the current native core, laspy, lazrs and psutil. Inputs must be actual
COPC files; this script neither downloads data nor synthesizes larger headers.
Reports and raw artifacts are local outputs, not repository fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloudanalyzer"))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("source", type=Path, help="local complete COPC file")
    result.add_argument("--bounds", type=float, nargs=6, required=True, metavar="XYZ")
    result.add_argument("--output", type=Path, required=True, help="new report JSON; refuses overwrite")
    result.add_argument("--chunk-size", type=int, default=10_000)
    result.add_argument("--oracle-chunk-size", type=int, default=250_000)
    result.add_argument("--max-oracle-points", type=int, default=200_000)
    result.add_argument("--grid-size", type=float, default=100)
    result.add_argument("--halo", type=float, default=2)
    result.add_argument("--full-tiles", action="store_true", help="also persist the entire source (plan disk space)")
    result.add_argument("--http-source", help="optional HTTP(S) copy of the same source for a box check")
    result.add_argument("--worker", help=argparse.SUPPRESS)
    return result


def save(path: Path, value: object) -> None:
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output = args.output.resolve()
    return output, output.with_suffix(".artifacts")


def canonical(array) -> list[bytes]:
    # Used only for the explicitly capped small oracle, never the whole cloud.
    return sorted(row.tobytes() for row in array)


def worker(args: argparse.Namespace) -> dict:
    import cloudanalyzer_core as cc
    import laspy
    import numpy as np
    import psutil

    from ca import copc_tiles as jobs

    _, artifacts = paths(args)
    mode = args.worker
    box = np.asarray(args.bounds)
    expanded = box + [-args.halo, -args.halo, 0, args.halo, args.halo, 0]
    baseline = psutil.Process().memory_info().rss
    start = time.perf_counter()
    report = {"mode": mode, "baseline_rss_bytes": baseline}
    if mode == "oracle":
        retained, count, selected = [], 0, 0
        with laspy.open(args.source) as source:
            header = source.header
            # Sequential LAZ, independent of CloudAnalyzer's octree selection.
            for records in source.chunk_iterator(args.oracle_chunk_size):
                xyz = np.column_stack((records.x, records.y, records.z))
                mask = np.all((xyz >= expanded[:3]) & (xyz <= expanded[3:]), axis=1)
                selected += int(mask.sum())
                if selected > args.max_oracle_points:
                    raise ValueError("expanded oracle box exceeds --max-oracle-points; choose a smaller box")
                if mask.any():
                    retained.append(records.array[mask].copy())
                count += len(records)
            if count != header.point_count:
                raise ValueError("sequential decoded count disagrees with the LAS header")
        array = np.concatenate(retained) if retained else np.empty(0, dtype=header.point_format.dtype())
        np.save(artifacts / "oracle.npy", array, allow_pickle=False)
        report.update(total_points=count, expanded_box_points=selected, record_bytes=header.point_format.size,
                      scales=header.scales.tolist(), offsets=header.offsets.tolist(),
                      min_xyz=header.mins.tolist(), max_xyz=header.maxs.tolist(),
                      canonical_expanded_raw_sha256=hashlib.sha256(b"".join(canonical(array))).hexdigest())
    elif mode in {"stream-full", "stream-box", "stream-http-box"}:
        count = nodes = batches = 0
        decoded = 0
        retained = []
        source = args.http_source if mode == "stream-http-box" else args.source
        with cc.CopcStream(source, bounds=None if mode == "stream-full" else args.bounds,
                           chunk_size=args.chunk_size) as stream:
            for node in stream.nodes():
                nodes += 1
                decoded += node.count
                for batch in stream.read_batches(node):
                    if len(batch.positions) > args.chunk_size:
                        raise ValueError("stream exceeded its batch size")
                    count += len(batch.positions)
                    batches += 1
                    if mode != "stream-full":
                        if count > args.max_oracle_points:
                            raise ValueError("stream box exceeds the oracle cap")
                        np.testing.assert_array_equal(batch.positions, np.column_stack(
                            (batch.records.x, batch.records.y, batch.records.z)))
                        retained.append(batch.records.array.copy())
            if mode == "stream-full" and count != stream.total_points:
                raise ValueError("full-density stream count disagrees with the header")
            report.update(total_points=stream.total_points, selected_points=count, nodes=nodes,
                          decoded_points=decoded, batches=batches, source_read_bytes=stream.ranges.bytes_read,
                          requests=stream.ranges.requests, limits=vars(stream.limits))
            if mode != "stream-full":
                array = np.concatenate(retained) if retained else np.empty(0, dtype=stream.header.point_format.dtype())
                np.save(artifacts / f"{mode}.npy", array, allow_pickle=False)
                report["identity"] = stream.identity
    elif mode.startswith("tile-") and mode != "tile-verify":
        full = mode in {"tile-interrupt", "tile-full-resume", "tile-full-complete"}
        job = artifacts / ("tiles-full" if full else "tiles-box")
        if mode == "tile-interrupt":
            # Instrument only our benchmark worker. Parent kills this process
            # after pack publication, before its SQLite node commit.
            def hold(*_):
                (artifacts / "interrupt-ready").write_text("published, uncommitted", encoding="utf-8")
                time.sleep(30)
                raise RuntimeError("benchmark parent did not stop the interrupt worker")
            jobs._commit_node = hold
        result = jobs.tile_copc(str(args.source), str(job), args.grid_size, halo=args.halo,
                               bounds=None if full else args.bounds, chunk_size=args.chunk_size,
                               resume=mode in {"tile-box-resume", "tile-full-resume", "tile-full-complete"},
                               stop_after_nodes=2 if mode == "tile-box-pause" else None)
        if mode in {"tile-full-complete", "tile-full-resume", "tile-box-resume"}:
            if result["status"] != "complete":
                raise ValueError("resumed job did not complete")
        if mode == "tile-full-complete" and (result["committed_nodes_this_call"] or result["written_pack_bytes"]):
            raise ValueError("completed resume wrote or decoded new nodes")
        with sqlite3.connect(job / "manifest.sqlite3") as db:
            report.update(pack_bytes=db.execute("SELECT COALESCE(SUM(pack_bytes),0) FROM nodes").fetchone()[0],
                          fragments=db.execute("SELECT COUNT(*) FROM fragments").fetchone()[0])
        report.update(result)
    elif mode == "tile-verify":
        oracle = np.load(artifacts / "oracle.npy", allow_pickle=False)
        with laspy.open(args.source) as source:
            header = source.header
        records = laspy.ScaleAwarePointRecord(oracle, header.point_format, header.scales, header.offsets)
        xyz = np.column_stack((records.x, records.y, records.z))
        inside = np.all((xyz >= box[:3]) & (xyz <= box[3:]), axis=1)
        actual_box = np.load(artifacts / "stream-box.npy", allow_pickle=False)
        if canonical(actual_box) != canonical(oracle[inside]):
            raise ValueError("stream box raw-record multiset differs from the sequential oracle")
        if args.http_source and canonical(np.load(artifacts / "stream-http-box.npy", allow_pickle=False)) != canonical(oracle[inside]):
            raise ValueError("HTTP box raw-record multiset differs from the sequential oracle")
        job = artifacts / "tiles-box"
        with sqlite3.connect(job / "manifest.sqlite3") as db:
            keys = list(db.execute("SELECT i,j FROM tiles ORDER BY i,j"))
        owners = np.floor(xyz[:, :2] / args.grid_size).astype(np.int64)
        owned, halo_count, exported = [], 0, 0
        for i, j in keys:
            key = np.asarray([i, j])
            mask = np.all((xyz[:, :2] >= key * args.grid_size - args.halo) &
                          (xyz[:, :2] <= (key + 1) * args.grid_size + args.halo), axis=1)
            is_halo = ~(inside & np.all(owners == key, axis=1))
            expected = sorted((row.tobytes(), int(h)) for row, h in zip(oracle[mask], is_halo[mask], strict=True))
            actual, identities = [], set()
            for batch in jobs.iter_tile_batches(str(job), i, j):
                original_schema = laspy.ScaleAwarePointRecord(batch["records"].array, header.point_format,
                                                             header.scales, header.offsets)
                np.testing.assert_array_equal(batch["positions"], np.column_stack(
                    (original_schema.x, original_schema.y, original_schema.z)))
                for row, ordinal, h in zip(batch["records"].array, batch["ordinals"], batch["halo"], strict=True):
                    identity = (batch["node_offset"], int(ordinal))
                    if identity in identities:
                        raise ValueError("duplicate source identity in a tile")
                    identities.add(identity)
                    actual.append((row.tobytes(), int(h)))
                    halo_count += int(h)
                    if not h:
                        owned.append(row.tobytes())
            if sorted(actual) != expected:
                raise ValueError(f"tile {(i, j)} raw records/halo differ from the sequential oracle")
            output = artifacts / f"tile-{i}-{j}-core.las"
            jobs.export_copc_tile(str(job), i, j, str(output))
            saved = laspy.read(output)
            if canonical(saved.points.array) != sorted(raw for raw, h in expected if not h):
                raise ValueError("core LAS export differs from the sequential oracle")
            np.testing.assert_array_equal(saved.header.scales, header.scales)
            np.testing.assert_array_equal(saved.header.offsets, header.offsets)
            vlrs = lambda h: [(v.user_id, v.record_id, v.record_data_bytes()) for v in h.vlrs
                              if v.user_id not in {"copc", "laszip encoded"}]
            if vlrs(saved.header) != vlrs(header):
                raise ValueError("core LAS export changed source VLR metadata")
            exported += len(saved.points)
        if sorted(owned) != canonical(oracle[inside]):
            raise ValueError("tile core ownership differs from the sequential oracle")
        report.update(oracle_core_points=int(inside.sum()), oracle_tiles=len(keys), halo_copies=halo_count,
                      raw_record_multiset_equal=True, unique_tile_source_identities=True,
                      xyz64_equal=True, core_export_points=exported, export_scales_offsets_vlrs_equal=True)
    else:
        raise ValueError(f"unknown worker: {mode}")
    report["seconds"] = time.perf_counter() - start
    return report


def measured_worker(args: argparse.Namespace, mode: str) -> dict:
    import psutil

    _, artifacts = paths(args)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", mode]
    # Files avoid pipe-buffer deadlocks and retain diagnostics after failures.
    with (artifacts / f"{mode}.stdout").open("wb") as out, (artifacts / f"{mode}.stderr").open("wb") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        process = psutil.Process(child.pid)
        peak = 0
        start = time.perf_counter()
        progress_at = start
        interrupted = False
        try:
            while child.poll() is None:
                try:
                    memory = process.memory_info()
                    peak = max(peak, memory.rss, getattr(memory, "peak_wset", 0))
                except psutil.NoSuchProcess:
                    pass
                if mode == "tile-interrupt":
                    if (artifacts / "interrupt-ready").exists():
                        child.terminate()
                        child.wait(timeout=10)
                        interrupted = True
                    elif time.perf_counter() - start > 120:
                        raise TimeoutError("interrupt worker did not publish a pack within 120 s")
                if time.perf_counter() - progress_at >= 30:
                    print(f"{mode}: {time.perf_counter()-start:.0f}s, observed peak RSS {peak / (1 << 20):.1f} MiB", flush=True)
                    progress_at = time.perf_counter()
                time.sleep(.002)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
    if mode == "tile-interrupt":
        if not interrupted:
            raise RuntimeError("interrupt worker exited before reaching the publication marker")
        with sqlite3.connect(artifacts / "tiles-full/manifest.sqlite3") as db:
            if db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] != 0:
                raise ValueError("interrupt test accidentally committed its first node")
        if not any((artifacts / "tiles-full/nodes").glob("*.pack")):
            raise ValueError("interrupt test did not leave a published uncommitted pack")
        result = {"mode": mode, "published_uncommitted_pack": True, "exit_code": child.returncode}
    else:
        if child.returncode:
            raise RuntimeError(f"{mode} failed; see {artifacts / (mode + '.stderr')}")
        result = json.loads((artifacts / f"{mode}.stdout").read_text(encoding="utf-8"))
    result.update(peak_rss_bytes=peak, process_seconds=time.perf_counter() - start)
    return result


def main() -> int:
    args = parser().parse_args()
    if (not all(math.isfinite(v) for v in args.bounds) or
            any(low > high for low, high in zip(args.bounds[:3], args.bounds[3:], strict=True))):
        raise ValueError("bounds must be finite min XYZ <= max XYZ")
    if not 1 <= args.chunk_size <= 1_000_000 or not 1 <= args.oracle_chunk_size <= 1_000_000:
        raise ValueError("chunk sizes must be in 1..1000000")
    if not 1 <= args.max_oracle_points <= 1_000_000:
        raise ValueError("oracle cap must be in 1..1000000")
    if not math.isfinite(args.grid_size) or args.grid_size <= 0 or not math.isfinite(args.halo) or args.halo < 0:
        raise ValueError("grid size must be finite/positive and halo finite/nonnegative")
    if args.http_source and urlsplit(args.http_source).scheme not in {"http", "https"}:
        raise ValueError("--http-source must use HTTP(S)")
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return 0
    output, artifacts = paths(args)
    if output.exists() or artifacts.exists():
        raise FileExistsError("benchmark report/artifacts already exist; choose a new --output")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    def source_identity() -> tuple[int, ...]:
        stat = args.source.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    identity = source_identity()
    digest = hashlib.sha256()
    with args.source.open("rb") as source:
        while data := source.read(1 << 20):
            digest.update(data)
    report = {"source": str(args.source.resolve()), "source_file_bytes": args.source.stat().st_size,
              "source_sha256": digest.hexdigest(), "python": sys.version, "platform": platform.platform(),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("cloudanalyzer-core", "numpy", "laspy", "lazrs", "psutil")},
              "bounds": args.bounds, "chunk_size": args.chunk_size, "oracle_chunk_size": args.oracle_chunk_size,
              "max_oracle_points": args.max_oracle_points, "grid_size": args.grid_size, "halo": args.halo,
              "full_tiles": args.full_tiles, "sampling_interval_seconds": .002, "runs": [], "status": "running"}
    if args.http_source:
        url = urlsplit(args.http_source)
        host = url.netloc.rsplit("@", 1)[-1]
        report["http_source"] = urlunsplit((url.scheme, host, url.path, "", ""))
    modes = ["oracle", "stream-box", "stream-full", "tile-box-pause", "tile-box-resume", "tile-verify"]
    if args.http_source:
        modes.insert(2, "stream-http-box")
    if args.full_tiles:
        modes += ["tile-interrupt", "tile-full-resume", "tile-full-complete"]
    save(output, report)
    try:
        for mode in modes:
            if source_identity() != identity:
                raise ValueError("benchmark source changed between measurements")
            print(f"Starting {mode}", flush=True)
            result = measured_worker(args, mode)
            if source_identity() != identity:
                raise ValueError("benchmark source changed during a measurement")
            report["runs"].append(result)
            save(output, report)
            print(json.dumps(result, allow_nan=False), flush=True)
        report["status"] = "complete"
    except BaseException:
        report["status"] = "failed"
        save(output, report)
        raise
    save(output, report)
    print(f"Report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
