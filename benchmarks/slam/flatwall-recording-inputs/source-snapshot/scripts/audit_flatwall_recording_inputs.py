#!/usr/bin/env python3
"""Inspect real Flatwall inputs without estimation or reference-archive access."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_GAP_NS = 50_000_000
TOPICS = {"/livox/imu": "sensor_msgs/msg/Imu", "/livox/lidar": "livox_ros_driver/msg/CustomMsg",
          "/livox/points": "sensor_msgs/msg/PointCloud2"}
FIELDS = {"x": (7, "f4"), "y": (7, "f4"), "z": (7, "f4"), "t": (6, "u4"),
          "intensity": (7, "f4"), "tag": (2, "u1"), "line": (2, "u1")}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def stamp_ns(stamp):
    sec, ns = int(stamp.sec), int(stamp.nanosec)
    if sec < 0 or not 0 <= ns < 10 ** 9 or sec * 10 ** 9 + ns > np.iinfo(np.int64).max:
        raise ValueError("invalid ROS timestamp")
    return sec * 10 ** 9 + ns


def times(values):
    values = np.asarray(values)
    if (values.dtype != np.dtype("int64") or values.ndim != 1 or not len(values)
            or values[0] < 0 or np.any(values[1:] <= values[:-1])):
        raise ValueError("timestamps must be nonnegative, strictly increasing int64")
    return values


def coverage(imu_times, start, end):
    """Inclusive endpoint coverage, with every intersected IMU interval checked."""
    imu_times = times(imu_times)
    if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)) or start > end:
        raise ValueError("invalid point interval")
    if start < imu_times[0] or end > imu_times[-1]:
        return dict(status="outside_imu_range", left_index=None, right_index=None, maximum_gap_ns=None)
    left = int(np.searchsorted(imu_times, start, side="right")) - 1
    right = int(np.searchsorted(imu_times, end, side="left"))
    gaps = np.diff(imu_times[left:right + 1])
    maximum = int(gaps.max()) if len(gaps) else 0
    return dict(status="imu_gap" if maximum > MAX_GAP_NS else "covered",
                left_index=left, right_index=right, maximum_gap_ns=maximum)


def point_columns(message):
    """Decode this recording's typed fields, including row padding and byte order."""
    fields = {f.name: f for f in message.fields}
    if len(fields) != len(message.fields) or set(fields) != set(FIELDS):
        raise ValueError("unexpected or duplicate point fields")
    height, width, step, row = map(int, [message.height, message.width, message.point_step, message.row_step])
    if min(height, width, step) <= 0 or row < width * step or len(message.data) != row * height:
        raise ValueError("invalid point layout")
    result, used = {}, set()
    for name, (datatype, kind) in FIELDS.items():
        field = fields[name]
        dtype = np.dtype((">" if message.is_bigendian else "<") + kind)
        occupied = set(range(int(field.offset), int(field.offset) + dtype.itemsize))
        if (int(field.datatype) != datatype or int(field.count) != 1 or int(field.offset) < 0
                or int(field.offset) + dtype.itemsize > step or occupied & used):
            raise ValueError("invalid point field: " + name)
        used |= occupied
        result[name] = np.ndarray((height, width), dtype=dtype, buffer=message.data,
                                  offset=int(field.offset), strides=(row, step)).reshape(-1)
    return result


def cloud_arrays(columns):
    return dict(positions=np.column_stack([columns[n] for n in "xyz"]).astype("<f8"),
                offsets_ns=columns["t"].astype("<u4"), intensity=columns["intensity"].astype("<f4"),
                tag=columns["tag"].astype("u1"), line=columns["line"].astype("u1"))


def compare_custom(message, arrays):
    if int(message.point_num) != len(message.points) or len(message.points) != len(arrays["positions"]):
        raise ValueError("custom point count mismatch")
    expected = dict(positions=np.array([[p.x, p.y, p.z] for p in message.points], dtype="<f8"),
                    offsets_ns=np.array([p.offset_time for p in message.points], dtype="<u4"),
                    intensity=np.array([p.reflectivity for p in message.points], dtype="<f4"),
                    tag=np.array([p.tag for p in message.points], dtype="u1"),
                    line=np.array([p.line for p in message.points], dtype="u1"))
    for key, value in expected.items():
        if value.tobytes() != arrays[key].tobytes():
            raise ValueError("custom/PointCloud2 mismatch: " + key)


def cloud_digest(arrays):
    digest = hashlib.sha256()
    for key, array in sorted(arrays.items()):
        digest.update(json.dumps([key, array.dtype.str, array.shape]).encode() + b"\0" + array.tobytes())
    return digest.hexdigest()


def time_summary(values):
    values = times(values)
    delta = np.diff(values)
    return dict(count=len(values), first_ns=int(values[0]), last_ns=int(values[-1]),
                duration_ns=int(values[-1]) - int(values[0]),
                minimum_delta_ns=int(delta.min()) if len(delta) else None,
                median_delta_ns=float(np.median(delta)) if len(delta) else None,
                maximum_delta_ns=int(delta.max()) if len(delta) else None,
                gaps_over_50ms=int(np.count_nonzero(delta > MAX_GAP_NS)))


def summarize(rows, imu):
    if not rows or [r["scan_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("original scan identities must be complete")
    imu_times = times(imu["timestamps_ns"])
    for row in rows:
        header, base = row["header_ns"], row["custom_timebase_ns"]
        if (header - base != row["header_minus_timebase_ns"] or not 0 < row["finite_points"] <= row["raw_points"]
                or not 0 <= row["offset_min_ns"] <= row["offset_max_ns"] < 2 ** 32):
            raise ValueError("invalid scan timing/count metadata")
        for origin, start in [("header", header), ("timebase", base)]:
            expected = coverage(imu_times, start + row["offset_min_ns"], start + row["offset_max_ns"])
            if row[origin + "_coverage"] != expected:
                raise ValueError("scan IMU coverage changed")
    for key in ["header_ns", "custom_timebase_ns"]:
        times(np.array([r[key] for r in rows], dtype=np.int64))
    acceleration = np.linalg.norm(imu["raw_linear_acceleration"], axis=1)
    gyro = np.linalg.norm(imu["raw_angular_velocity"], axis=1)
    if any(a.shape != (len(imu_times), 3) or not np.isfinite(a).all()
           for a in [imu["raw_linear_acceleration"], imu["raw_angular_velocity"]]):
        raise ValueError("invalid original IMU vectors")
    delta = [r["header_minus_timebase_ns"] for r in rows]
    return dict(scans=len(rows), raw_points=sum(r["raw_points"] for r in rows),
        finite_points=sum(r["finite_points"] for r in rows), imu=time_summary(imu_times),
        scan_headers=time_summary(np.array([r["header_ns"] for r in rows], dtype=np.int64)),
        point_time_decreases_retained=sum(r["point_time_decreases"] for r in rows),
        header_minus_timebase_ns=dict(min=min(delta), max=max(delta), nonzero=sum(d != 0 for d in delta)),
        coverage_counts={o: dict(Counter(r[o + "_coverage"]["status"] for r in rows)) for o in ["header", "timebase"]},
        covered_scan_indices={o: [r["scan_index"] for r in rows if r[o + "_coverage"]["status"] == "covered"] for o in ["header", "timebase"]},
        raw_acceleration_norm=dict(min=float(acceleration.min()), median=float(np.median(acceleration)), max=float(acceleration.max())),
        raw_angular_velocity_norm=dict(min=float(gyro.min()), median=float(np.median(gyro)), max=float(gyro.max())))


def inspect_recording(bag, out, core):
    from rosbags.highlevel import AnyReader

    out.mkdir(exist_ok=False)
    native = core.BagReader(str(bag))
    raw = native.imu_raw("/livox/imu")
    with AnyReader([bag]) as reader:
        if Counter((c.topic, c.msgtype) for c in reader.connections) != Counter(TOPICS.items()):
            raise ValueError("expected exactly the three original recording connections")
        connections = []
        for c in reader.connections:
            connections.append(dict(id=c.id, topic=c.topic, msgtype=c.msgtype, messages=c.msgcount,
                message_definition_sha256=hashlib.sha256(c.msgdef.data.encode()).hexdigest(), digest=c.digest,
                callerid=c.ext.callerid, latching=c.ext.latching))
            with (out / (str(c.id) + "-message-definition.txt")).open("x", encoding="utf-8", newline="\n") as f:
                f.write(c.msgdef.data)
        def messages(topic):
            for _, recorded, payload in reader.messages(connections=[c for c in reader.connections if c.topic == topic]):
                yield int(recorded), reader.deserialize(payload, TOPICS[topic])
        imu_rows, frames, seqs = [], [], []
        for recorded, m in messages("/livox/imu"):
            imu_rows.append([stamp_ns(m.header.stamp), recorded, m.angular_velocity.x, m.angular_velocity.y,
                             m.angular_velocity.z, m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
            frames.append(m.header.frame_id)
            seqs.append(int(m.header.seq))
        imu = dict(timestamps_ns=np.array([r[0] for r in imu_rows], dtype=np.int64),
                   recorded_ns=np.array([r[1] for r in imu_rows], dtype=np.int64),
                   sequence=np.array(seqs, dtype=np.uint32),
                   raw_angular_velocity=np.array([r[2:5] for r in imu_rows], dtype="<f8"),
                   raw_linear_acceleration=np.array([r[5:8] for r in imu_rows], dtype="<f8"))
        for native_name, key in [("timestamps_ns", "timestamps_ns"), ("angular_velocity_rad_s", "raw_angular_velocity"),
                                 ("linear_acceleration_m_s2", "raw_linear_acceleration")]:
            np.testing.assert_array_equal(raw[native_name], imu[key])
        if raw["frame_ids"] != frames:
            raise ValueError("native IMU frame mismatch")
        rows, digest = [], hashlib.sha256()
        for index, ((recorded, m), (custom_recorded, custom), scan) in enumerate(zip(
                messages("/livox/points"), messages("/livox/lidar"),
                native.scans("/livox/points", raw_time_field="t"), strict=True)):
            arrays = cloud_arrays(point_columns(m))
            compare_custom(custom, arrays)
            header, base = stamp_ns(m.header.stamp), int(custom.timebase)
            if (stamp_ns(custom.header.stamp) != header or custom.header.frame_id != m.header.frame_id
                    or not 0 <= base <= np.iinfo(np.int64).max or scan.stamp_ns != header or scan.frame_id != m.header.frame_id):
                raise ValueError("point header/frame/timebase mismatch")
            finite = np.isfinite(arrays["positions"]).all(axis=1)
            offset = arrays["offsets_ns"][finite].astype(np.int64)
            if not len(offset) or max(header, base) + int(offset.max()) > np.iinfo(np.int64).max:
                raise ValueError("empty or overflowing finite point time range")
            np.testing.assert_array_equal(scan.positions(), arrays["positions"][finite])
            np.testing.assert_array_equal(scan.point_times_ns(unit="ns", origin="header"), header + offset)
            value = cloud_digest(arrays)
            digest.update(index.to_bytes(4, "little") + bytes.fromhex(value))
            rows.append(dict(scan_index=index, recorded_ns=recorded, custom_recorded_ns=custom_recorded,
                header_ns=header, custom_timebase_ns=base, header_minus_timebase_ns=header-base,
                frame_id=m.header.frame_id, sequence=int(m.header.seq), custom_sequence=int(custom.header.seq),
                raw_points=len(finite), finite_points=int(finite.sum()),
                offset_min_ns=int(offset.min()), offset_max_ns=int(offset.max()),
                point_time_decreases=int(np.count_nonzero(offset[1:] < offset[:-1])),
                fields=[[f.name, int(f.offset), int(f.datatype), int(f.count)] for f in m.fields],
                point_step=int(m.point_step), row_step=int(m.row_step), width=int(m.width), height=int(m.height),
                is_bigendian=bool(m.is_bigendian), points_and_offsets_sha256=value,
                header_coverage=coverage(imu["timestamps_ns"], header+int(offset.min()), header+int(offset.max())),
                timebase_coverage=coverage(imu["timestamps_ns"], base+int(offset.min()), base+int(offset.max()))))
        metadata = dict(connections=connections, recorded_start_ns=int(reader.start_time), recorded_end_exclusive_ns=int(reader.end_time),
            imu_frame_ids=sorted(set(frames)), imu_frame_counts=dict(Counter(frames)),
            tf_or_calibration_topics_present=False, recording_producer_build_identified=False,
            native_raw_imu_and_points_match_independent=True, all_custom_points_and_offsets_match_pointcloud2=True,
            complete_points_and_offsets_sha256=digest.hexdigest(), reference_archive_read_in_this_stage=False)
    summary = summarize(rows, imu)
    save(out / "scans.json", rows)
    with (out / "imu.npz").open("xb") as stream:
        np.savez_compressed(stream, **imu)
    save(out / "metadata.json", metadata)
    save(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["protocol", "freeze", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (protocol["schema"] != "cloudanalyzer.flatwall_input_inspection.v1"
            or freeze["protocol_sha256"] != sha(args.protocol)):
        raise ValueError("input inspection protocol changed")
    required = [Path(__file__).resolve(), *(ROOT / p for p in protocol["recordings"].values()),
                ROOT / protocol["native_package"] / "_core.pyd"]
    if any(not p.is_relative_to(ROOT) or p.relative_to(ROOT).as_posix() not in freeze["files_sha256"] for p in required):
        raise ValueError("controller, original recordings and isolated native binary must be frozen")
    for name in freeze["files_sha256"]:
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if args.out.resolve() == path.resolve() or args.out.resolve() in path.resolve().parents:
            raise ValueError("output must not contain a frozen input")
    def verify():
        for name, digest in freeze["files_sha256"].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha(path) != digest:
                raise ValueError("frozen input changed: " + name)
    verify()
    sys.path.insert(0, str((ROOT / protocol["native_package"]).parent))
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    if Path(native.__file__).resolve().parent != (ROOT / protocol["native_package"]).resolve():
        raise ValueError("unexpected native package")
    args.out.mkdir(exist_ok=False)
    try:
        results = {name: inspect_recording(ROOT / path, args.out / name, core) for name, path in protocol["recordings"].items()}
        verify()
        save(args.out / "receipt.json", dict(at_utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(args.protocol), controller_freeze_sha256=sha(args.freeze),
            results=results, reference_archive_read_in_this_stage=False,
            output_files_sha256={p.relative_to(args.out).as_posix(): sha(p) for p in sorted(args.out.rglob("*")) if p.is_file()}))
    except Exception as exc:
        save(args.out / "failure.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), error_type=type(exc).__name__, error=str(exc)))
        raise
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
