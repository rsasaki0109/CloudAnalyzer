"""Audit frozen point removal against explicitly paired SemanticKITTI labels.

This is a binary removal audit, not the official 19-class benchmark. No inference,
label propagation, threshold tuning or point-file writes occur here. Raw label IDs,
ignored classes and the validation split follow PRBonn/semantic-kitti-api config
at a9c749e8124b2243b6eef1b8bcf971a9f1173a2d (see README for attribution).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct

MAX_POINTS = 150_000
CLASSES = ("unlabeled", "car", "bicycle", "motorcycle", "truck", "other-vehicle",
           "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
           "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign")
RAW_GROUPS = ((0, 1, 52, 99), (10, 252), (11,), (15,), (18, 258), (13, 16, 20, 256, 257, 259),
              (30, 254), (31, 253), (32, 255), (40, 60), (44,), (48,), (49,), (50,), (51,),
              (70,), (71,), (72,), (80,), (81,))
LEARNING_MAP = {raw: learn for learn, ids in enumerate(RAW_GROUPS) for raw in ids}
CHECKPOINT = "1757b8917e81b39058feb78dd17d690d74a878cab272a2c7b47bd8e5869ee811"
ONNX = "8d61577f0d9582adfa28baea4246f915c29d4c5d5e3b1d27adac83ffef4815f5"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def projected_owners(scan: bytes) -> tuple[bytearray, dict[str, int]]:
    """Independently reproduce the bounded nearest-only ownership contract."""
    check(bool(scan) and len(scan) % 16 == 0 and len(scan) <= MAX_POINTS * 16, "invalid single-scan XYZI records")
    owners = [-1] * (64 * 2048)
    ranges = [math.inf] * len(owners)
    report = dict(points=len(scan) // 16, projected=0, occluded=0, invalid=0, outside_view=0, outside_range=0)
    lower, upper = math.radians(-25), math.radians(3)
    eligible = 0
    for i, point in enumerate(struct.iter_unpack("<4f", scan)):
        if not all(map(math.isfinite, point)):
            report["invalid"] += 1
            continue
        x, y, z, intensity = point
        check(0 <= intensity <= 1, "remission must already be in 0..1")
        distance = math.hypot(math.hypot(x, y), z)
        if not 1e-6 <= distance <= 100:
            report["outside_range"] += 1
            continue
        pitch = math.asin(z / distance)
        if not lower <= pitch <= upper:
            report["outside_view"] += 1
            continue
        col = min(2047, math.floor(.5 * (-math.atan2(y, x) / math.pi + 1) * 2048))
        row = min(63, math.floor((upper - pitch) / (upper - lower) * 64))
        pixel = row * 2048 + col
        eligible += 1
        if distance < ranges[pixel]:
            owners[pixel], ranges[pixel] = i, distance
    classified = bytearray(report["points"])
    for owner in owners:
        if owner >= 0:
            classified[owner] = 1
            report["projected"] += 1
    report["occluded"] = eligible - report["projected"]
    return classified, report


def validate_pairing(pairing: dict, scan: bytes, labels: bytes) -> None:
    check(pairing.get("app") == "CloudAnalyzer SemanticKITTI pairing" and pairing.get("version") == 1, "unsupported annotation pairing")
    seq, frame = pairing.get("sequence"), pairing.get("frame")
    check(isinstance(seq, str) and re.fullmatch(r"0[0-9]|10", seq) is not None, "ground truth sequence must be 00..10")
    check(isinstance(frame, str) and re.fullmatch(r"[0-9]{6}", frame) is not None, "invalid frame ID")
    check(pairing.get("split") == ("validation" if seq == "08" else "train"), "incorrect official split")
    check(pairing.get("point_order") == "original-kitti-xyzi" and pairing.get("label_format") == "semantickitti-uint32-le", "unsupported label format or order")
    check(pairing.get("scan_sha256") == sha(scan), "pairing scan SHA256 differs")
    check(pairing.get("labels_sha256") == sha(labels), "pairing labels SHA256 differs; do not substitute same-length labels")
    for key, folder, ext in [("scan_member", "velodyne", "bin"), ("labels_member", "labels", "label")]:
        member = pairing.get(key)
        check(isinstance(member, str) and member.endswith(f"sequences/{seq}/{folder}/{frame}.{ext}"), "annotation sequence/frame pairing differs")


def confusion(tp: int, fp: int, fn: int, tn: int) -> dict:
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                iou=ratio(tp, tp + fp + fn), non_target_false_removal_rate=ratio(fp, fp + tn))


def evaluate(scan: bytes, labels: bytes, mask: bytes, retained: bytes, candidates: bytes,
             report: dict, pairing: dict) -> dict:
    """Read labels only as a reference for a previously frozen result."""
    check(isinstance(report, dict) and isinstance(pairing, dict), "report and pairing must be JSON objects")
    check(bool(scan) and len(scan) % 16 == 0 and len(scan) <= MAX_POINTS * 16, "invalid single-scan XYZI records")
    count = len(scan) // 16
    check(len(labels) == count * 4, "one uint32 annotation is required per original point")
    validate_pairing(pairing, scan, labels)
    check(report.get("model") == "SalsaNext SemanticKITTI20" and report.get("backend") == "wgpu" and
          report.get("experimental") is True and report.get("device_selection") == "discrete_gpu_0" and
          report.get("source_weight_sha256") == CHECKPOINT and report.get("onnx_sha256") == ONNX and
          report.get("source_modified") is False and report.get("class_names") == list(CLASSES), "unsupported model result")
    profile = report.get("sensor_profile", {})
    check(all(profile.get(k) == v for k, v in dict(height=64, width=2048, fov_up_deg=3, fov_down_deg=-25, max_range_m=100,
                                                remission="unchanged normalized0..1").items()), "unsupported sensor profile")
    check(report.get("input_sha256") == sha(scan), "result scan SHA256 differs")
    check(len(mask) == count and all(v in (0, 1) for v in mask), "invalid binary source-order mask")
    check(report.get("mask_bytes") == count, "mask length differs from report")
    selected = report.get("selection", {})
    targets, threshold = selected.get("target_classes"), selected.get("threshold")
    check(isinstance(targets, list) and bool(targets) and all(c in CLASSES[1:] for c in targets), "invalid target classes")
    check(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1, "invalid threshold")
    target_ids = {CLASSES.index(c) for c in targets}
    classified, projection = projected_owners(scan)
    check(report.get("projection") == projection, "source projection differs from report")
    class_counts = selected.get("class_counts")
    check(isinstance(class_counts, list) and len(class_counts) == len(CLASSES) and
          all(type(c) is int and 0 <= c <= MAX_POINTS for c in class_counts) and
          sum(class_counts) == projection["projected"], "prediction class counts differ")
    check(selected.get("removed") == sum(mask) and selected.get("retained") == count - sum(mask) and
          selected.get("unclassified_retained") == count - projection["projected"], "selection counts differ")
    check(all(not remove or classified[i] for i, remove in enumerate(mask)), "unclassified returns cannot be removed")
    kept_expected = b"".join(scan[i * 16:(i + 1) * 16] for i, remove in enumerate(mask) if not remove)
    removed_expected = b"".join(scan[i * 16:(i + 1) * 16] for i, remove in enumerate(mask) if remove)
    check(retained == kept_expected and candidates == removed_expected, "result records differ from exact source split")
    check(report.get("retained_sha256") == sha(retained) and report.get("candidate_sha256") == sha(candidates), "output SHA256 differs")
    raw_ids = [value[0] & 0xffff for value in struct.iter_unpack("<I", labels)]
    check(all(raw in LEARNING_MAP for raw in raw_ids), "unknown SemanticKITTI raw label ID")
    sums = dict(tp=0, fp=0, fn=0, tn=0)
    owner_sums = sums.copy()
    ignored_removed = ignored_retained = target_unclassified = 0
    false_classes: Counter = Counter()
    instances: dict[int, dict] = defaultdict(lambda: dict(points=0, removed=0, projected=0, raw_ids=set()))
    instance_zero_target = 0
    for i, ((packed,), raw, remove) in enumerate(zip(struct.iter_unpack("<I", labels), raw_ids, mask)):
        learned = LEARNING_MAP[raw]
        if learned == 0:
            ignored_removed += bool(remove)
            ignored_retained += not remove
            continue
        positive = learned in target_ids
        key = ("tp" if remove else "fn") if positive else ("fp" if remove else "tn")
        sums[key] += 1
        if classified[i]:
            owner_sums[key] += 1
        elif positive:
            target_unclassified += 1
        if key == "fp":
            false_classes[raw] += 1
        if positive:
            instance_id = packed >> 16
            if instance_id == 0:
                instance_zero_target += 1
                continue
            instance = instances[instance_id]
            instance["points"] += 1
            instance["removed"] += remove
            instance["projected"] += classified[i]
            instance["raw_ids"].add(raw)
    targets_total = sums["tp"] + sums["fn"]
    return {
        "app": "CloudAnalyzer LiDAR removal audit", "version": 1,
        "evaluation_scope": "binary selected-class point removal on one annotated scan; not official semantic-segmentation mIoU",
        "dataset": "SemanticKITTI", "sequence": pairing["sequence"], "frame": pairing["frame"], "split": pairing["split"],
        "target_classes": targets, "score_threshold": threshold,
        "points": count, "removed": sum(mask), "retained": count - sum(mask),
        "evaluated_labeled_points": sum(sums.values()), "ignored_removed": ignored_removed, "ignored_retained": ignored_retained,
        "full_scan": confusion(**sums), "projected_owners_only": confusion(**owner_sums),
        "precision_if_all_ignored_removals_are_errors": ratio(sums["tp"], sum(mask)),
        "target_unclassified_retained": target_unclassified,
        "full_scan_recall_ceiling_from_nearest_only_ownership": ratio(targets_total - target_unclassified, targets_total),
        "false_removals_by_raw_class": {str(k): v for k, v in sorted(false_classes.items())},
        "target_instance_zero_points": instance_zero_target,
        "target_instances": [dict(instance_id=k, points=v["points"], removed=v["removed"], retained=v["points"] - v["removed"],
                                  projected=v["projected"], removed_fraction=ratio(v["removed"], v["points"]), raw_ids=sorted(v["raw_ids"]))
                             for k, v in sorted(instances.items())],
        "hashes": dict(scan_sha256=sha(scan), labels_sha256=sha(labels), mask_sha256=sha(mask), retained_sha256=sha(retained),
                       candidates_sha256=sha(candidates), pairing_sha256=sha((json.dumps(pairing, sort_keys=True) + "\n").encode())),
        "annotation_pairing": pairing,
        "limits": ["References are evaluated after frozen inference and never passed to the model",
                   "Ignored raw IDs 0,1,52,99 are excluded from confusion; removed ignored points are reported separately",
                   "Full-scan recall counts retained occluded/out-of-view target points as false negatives",
                   "Softmax threshold is not calibrated correctness probability",
                   "A few validation frames do not establish held-out accuracy, complete-object removal or fused-map support"],
    }


def bounded_read(path: Path, limit: int) -> bytes:
    check(path.stat().st_size <= limit, f"file exceeds audit byte budget: {path.name}")
    with path.open("rb") as file:
        raw = file.read(limit + 1)
    check(len(raw) <= limit, "file grew beyond audit budget")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scan", "labels", "pairing", "result", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        check(not args.out.exists(), "output must be a new JSON file; no source/result files are replaced")
        # The frozen native result is loaded and validated together with explicit pairing.
        result = evaluate(bounded_read(args.scan, MAX_POINTS * 16), bounded_read(args.labels, MAX_POINTS * 4),
                          bounded_read(args.result / "candidate-mask.u8", MAX_POINTS),
                          bounded_read(args.result / "retained.bin", MAX_POINTS * 16),
                          bounded_read(args.result / "candidates.bin", MAX_POINTS * 16),
                          json.loads(bounded_read(args.result / "report.json", 65536)), json.loads(bounded_read(args.pairing, 65536)))
        with args.out.open("x", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("sequence", "frame", "score_threshold", "full_scan", "target_unclassified_retained")}, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Audit failed: {error}\n")


if __name__ == "__main__":
    main()
