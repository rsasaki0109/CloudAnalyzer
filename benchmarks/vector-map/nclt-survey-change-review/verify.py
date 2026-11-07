"""Independent PLY checks and changed-core support for box/lane associations.

Requires NumPy. This does not supply physical-change or surveyed-map ground truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def ply_vertices(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    marker = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:marker].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("Expected a binary little-endian PLY")
    types = {"char":"i1","uchar":"u1","short":"<i2","ushort":"<u2","int":"<i4","uint":"<u4","float":"<f4","double":"<f8",
             "int8":"i1","uint8":"u1","int16":"<i2","uint16":"<u2","int32":"<i4","uint32":"<u4","float32":"<f4","float64":"<f8"}
    properties = []
    count = None
    vertex = False
    for line in lines:
        fields = line.split()
        if fields[:1] == ["element"]:
            vertex = fields[1] == "vertex"
            if vertex:
                count = int(fields[2])
            elif int(fields[2]) != 0:
                raise ValueError("Only vertex-only PLY exports are supported")
        elif fields[:1] == ["property"] and vertex:
            if len(fields) != 3 or fields[1] not in types:
                raise ValueError("Unsupported vertex property")
            properties.append((fields[2], types[fields[1]]))
    dtype = np.dtype(properties)
    if count is None or len(raw) - marker != count * dtype.itemsize:
        raise ValueError("PLY vertex payload length mismatch")
    return np.frombuffer(raw, dtype=dtype, offset=marker, count=count)


def resample(curve: np.ndarray, count: int) -> np.ndarray:
    station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve[:, :2], axis=0), axis=1))]
    if np.any(np.diff(station) <= 0):
        raise ValueError("Independent check does not interpret duplicate/vertical XY stations")
    target = np.linspace(0, station[-1], count)
    return np.column_stack([np.interp(target, station, curve[:, axis]) for axis in range(3)])


def lane_triangles(lane: dict, boundaries: dict) -> np.ndarray:
    def curve(ref):
        key = ref if isinstance(ref, int) else ref["id"]
        result = np.array(boundaries[key], dtype=float)
        return result[::-1] if isinstance(ref, dict) and ref.get("reversed", False) else result
    left, right = curve(lane["left"]), curve(lane["right"])
    count = max(len(left), len(right), 2)
    left, right = resample(left, count), resample(right, count)
    return np.array([triangle for i in range(1, count) for triangle in
                     [(left[i-1], right[i-1], left[i]), (left[i], right[i-1], right[i])]])


def core_support(points: np.ndarray, triangles: np.ndarray, height_margin: float) -> np.ndarray:
    """XY barycentric inclusion plus interpolated surface Z, not triangle/box overlap."""
    supported = np.zeros(len(points), dtype=bool)
    for a, b, c in triangles:
        denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1e-12:
            continue
        u = ((b[1]-c[1])*(points[:,0]-c[0]) + (c[0]-b[0])*(points[:,1]-c[1])) / denominator
        v = ((c[1]-a[1])*(points[:,0]-c[0]) + (a[0]-c[0])*(points[:,1]-c[1])) / denominator
        w = 1-u-v
        surface_z = u*a[2] + v*b[2] + w*c[2]
        supported |= (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9) & (np.abs(points[:,2]-surface_z) <= height_margin+1e-9)
    return supported


def verify(artifacts: Path, samples: Path) -> dict:
    report = json.loads((artifacts / "verification.json").read_text())
    for source in report["sources"]:
        raw = (samples / source["name"]).read_bytes()
        assert len(raw) == source["bytes"] and hashlib.sha256(raw).hexdigest() == source["sha256"]
    raw = (artifacts / "m3c2-result.ply").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == report["resultPly"]["sha256"]
    cloud = ply_vertices(artifacts / "m3c2-result.ply")
    review = json.loads((artifacts / "survey-change-review.initial.json").read_text())
    assert len(cloud) == review["diagnostics"]["corePoints"]
    assert int(np.isfinite(cloud["m3c2_distance"]).sum()) == review["diagnostics"]["measured"]
    map_ = json.loads((artifacts / "reopened-map.json").read_text())
    boundaries = {b["id"]: b["geometry"] for b in map_["boundaries"]}
    surfaces = {lane["id"]: lane_triangles(lane, boundaries) for lane in map_["lanes"]}
    associations = json.loads((artifacts / "candidate-lane-overlaps.json").read_text())
    coordinates = np.column_stack([cloud[name].astype(float) for name in ("x", "y", "z")])
    tolerance = max(np.finfo(cloud[name].dtype).eps for name in ("x", "y", "z")) * max(1, float(np.abs(coordinates).max())) * 8
    pairs = []
    for candidate, association in zip(review["candidates"], associations, strict=True):
        assert candidate["rank"] == association["rank"]
        mask = cloud["change_object"] == candidate["rank"]
        points = coordinates[mask]
        assert len(points) == candidate["points"]
        np.testing.assert_allclose(np.r_[points.min(axis=0), points.max(axis=0)], candidate["bounds"], rtol=0, atol=tolerance)
        np.testing.assert_allclose(points.mean(axis=0), candidate["centroid"], rtol=0, atol=tolerance)
        np.testing.assert_allclose(cloud["m3c2_distance"][mask].astype(float).mean(), candidate["meanChange"], rtol=0, atol=1e-7)
        assert association["xyMargin"] == 0 and association["zMargin"] == 1
        for lane in association["ids"]:
            supported = int(core_support(points, surfaces[lane], 1).sum())
            pairs.append({"candidate":candidate["rank"],"lane":lane,"candidateCorePoints":len(points),
                          "coresInsideLaneAtHeightMargin1m":supported,"fraction":supported/len(points)})
    chosen = report["selection"]
    return {"scope":"Independent individual-core projection check on generated lane strips. This is a geometric proxy, not physical-change truth, lane accuracy, or a false-positive rate.",
            "checkedCorePoints":len(cloud),"verifiedCandidateObjects":len(review["candidates"]),"coordinateToleranceM":tolerance,
            "boxLanePairs":len(pairs),"pairsWithAtLeastOneCore":sum(p["coresInsideLaneAtHeightMargin1m"]>0 for p in pairs),
            "pairsWithNoCore":sum(p["coresInsideLaneAtHeightMargin1m"]==0 for p in pairs),
            "selectedPair":next(p for p in pairs if p["candidate"]==chosen["rank"] and p["lane"]==chosen["laneId"]),"pairs":pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--samples", type=Path, default=Path("web/public/samples"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.artifacts, args.samples)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
