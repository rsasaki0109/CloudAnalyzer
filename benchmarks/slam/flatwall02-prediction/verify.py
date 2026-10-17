#!/usr/bin/env python3
"""Independently verify exact reference pairing, eight trajectories and diagnostics."""
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ["raw-legacy", "gyro-legacy", "raw-0.1875", "gyro-0.1875"]
POLICIES = ["repeat-motion", "last-pose"]


def read(path): return json.loads(path.read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a, b, atol=2e-8):
    if not np.allclose(a, b, rtol=2e-12, atol=atol): raise ValueError("numeric accounting differs")
def inverse(p):
    q = np.eye(4)
    q[:3, :3], q[:3, 3] = p[:3, :3].T, -p[:3, :3].T@p[:3, 3]
    return q
def angles(r):
    v = np.stack([r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]], axis=-1)
    return np.degrees(np.arctan2(np.linalg.norm(v, axis=-1)/2, (np.trace(r, axis1=-2, axis2=-1)-1)/2))
def stats(x):
    x = np.asarray(x, dtype=float)
    if not len(x): return None
    if not np.isfinite(x).all(): raise ValueError("nonfinite errors")
    return dict(count=len(x), rmse=float(np.sqrt(np.mean(x*x))), mean=float(x.mean()), median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max()))
def check_stats(actual, values, atol=2e-8):
    expected = stats(values)
    if expected is None:
        if actual is not None: raise ValueError("empty statistic differs")
    else:
        if set(actual) != set(expected): raise ValueError("statistic fields differ")
        for k, v in expected.items(): close(actual[k], v, atol)
def poses(p):
    if p.ndim != 3 or p.shape[1:] != (4, 4) or not np.isfinite(p).all(): raise ValueError("invalid poses")
    close(p[:, 3], [0, 0, 0, 1], 1e-12)
    close(p[:, :3, :3]@p[:, :3, :3].transpose(0, 2, 1), np.eye(3), 1e-6)
    close(np.linalg.det(p[:, :3, :3]), 1., 1e-6)
def quaternion(q):
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def reference_pairs(path, estimate_times):
    stamps, rows = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        tokens = line.split()
        if len(tokens) != 8: raise ValueError("invalid selected reference columns")
        stamp = Fraction(tokens[0])*10**9
        if stamp.denominator != 1 or not -(2**63) <= stamp < 2**63: raise ValueError("reference timestamp is not exact int64")
        values = np.array(tokens[1:], dtype=float)
        if not np.isfinite(values).all() or abs(np.linalg.norm(values[3:])-1) > 1e-5: raise ValueError("invalid reference values")
        stamps.append(int(stamp))
        rows.append(values)
    if not stamps or any(b <= a for a, b in zip(stamps, stamps[1:])): raise ValueError("unordered reference timestamps")
    indices, paired, coverage = [], [], []
    for i, t in enumerate(estimate_times):
        t = int(t)
        right = bisect_left(stamps, t)
        row = dict(timestamp_ns=t, left=None, right=None, bracket_span_ns=None)
        if right < len(stamps) and stamps[right] == t:
            v = rows[right]
            row.update(status="exact", left=right, right=right, bracket_span_ns=0, fraction=0.)
            translation, q = v[:3], v[3:]
        elif right == 0 or right == len(stamps):
            row["status"] = "before_reference" if right == 0 else "after_reference"
            coverage.append(row)
            continue
        else:
            left = right-1
            span = stamps[right]-stamps[left]
            row.update(left=left, right=right, bracket_span_ns=span)
            if span > 100_000_000:
                row["status"] = "reference_gap"
                coverage.append(row)
                continue
            alpha = (t-stamps[left])/span
            translation = rows[left][:3]+alpha*(rows[right][:3]-rows[left][:3])
            a, b = rows[left][3:]/np.linalg.norm(rows[left][3:]), rows[right][3:]/np.linalg.norm(rows[right][3:])
            if a@b < 0: b = -b
            theta = np.arccos(np.clip(a@b, -1., 1.))
            q = (1-alpha)*a+alpha*b if theta < 1e-8 else (np.sin((1-alpha)*theta)*a+np.sin(alpha*theta)*b)/np.sin(theta)
            row.update(status="interpolated", fraction=alpha)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = quaternion(q), translation
        indices.append(i)
        paired.append(pose)
        coverage.append(row)
    return np.array(indices, dtype=np.int64), np.array(paired), coverage, stamps


def verify(folder=None):
    folder = Path(folder).resolve() if folder is not None else Path(__file__).resolve().parent
    repo = folder.parents[2]
    manifest = read(folder/"manifest.json")
    for mapping, base in [(manifest["files_sha256"], folder), (manifest["dependencies"], repo), (manifest["checkout_sources_sha256"], repo)]:
        for name, digest in mapping.items():
            path = (base/name).resolve()
            if not path.is_relative_to(base) or sha(path) != digest: raise ValueError("frozen file hash mismatch")
    protocol, freeze, closure, summary, oracle = [read(folder/n) for n in ["protocol.json", "controller-freeze.json", "estimates-receipt.json", "summary.json", "oracle-receipt.json"]]
    if (protocol["policies"] != POLICIES or list(protocol["arms"]) != ARMS or protocol["scan_indices"] != list(range(1, 136))
            or [r["scan_index"] for r in protocol["excluded_scans"]] != [0, 136]
            or summary["original_scan_indices"] != protocol["scan_indices"] or summary["estimates_per_case"] != 135
            or summary["physical_provenance_certified"] is not False or summary["defaults_changed"] is not False
            or not summary["all_estimates_frozen_before_scoring"]
            or closure["snapshots"] != 1080 or closure["points_per_case"] != 3_240_000
            or closure["map_matching_independently_replayed"] is not False): raise ValueError("experiment cohort/boundary differs")
    if (closure["protocol_sha256"] != sha(folder/"protocol.json") or freeze["protocol_sha256"] != sha(folder/"protocol.json")
            or closure["controller_freeze_sha256"] != sha(folder/"controller-freeze.json")): raise ValueError("experiment freeze differs")
    boundary = read(folder/"scoring-boundary.json")
    if not boundary["all_eight_estimates_and_snapshots_closed"] or boundary["full_selected_reference_parsed_before_this_boundary"]:
        raise ValueError("reference scoring boundary differs")
    times, ids = np.array(protocol["reference_timestamps_ns"], dtype=np.int64), np.array(protocol["scan_indices"], dtype=np.int64)
    indices, reference, coverage, ref_stamps = reference_pairs(folder/"reference-source.txt", times)
    member = read(folder/"reference-member.json")
    if (member["member"] != "gt/flatwall_02.txt" or member["sha256"] != sha(folder/"reference-source.txt")
            or member["rows"] != len(ref_stamps) or member["first_timestamp_ns"] != ref_stamps[0] or member["last_timestamp_ns"] != ref_stamps[-1]):
        raise ValueError("selected reference identity differs")
    with np.load(folder/"paired-reference.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["cohort_indices"], indices)
        np.testing.assert_array_equal(saved["scan_indices"], ids[indices])
        np.testing.assert_array_equal(saved["timestamps_ns"], times[indices])
        close(saved["reference"], reference)
    if summary["coverage_counts"] != dict(Counter(r["status"] for r in coverage)) or summary["paired_estimates_per_case"] != len(indices):
        raise ValueError("reference coverage differs")
    catalog = read(folder/"snapshot-catalog.json")
    if [(r["policy"], r["arm"], r["scan_index"]) for r in catalog] != [(p, a, int(i)) for i in ids for p in POLICIES for a in ARMS]:
        raise ValueError("incomplete native snapshot catalog")
    for r in catalog:
        for name, item in r["files"].items():
            if closure["output_files_sha256"][name] != item["sha256"] or item["bytes"] <= 0: raise ValueError("snapshot receipt binding differs")
    cases, rpe_positions = {}, np.flatnonzero(np.diff(ids[indices]) == 1)
    for policy in POLICIES:
        for arm in ARMS:
            key, destination = (policy, arm), folder/policy/arm
            with np.load(destination/"trajectory.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["original_scan_indices"], ids)
                np.testing.assert_array_equal(data["timestamps_ns"], times)
                entire = data["poses"].copy()
            poses(entire)
            if len(entire) != 135: raise ValueError("incomplete trajectory")
            estimate = entire[indices]
            reports, samples, edges = [read(destination/n) for n in ["per-scan.json", "samples.json", "edges.json"]]
            if [r["scan_index"] for r in reports] != ids.tolist() or [r["reference_ns"] for r in reports] != times.tolist(): raise ValueError("scan report identities differ")
            origin = reference[0]@inverse(estimate[0])
            anchored = origin@estimate
            errors = dict(origin_position_ape_m=np.linalg.norm(anchored[:, :3, 3]-reference[:, :3, 3], axis=1),
                origin_rotation_ape_deg=angles(reference[:, :3, :3].transpose(0, 2, 1)@anchored[:, :3, :3]))
            result = summary["controls"][policy][arm]
            close(result["origin_world_alignment"], origin)
            x, y = estimate[:, :3, 3]-estimate[:, :3, 3].mean(0), reference[:, :3, 3]-reference[:, :3, 3].mean(0)
            u, s, vt = np.linalg.svd(x.T@y)
            spectra = {"estimate": np.linalg.svd(x, compute_uv=False), "reference": np.linalg.svd(y, compute_uv=False), "cross_covariance": s}
            ranks = {k: int(np.sum(v > v[0]*1e-10)) if v[0] else 0 for k, v in spectra.items()}
            if result["position_fit"]["ranks"] != ranks: raise ValueError("position-fit rank differs")
            for k, v in spectra.items(): close(result["position_fit"]["singular_values"][k], v)
            if min(ranks.values()) >= 2:
                r = vt.T@np.diag([1, 1, 1 if np.linalg.det(vt.T@u.T) >= 0 else -1])@u.T
                t = reference[:, :3, 3].mean(0)-r@estimate[:, :3, 3].mean(0)
                if result["position_fit"]["status"] != "identified" or result["position_fit"]["scale"] != 1.: raise ValueError("proper rigid fit differs")
                close(np.array(result["position_fit"]["world_alignment"])[:3, :3], r)
                close(np.array(result["position_fit"]["world_alignment"])[:3, 3], t)
                errors["rigid_position_ape_m"] = np.linalg.norm(estimate[:, :3, 3]@r.T+t-reference[:, :3, 3], axis=1)
            elif result["position_fit"]["status"] != "degenerate_rank": raise ValueError("rank guard missing")
            delta = np.array([inverse(inverse(reference[i])@reference[i+1])@(inverse(estimate[i])@estimate[i+1]) for i in rpe_positions]).reshape(-1, 4, 4)
            errors["consecutive_rpe_translation_m"], errors["consecutive_rpe_rotation_deg"] = np.linalg.norm(delta[:, :3, 3], axis=1), angles(delta[:, :3, :3])
            if len(samples) != 135: raise ValueError("incomplete coverage rows")
            paired_lookup = {int(index): j for j, index in enumerate(indices)}
            for i, row in enumerate(samples):
                if row["cohort_index"] != i or row["scan_index"] != ids[i]: raise ValueError("coverage scan identity differs")
                for k, v in coverage[i].items():
                    if isinstance(v, float): close(row[k], v)
                    elif row[k] != v: raise ValueError("exact reference bracket differs")
                if i not in paired_lookup:
                    if row["errors"] is not None: raise ValueError("uncovered pose has errors")
                else:
                    for k in [k for k in errors if "ape_" in k]: close(row["errors"][k], errors[k][paired_lookup[i]], 1e-5 if k.endswith("deg") else 2e-8)
            if [(e["left_scan_index"], e["right_scan_index"]) for e in edges] != [(int(ids[indices[i]]), int(ids[indices[i+1]])) for i in rpe_positions]: raise ValueError("RPE bridges an original exclusion")
            for j, i in enumerate(rpe_positions):
                if edges[j]["elapsed_ns"] != int(times[indices[i+1]])-int(times[indices[i]]): raise ValueError("edge time differs")
                close(edges[j]["translation_m"], errors["consecutive_rpe_translation_m"][j])
                close(edges[j]["rotation_deg"], errors["consecutive_rpe_rotation_deg"][j], 1e-5)
            if set(result["metrics"]) != set(errors): raise ValueError("metric inventory differs")
            for k, v in errors.items(): check_stats(result["metrics"][k], v, 1e-5 if k.endswith("deg") else 2e-8)
            for name, values in [("estimate_m", estimate), ("reference_m", reference)]: close(result["consecutive_path_lengths"][name], np.linalg.norm(np.diff(values[:, :3, 3], axis=0)[rpe_positions], axis=1).sum())
            frames = [json.loads(line) for line in (destination/"frames.jsonl").read_text(encoding="utf-8").splitlines()]
            if [r["scan_index"] for r in frames] != ids.tolist(): raise ValueError("incomplete frame diagnostics")
            for i, row in enumerate(frames):
                predicted = np.eye(4) if i == 0 else (entire[i-1] if policy == "last-pose" or i == 1 else entire[i-1]@(inverse(entire[i-2])@entire[i-1]))
                close(row["prediction_pose"], predicted)
                close(row["returned_pose"], entire[i])
                correction = inverse(predicted)@entire[i]
                close(row["prediction_correction_translation_m"], np.linalg.norm(correction[:3, 3]))
                close(row["prediction_correction_rotation_deg"], angles(correction[:3, :3]), 1e-5)
                if row["source_points"] != reports[i]["registration"]["registration_points"] or row["termination"] != reports[i]["registration"]["termination"]: raise ValueError("diagnostic report differs")
                h = np.array(row["normal_matrix"])
                if row["prediction_only"] is not (not row["alignment_eligible"]): raise ValueError("prediction-only label differs")
                if row["prediction_only"]:
                    if row["matches"] or row["geometry"] is not None or np.any(h) or np.any(row["right_hand_side"]): raise ValueError("prediction-only support differs")
                    if row["prediction_correction_translation_m"] > 2e-8 or row["prediction_correction_rotation_deg"] > 1e-5: raise ValueError("unregistered correction differs")
                if row["geometry"] is not None:
                    d = np.array([1, 1, 1, *[1/row["radius_m"]]*3])
                    scaled = h*d[:, None]*d[None, :]/row["weight_sum"]
                    eigen = np.linalg.eigvalsh((scaled+scaled.T)/2)
                    close(row["geometry"]["eigenvalues"], eigen)
                    close(row["geometry"]["min_max_ratio"], eigen[0]/eigen[-1])
                    if row["geometry"]["weak_directions_below_1e-3"] != np.count_nonzero(eigen/eigen[-1] < 1e-3): raise ValueError("spectrum threshold differs")
            diagnostic = summary["diagnostics"][policy][arm]
            if diagnostic["termination_counts"] != dict(Counter(r["termination"] for r in frames)) or diagnostic["prediction_only_scans"] != [r["scan_index"] for r in frames if r["prediction_only"]]: raise ValueError("termination summary differs")
            supported = [r for r in frames if r["geometry"] is not None]
            if diagnostic["supported_spectra"] != len(supported) or diagnostic["weak_spectra"] != sum(r["geometry"]["weak_directions_below_1e-3"] > 0 for r in supported): raise ValueError("spectrum summary differs")
            for k, values in {"source_points": [r["source_points"] for r in frames], "spectrum_ratio": [r["geometry"]["min_max_ratio"] for r in supported],
                    **{k: [r[k] for r in frames] for k in ["prediction_correction_translation_m", "prediction_correction_rotation_deg"]}}.items():
                check_stats(diagnostic[k], values, 1e-5 if k.endswith("deg") else 2e-8)
            entry = oracle["controls"][policy][arm]
            if entry["paired_estimates"] != len(indices) or entry["consecutive_edges"] != len(edges): raise ValueError("oracle coverage differs")
            for k, v in entry["maximum_abs_differences"].items():
                if not 0 <= v <= (1e-5 if k.endswith("deg") else 2e-8): raise ValueError("oracle tolerance exceeded")
            cases[key] = entire, reports
            for name in ["trajectory.npz", "per-scan.json", "frames.jsonl"]:
                if closure["output_files_sha256"][f"{policy}/{arm}/{name}"] != sha(destination/name): raise ValueError("estimate closure binding differs")
    for arm in ARMS:
        a, b = cases["repeat-motion", arm], cases["last-pose", arm]
        np.testing.assert_array_equal(a[0][:2].view(np.uint64), b[0][:2].view(np.uint64))
        if a[1][:2] != b[1][:2] or [r["registration"]["registration_points"] for r in a[1]] != [r["registration"]["registration_points"] for r in b[1]]: raise ValueError("policy-equivalent prefix/source counts differ")
    cross = read(folder/"cross-recording.json")
    prior = read(repo/cross["prior_summary"])
    if sha(repo/cross["prior_summary"]) != cross["prior_summary_sha256"] or len(cross["cases"]) != 16: raise ValueError("prior cross-recording source differs")
    if [(r["recording"], r["policy"], r["arm"]) for r in cross["cases"]] != [(s, p, a) for s in ["flatwall_01", "flatwall_02"] for p in POLICIES for a in ARMS]: raise ValueError("cross-recording cases differ")
    for row in cross["cases"]:
        metrics = (prior if row["recording"] == "flatwall_01" else summary)["controls"][row["policy"]][row["arm"]]["metrics"]
        for key in ["rigid_position_ape_m", "origin_position_ape_m", "consecutive_rpe_translation_m", "consecutive_rpe_rotation_deg"]:
            if row[key] != (metrics[key]["rmse"] if metrics.get(key) else None): raise ValueError("cross-recording arithmetic differs")
    return dict(cases=8, estimates_per_case=135, paired_per_case=len(indices), consecutive_edges_per_case=len(rpe_positions), snapshots=1080,
        conditional_rigid_position_rmse_m={p: {a: summary["controls"][p][a]["metrics"].get("rigid_position_ape_m", {}).get("rmse") for a in ARMS} for p in POLICIES},
        physical_provenance_certified=False, scope="Published original selected reference, independent exact-time association, all eight poses/errors and diagnostics. No raw-bag/native snapshot map-search rerun or physical-calibration certificate.")


if __name__ == "__main__": print(json.dumps(verify(), indent=2))
