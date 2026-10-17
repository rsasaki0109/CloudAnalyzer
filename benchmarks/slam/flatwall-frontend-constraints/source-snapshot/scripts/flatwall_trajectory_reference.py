"""Conditional trajectory arithmetic; file syntax does not establish body/clock provenance."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

MAX_BRACKET_NS = 100_000_000
QUATERNION_ATOL = 1e-5
RANK_RTOL = 1e-10


def timestamps(values):
    a = np.asarray(values)
    if a.ndim != 1 or not len(a) or a.dtype.kind != "i" or a.dtype.itemsize != 8:
        raise ValueError("timestamps must be a nonempty int64 vector")
    if any(int(b) <= int(a) for a, b in zip(a[:-1], a[1:])):
        raise ValueError("timestamps must be strictly increasing")
    return a


def validate_poses(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (4, 4) or not len(a) or not np.isfinite(a).all():
        raise ValueError("poses must be finite nonempty (N,4,4) matrices")
    if not np.allclose(a[:, 3], [0., 0., 0., 1.], rtol=0, atol=1e-12):
        raise ValueError("invalid homogeneous row")
    r = a[:, :3, :3]
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-6):
        raise ValueError("rotation must be orthogonal")
    if not np.allclose(np.linalg.det(r), 1., rtol=0, atol=1e-6):
        raise ValueError("rotation must be proper")
    return a


def exact_nanoseconds(token):
    """Convert decimal digits without context-dependent Decimal arithmetic."""
    value = Decimal(token)
    if not value.is_finite():
        raise ValueError("timestamp must be finite")
    if not value:
        return 0
    sign, digits, exponent = value.as_tuple()
    exponent += 9
    if exponent < 0:
        integer_digits = len(digits) + exponent
        if integer_digits <= 0 or any(digits[integer_digits:]):
            raise ValueError("timestamp must be exact integer nanoseconds")
        digits, exponent = digits[:integer_digits], 0
    if len(digits) + exponent > 19:
        raise ValueError("timestamp exceeds int64 nanoseconds")
    ns = int("".join(str(d) for d in digits)) * 10**exponent * (-1 if sign else 1)
    if not -(2**63) <= ns < 2**63:
        raise ValueError("timestamp exceeds int64 nanoseconds")
    return ns


def parse_tum_assumption(text):
    """Explicit assumed seconds/xyz/xyzw layout; retain exact decimal time."""
    times, rows = [], []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"reference line {line_number} requires eight columns")
        try:
            ns = exact_nanoseconds(fields[0])
            values = np.asarray(fields[1:], dtype=np.float64)
        except (InvalidOperation, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid reference line {line_number}: {exc}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite reference line {line_number}")
        times.append(int(ns))
        rows.append(values)
    times = timestamps(np.asarray(times, dtype=np.int64))
    rows = np.asarray(rows)
    norms = np.linalg.norm(rows[:, 3:], axis=1)
    norm_error = np.abs(norms - 1.)
    if not np.isfinite(norms).all() or np.any(norm_error > QUATERNION_ATOL):
        raise ValueError("reference quaternion norm exceeds declared tolerance")
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, :3, 3] = rows[:, :3]
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 3:] / norms[:, None]).as_matrix()
    return times, poses, float(np.max(norm_error))


def read_selected_reference(archive: Path):
    """Read one literal regular member without extracting archive paths."""
    with tarfile.open(archive, "r:gz") as stream:
        matches = [m for m in stream.getmembers() if m.name == "gt/flatwall_01.txt"]
        if len(matches) != 1 or not matches[0].isfile() or not 0 < matches[0].size <= 8_388_608:
            raise ValueError("expected one bounded regular gt/flatwall_01.txt member")
        with stream.extractfile(matches[0]) as member:
            data = member.read(8_388_609)
        if len(data) != matches[0].size:
            raise ValueError("reference member size mismatch")
    times, poses, norm_error = parse_tum_assumption(data.decode("utf-8"))
    return times, poses, {
        "member": "gt/flatwall_01.txt", "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "rows": len(times),
        "first_timestamp_ns": int(times[0]), "last_timestamp_ns": int(times[-1]),
        "maximum_original_quaternion_norm_error": norm_error,
        "interpretation": "Assumed TUM syntax and reference-world-from-LiDAR; archive-specific axes and clock mapping unverified.",
    }


def associate(estimate_times, reference_times, reference_poses):
    """Bracket in integer time; local fractional time avoids Unix float loss."""
    times, ref = timestamps(estimate_times), timestamps(reference_times)
    poses = validate_poses(reference_poses)
    if len(ref) != len(poses):
        raise ValueError("reference times and poses differ in length")
    rows, indices, paired = [], [], []
    for i, stamp in enumerate(times):
        t = int(stamp)
        right = int(np.searchsorted(ref, stamp))
        row = {"estimate_index": i, "timestamp_ns": t, "left": None, "right": None, "bracket_span_ns": None}
        if right < len(ref) and int(ref[right]) == t:
            row.update(status="exact", left=right, right=right, bracket_span_ns=0, fraction=0.)
            pose = poses[right].copy()
        elif right == 0 or right == len(ref):
            row["status"] = "before_reference" if right == 0 else "after_reference"
            rows.append(row)
            continue
        else:
            left = right - 1
            span = int(ref[right]) - int(ref[left])
            row.update(left=left, right=right, bracket_span_ns=span)
            if span > MAX_BRACKET_NS:
                row["status"] = "reference_gap"
                rows.append(row)
                continue
            alpha = (t - int(ref[left])) / span
            pose = np.eye(4)
            pose[:3, 3] = (1 - alpha) * poses[left, :3, 3] + alpha * poses[right, :3, 3]
            pose[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(poses[left:right+1, :3, :3]))([alpha]).as_matrix()[0]
            row.update(status="interpolated", fraction=alpha)
        rows.append(row)
        indices.append(i)
        paired.append(pose)
    return rows, np.asarray(indices, dtype=np.int64), np.asarray(paired).reshape((-1, 4, 4))


def inverse(pose):
    out = np.eye(4)
    out[:3, :3] = pose[:3, :3].T
    out[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return out


def angles_deg(rotations):
    r = np.asarray(rotations)
    skew = np.stack([r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]], axis=-1)
    return np.degrees(np.arctan2(np.linalg.norm(skew, axis=-1)/2., (np.trace(r, axis1=-2, axis2=-1)-1.)/2.))


def rigid_position_fit(estimated, reference):
    """One proper SE3 position fit. Expose rank instead of choosing a hidden gauge."""
    pe, pr = estimated[:, :3, 3], reference[:, :3, 3]
    ce, cr = pe.mean(axis=0), pr.mean(axis=0)
    x, y = pe-ce, pr-cr
    u, sigma, vt = np.linalg.svd(x.T @ y)
    spectra = {"estimate": np.linalg.svd(x, compute_uv=False), "reference": np.linalg.svd(y, compute_uv=False), "cross_covariance": sigma}
    rank = {k: int(np.sum(s > s[0]*RANK_RTOL)) if s[0] > 0 else 0 for k, s in spectra.items()}
    info = {"singular_values": {k: s.tolist() for k, s in spectra.items()}, "ranks": rank, "scale": 1.0, "relative_rank_threshold": RANK_RTOL}
    if min(rank.values()) < 2:
        return None, dict(info, status="degenerate_rank", world_alignment=None)
    r = vt.T @ np.diag([1., 1., 1. if np.linalg.det(vt.T @ u.T) >= 0 else -1.]) @ u.T
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r, cr-r @ ce
    return transform, dict(info, status="identified", world_alignment=transform.tolist())


def summarize(values):
    a = np.asarray(values)
    if not len(a):
        return None
    if not np.isfinite(a).all():
        raise ValueError("nonfinite trajectory error")
    return {"count": len(a), "rmse": float(np.sqrt(np.mean(a*a))), "mean": float(a.mean()),
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def score(estimated, reference, original_indices, paired_times):
    est, ref = validate_poses(estimated), validate_poses(reference)
    indices, times = timestamps(original_indices), timestamps(paired_times)
    if est.shape != ref.shape or len(est) != len(indices) or len(est) != len(times):
        raise ValueError("paired pose, index and time lengths differ")
    fit, fit_info = rigid_position_fit(est, ref)
    origin = ref[0] @ inverse(est[0])
    anchored = origin @ est
    errors = {
        "origin_position_ape_m": np.linalg.norm(anchored[:, :3, 3]-ref[:, :3, 3], axis=1),
        "origin_rotation_ape_deg": angles_deg(ref[:, :3, :3].transpose(0, 2, 1) @ anchored[:, :3, :3]),
    }
    if fit is not None:
        aligned = fit @ est
        errors["rigid_position_ape_m"] = np.linalg.norm(aligned[:, :3, 3]-ref[:, :3, 3], axis=1)
    edges = np.flatnonzero(np.diff(indices) == 1)
    rpe = [inverse(inverse(ref[i]) @ ref[i+1]) @ (inverse(est[i]) @ est[i+1]) for i in edges]
    rpe = np.asarray(rpe).reshape((-1, 4, 4))
    errors["consecutive_rpe_translation_m"] = np.linalg.norm(rpe[:, :3, 3], axis=1)
    errors["consecutive_rpe_rotation_deg"] = angles_deg(rpe[:, :3, :3])
    samples = [{"estimate_index": int(index), **{k: float(v[i]) for k, v in errors.items() if "ape_" in k}} for i, index in enumerate(indices)]
    edge_rows = [{"left_estimate_index": int(indices[i]), "right_estimate_index": int(indices[i+1]),
                  "elapsed_ns": int(times[i+1])-int(times[i]),
                  "translation_m": float(errors["consecutive_rpe_translation_m"][j]),
                  "rotation_deg": float(errors["consecutive_rpe_rotation_deg"][j])} for j, i in enumerate(edges)]
    lengths = {name: float(np.linalg.norm(np.diff(a[:, :3, 3], axis=0)[edges], axis=1).sum()) for name, a in [("estimate_m", est), ("reference_m", ref)]}
    result = {"metrics": {k: summarize(v) for k, v in errors.items()}, "position_fit": fit_info,
              "origin_world_alignment": origin.tolist(), "consecutive_path_lengths": lengths,
              "samples": samples, "edges": edge_rows}
    return result, errors
