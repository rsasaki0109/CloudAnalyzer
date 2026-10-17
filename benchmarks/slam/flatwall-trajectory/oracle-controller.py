"""Independent installed-evo arithmetic on one frozen exploratory cohort."""
from bisect import bisect_left
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
from evo.core import geometry, lie_algebra as lie, metrics, trajectory, transformations as tr
from evo.tools import file_interface

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "notes/slam-flatwall-trajectory-20261017"
sys.path.insert(0, str(ROOT / "scripts"))
import audit_flatwall_trajectory as audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def main():
    freeze_path = STUDY / "controller-freeze.json"
    freeze = read(freeze_path)
    protocol_path = STUDY / "protocol.json"
    protocol = read(protocol_path)
    inputs = {k: ROOT / v["path"] for k, v in protocol["input_artifacts"].items()}
    prior = read(STUDY / "prior-artifacts.json")

    def verify():
        for name, expected in (prior | freeze["source_files_sha256"]).items():
            assert sha(ROOT / name) == expected, name
        for name, expected in freeze["dependency_files_sha256"].items():
            assert sha(Path(name)) == expected, name
        assert sha(protocol_path) == freeze["protocol_sha256"]

    verify()
    started = datetime.now(timezone.utc).isoformat()
    summary = audit.audit(protocol_path, freeze_path, STUDY / "results")
    out = STUDY / "oracle"
    out.mkdir(exist_ok=False)
    try:
        with tarfile.open(inputs["reference_archive"], "r:gz") as stream:
            member = stream.getmember("gt/flatwall_01.txt")
            assert member.isfile() and member.size <= 8_388_608
            with stream.extractfile(member) as source:
                raw = source.read()
        tokens = [line.split() for line in raw.decode("utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
        exact_times = [int(Decimal(row[0]) * 1_000_000_000) for row in tokens]
        assert all(b > a for a, b in zip(exact_times[:-1], exact_times[1:]))
        # Whitespace canonicalization only; retain every original numeric token.
        canonical = "\n".join(" ".join(row) for row in tokens) + "\n"
        parsed = file_interface.read_tum_trajectory_file(io.StringIO(canonical))
        assert parsed.num_poses == len(exact_times)
        rows = read(STUDY / "results/samples.json")
        reference, expected_indices = [], []
        for row in rows:
            i, t = row["estimate_index"], row["timestamp_ns"]
            right = bisect_left(exact_times, t)
            if right < len(exact_times) and exact_times[right] == t:
                status, left, span = "exact", right, 0
            elif right == 0 or right == len(exact_times):
                status = "before_reference" if right == 0 else "after_reference"
                assert row["status"] == status
                continue
            else:
                left = right - 1
                span = exact_times[right] - exact_times[left]
                status = "reference_gap" if span > 100_000_000 else "interpolated"
            assert (row["status"], row["left"], row["right"], row["bracket_span_ns"]) == (status, left, right, span)
            if status == "reference_gap":
                continue
            if status == "exact":
                pose = parsed.poses_se3[right].copy()
            else:
                alpha = (t-exact_times[left])/span
                q = tr.quaternion_slerp(parsed.orientations_quat_wxyz[left].copy(), parsed.orientations_quat_wxyz[right].copy(), alpha)
                pose = tr.quaternion_matrix(q)
                pose[:3, 3] = parsed.positions_xyz[left] + alpha*(parsed.positions_xyz[right]-parsed.positions_xyz[left])
            reference.append(pose)
            expected_indices.append(i)
        ref = np.asarray(reference).reshape((-1, 4, 4))
        with np.load(STUDY / "results/paired.npz", allow_pickle=False) as pairs:
            est, actual_ref = pairs["estimated"].copy(), pairs["reference"].copy()
            indices = pairs["original_indices"].copy()
            np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_allclose(ref, actual_ref, rtol=0, atol=2e-8)
        maximum = {"reference_pose_matrix": float(np.max(np.abs(ref-actual_ref))) if len(ref) else 0.}
        independent = {}

        def ape(name, estimated, relation):
            metric = metrics.APE(relation)
            metric.process_data((trajectory.PosePath3D(poses_se3=ref), trajectory.PosePath3D(poses_se3=estimated)))
            independent[name] = metric.error

        if len(ref):
            origin = ref[0] @ lie.se3_inverse(est[0])
            ape("origin_position_ape_m", origin @ est, metrics.PoseRelation.translation_part)
            ape("origin_rotation_ape_deg", origin @ est, metrics.PoseRelation.rotation_angle_deg)
            if summary["position_fit"]["status"] == "identified":
                r, t, scale = geometry.umeyama_alignment(est[:, :3, 3].T, ref[:, :3, 3].T, with_scale=False)
                assert scale == 1.
                fit = lie.se3(r, t)
                np.testing.assert_allclose(fit, summary["position_fit"]["world_alignment"], rtol=0, atol=2e-8)
                maximum["world_alignment_matrix"] = float(np.max(np.abs(fit-np.array(summary["position_fit"]["world_alignment"]))))
                ape("rigid_position_ape_m", fit @ est, metrics.PoseRelation.translation_part)
            edges = np.flatnonzero(np.diff(indices) == 1)
            delta = [metrics.RPE.rpe_base(ref[i], ref[i+1], est[i], est[i+1]) for i in edges]
            independent["consecutive_rpe_translation_m"] = np.array([np.linalg.norm(d[:3, 3]) for d in delta])
            independent["consecutive_rpe_rotation_deg"] = np.array([abs(lie.so3_log_angle(d[:3, :3], degrees=True)) for d in delta])
        with np.load(STUDY / "results/errors.npz", allow_pickle=False) as actual:
            assert set(actual.files) == set(independent)
            for name, expected in independent.items():
                np.testing.assert_allclose(actual[name], expected, rtol=0, atol=1e-5 if name.endswith("deg") else 2e-8)
                maximum[name] = float(np.max(np.abs(actual[name]-expected))) if len(expected) else 0.
        np.savez_compressed(out / "evo-errors.npz", **independent)
        verify()
        record = dict(started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      controller_freeze_sha256=sha(freeze_path), protocol_sha256=sha(protocol_path),
                      full_reference_first_read_after_controller_freeze=True, selected_reference_member="gt/flatwall_01.txt",
                      reference_rows=len(exact_times), estimates_checked=len(rows), paired_estimates=len(ref),
                      consecutive_edges_checked=len(delta) if len(ref) else 0, maximum_abs_differences=maximum,
                      reference_timestamp_method="Independent bisect on exact Decimal nanoseconds; evo floating Unix timestamps are not used for association.",
                      independent_methods="Installed evo TUM parser and quaternion SLERP, no-scale Umeyama, APE, SE3 inverse and RPE base.",
                      prior_artifacts_verified=len(prior), source_files_verified=len(freeze["source_files_sha256"]), selected_dependency_files_verified=len(freeze["dependency_files_sha256"]),
                      input_hashes_unchanged=True, physical_provenance_certified=False,
                      outputs_sha256={"evo-errors.npz": sha(out / "evo-errors.npz")})
        save(out / "receipt.json", record)
        print(json.dumps(dict(coverage=summary["coverage_counts"], paired=summary["paired_estimates"], metrics=summary["metrics"], oracle=maximum), indent=2))
    except Exception as exc:
        save(out / "failure.json", dict(error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == "__main__":
    main()
