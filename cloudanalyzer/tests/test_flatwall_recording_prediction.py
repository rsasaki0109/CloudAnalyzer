"""Count, original-edge and bounded-member controls, without real reference reads."""
import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("recording_prediction", ROOT / "scripts/audit_flatwall_recording_prediction.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_cohort_size_and_internal_exclusions_come_from_sensor_coverage():
    imu = np.array([100, 200, 60_000_201, 60_000_301], dtype=np.int64)
    rows = []
    for index, header in enumerate([99, 100, 150, 250, 60_000_201, 60_000_300, 60_000_302]):
        rows.append(dict(scan_index=index, header_ns=header, offset_min_ns=0, offset_max_ns=1,
                         header_coverage=audit.inputs.coverage(imu, header, header+1)))
    ids, times, excluded = audit.cohort(rows, imu)
    assert ids.tolist() == [1, 2, 4, 5]
    assert times.tolist() == [101, 151, 60_000_202, 60_000_301]
    assert [r["scan_index"] for r in excluded] == [0, 3, 6]
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 0, 3] = [0, 1, 20, 21]
    result, _ = audit.scoring.score(poses, poses, ids, times)
    assert [(e["left_estimate_index"], e["right_estimate_index"]) for e in result["edges"]] == [(1, 2), (4, 5)]
    assert result["consecutive_path_lengths"]["estimate_m"] == 2.
    rows[3]["header_coverage"]["status"] = "covered"
    with pytest.raises(ValueError, match="coverage"):
        audit.cohort(rows, imu)


@pytest.mark.parametrize("count", [3, 135, 185])
def test_estimates_require_the_exact_declared_cohort_at_any_size(tmp_path, count):
    ids = np.arange(1, count+1, dtype=np.int64)
    times = ids+1_656_505_431_000_000_000
    poses = np.tile(np.eye(4), (count, 1, 1))
    np.savez(tmp_path/"trajectory.npz", original_scan_indices=ids, timestamps_ns=times, poses=poses)
    audit.controls.save(tmp_path/"per-scan.json", [dict(scan_index=int(i), reference_ns=int(t)) for i, t in zip(ids, times)])
    actual, _ = audit.load_estimates(tmp_path, ids, times)
    np.testing.assert_array_equal(actual, poses)
    changed = times.copy()
    changed[-1] += 1
    with pytest.raises(AssertionError):
        audit.load_estimates(tmp_path, ids, changed)


def test_policies_use_only_previous_poses_and_anchor_without_an_explicit_prediction():
    class Spy:
        def __init__(self):
            self.calls = []
        def register(self, points, **kwargs):
            self.calls.append(kwargs)
            return np.eye(4)
    native = Spy()
    points = np.zeros((10, 3))
    for policy in audit.POLICIES:
        _, predicted = audit.step(native, points, [], policy)
        assert native.calls[-1] == {}
        np.testing.assert_array_equal(predicted, np.eye(4))
    previous = np.tile(np.eye(4), (2, 1, 1))
    previous[:, 0, 3] = [2., 5.]
    _, repeated = audit.step(native, points, list(previous), "repeat-motion")
    assert native.calls[-1] == {} and repeated[0, 3] == 8.
    _, held = audit.step(native, points, list(previous), "last-pose")
    np.testing.assert_array_equal(native.calls[-1]["relative_prediction"], np.eye(4))
    np.testing.assert_array_equal(held, previous[-1])
    with pytest.raises(ValueError, match="policy"):
        audit.step(native, points, [], "other")


def archive(path, entries):
    with tarfile.open(path, "w:gz") as stream:
        for name, payload, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(payload) if kind == tarfile.REGTYPE else 0
            stream.addfile(info, io.BytesIO(payload) if info.size else None)


TEXT = b"1656505431.000000001 0 0 0 0 0 0 1\n1656505431.000000002 1 0 0 0 0 0 1\n"


def test_only_selected_content_is_parsed_and_original_text_is_preserved(tmp_path):
    path = tmp_path/"references.tar.gz"
    archive(path, [("gt/flatwall_01.txt", b"invalid UTF8 \xff", tarfile.REGTYPE),
                   ("gt/flatwall_02.txt", TEXT, tarfile.REGTYPE)])
    copied = tmp_path/"selected.txt"
    times, _, meta = audit.selected_reference.read_selected(path, "gt/flatwall_02.txt", copy_to=copied)
    assert times.tolist() == [1_656_505_431_000_000_001, 1_656_505_431_000_000_002]
    assert copied.read_bytes() == TEXT and meta["member"] == "gt/flatwall_02.txt"
    with pytest.raises(FileExistsError):
        audit.selected_reference.read_selected(path, "gt/flatwall_02.txt", copy_to=copied)


@pytest.mark.parametrize("case", ["duplicate", "symlink", "empty", "missing", "traversal", "oversize"])
def test_ambiguous_or_unbounded_reference_members_are_rejected(tmp_path, case):
    path = tmp_path/"references.tar.gz"
    entries = [("gt/flatwall_02.txt", TEXT, tarfile.REGTYPE)]
    if case == "duplicate":
        entries += entries
    elif case == "symlink":
        entries = [("gt/flatwall_02.txt", b"", tarfile.SYMTYPE)]
    elif case == "empty":
        entries = [("gt/flatwall_02.txt", b"", tarfile.REGTYPE)]
    elif case == "missing":
        entries = []
    elif case == "oversize":
        entries = [("gt/flatwall_02.txt", b"\0"*(audit.selected_reference.MAX_BYTES+1), tarfile.REGTYPE)]
    archive(path, entries)
    with pytest.raises(ValueError):
        audit.selected_reference.read_selected(path, "../gt/flatwall_02.txt" if case == "traversal" else "gt/flatwall_02.txt")
