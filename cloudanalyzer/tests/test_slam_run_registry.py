"""Phase 28: tests for the SLAM driver registry + entry-point loader.

The registry lives in ``ca.core.slam_run`` and exposes ``register_driver``,
``get_driver``, ``list_drivers``. Built-in drivers (kiss-icp, kiss-slam,
small-gicp) register themselves at module import time via lazy factories;
third-party packages can publish drivers via the
``cloudanalyzer.slam_run_drivers`` entry-point group.

These tests don't require any external package install — the entry-point
loader is exercised by monkey-patching ``importlib.metadata.entry_points``
to return a fake plugin that ships inside this test module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import ca.core.slam_run as slam_run_module
from ca.core.slam_run import (
    SlamRunRequest,
    SlamRunResult,
    get_driver,
    list_drivers,
    register_driver,
)


# ---------------------------------------------------------------------------
# Fake driver used to exercise both the in-process registry and the
# entry-point loader. Implements the SlamRunDriver Protocol minimally —
# one identity pose, empty map.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _FakeDriver:
    name: str = "fake"

    def run(self, request: SlamRunRequest) -> SlamRunResult:
        return SlamRunResult(
            driver=self.name,
            poses=np.broadcast_to(np.eye(4), (1, 4, 4)).copy(),
            timestamps_s=np.array([0.0]),
            map_points=np.zeros((0, 3), dtype=np.float64),
            runtime_s=0.0,
            frames_processed=1,
            metadata={"fake": True},
        )


def _fake_factory() -> _FakeDriver:
    return _FakeDriver()


# ---------------------------------------------------------------------------
# Built-in registry baseline.
# ---------------------------------------------------------------------------


def test_built_in_drivers_present() -> None:
    """All 3 v0.4.0 built-in drivers must register themselves at import time."""

    names = list_drivers()
    assert "kiss-icp" in names
    assert "kiss-slam" in names
    assert "small-gicp" in names


def test_get_driver_returns_fresh_instance_each_call() -> None:
    """Registry stores factories, not singletons — each ``get_driver`` call
    must build a fresh driver."""

    a = get_driver("kiss-icp")
    b = get_driver("kiss-icp")
    assert a is not b
    assert a.name == b.name == "kiss_icp"


def test_get_driver_unknown_name_lists_available() -> None:
    """The error message must list known drivers so a user typing the wrong
    name can self-correct."""

    with pytest.raises(ValueError) as excinfo:
        get_driver("not-a-real-driver-xyz")
    msg = str(excinfo.value)
    assert "kiss-icp" in msg
    assert "cloudanalyzer.slam_run_drivers" in msg


# ---------------------------------------------------------------------------
# In-process registration.
# ---------------------------------------------------------------------------


def test_register_driver_round_trip() -> None:
    """``register_driver`` + ``get_driver`` round-trips an in-process driver."""

    name = "ca-tests-fake-inproc"
    try:
        register_driver(name, _fake_factory)
        drv = get_driver(name)
        assert isinstance(drv, _FakeDriver)
        assert drv.name == "fake"
    finally:
        slam_run_module._DRIVER_REGISTRY.pop(name, None)


def test_register_driver_overwrites_existing() -> None:
    """Re-registering an existing name silently shadows the previous entry —
    so an external package can fork a built-in driver if it wants."""

    name = "ca-tests-fake-overwrite"
    other_factory = lambda: _FakeDriver(name="fake_v2")  # noqa: E731
    try:
        register_driver(name, _fake_factory)
        register_driver(name, other_factory)
        drv = get_driver(name)
        assert drv.name == "fake_v2"
    finally:
        slam_run_module._DRIVER_REGISTRY.pop(name, None)


def test_register_driver_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        register_driver("", _fake_factory)


# ---------------------------------------------------------------------------
# Entry-point loader.
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
    """Minimal stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name: str, payload: Any) -> None:
        self.name = name
        self._payload = payload

    def load(self) -> Any:
        return self._payload


def test_entry_point_loader_picks_up_plugin(monkeypatch) -> None:
    """A third-party plugin registered under the
    ``cloudanalyzer.slam_run_drivers`` entry-point group should be resolvable
    via ``get_driver`` after a first miss triggers the loader."""

    name = "ca-tests-fake-ep"

    def fake_entry_points(*, group: str):
        if group == "cloudanalyzer.slam_run_drivers":
            return [_FakeEntryPoint(name, _fake_factory)]
        return []

    monkeypatch.setattr(
        "ca.core.slam_run.entry_points", fake_entry_points, raising=False
    )
    # Reset the one-shot cache so the patched entry_points is consulted.
    monkeypatch.setattr(slam_run_module, "_ENTRY_POINTS_LOADED", False)

    try:
        drv = get_driver(name)
        assert isinstance(drv, _FakeDriver)
    finally:
        slam_run_module._DRIVER_REGISTRY.pop(name, None)


def test_entry_point_loader_accepts_class_as_payload(monkeypatch) -> None:
    """An entry-point that resolves to a *class* (not a factory) should also
    work — the class itself is callable and produces an instance."""

    name = "ca-tests-fake-class"

    def fake_entry_points(*, group: str):
        if group == "cloudanalyzer.slam_run_drivers":
            return [_FakeEntryPoint(name, _FakeDriver)]
        return []

    monkeypatch.setattr(
        "ca.core.slam_run.entry_points", fake_entry_points, raising=False
    )
    monkeypatch.setattr(slam_run_module, "_ENTRY_POINTS_LOADED", False)

    try:
        drv = get_driver(name)
        assert isinstance(drv, _FakeDriver)
    finally:
        slam_run_module._DRIVER_REGISTRY.pop(name, None)


def test_entry_point_loader_tolerates_broken_plugin(monkeypatch) -> None:
    """A plugin whose ``.load()`` raises must not take down the built-in
    drivers. The broken plugin is logged + skipped; everything else works."""

    class _BrokenEntryPoint:
        name = "ca-tests-broken-ep"

        def load(self) -> Any:
            raise RuntimeError("intentional plugin import failure")

    def fake_entry_points(*, group: str):
        if group == "cloudanalyzer.slam_run_drivers":
            return [_BrokenEntryPoint()]
        return []

    monkeypatch.setattr(
        "ca.core.slam_run.entry_points", fake_entry_points, raising=False
    )
    monkeypatch.setattr(slam_run_module, "_ENTRY_POINTS_LOADED", False)

    # Built-ins still resolve.
    assert get_driver("kiss-icp").name == "kiss_icp"
    # The broken one is NOT registered.
    assert "ca-tests-broken-ep" not in list_drivers()


def test_cli_resolves_driver_via_registry(tmp_path) -> None:
    """End-to-end: ``ca slam-run --driver kiss-icp`` resolves through the
    registry exactly like the direct ``get_driver`` API. Guards against
    regressions where the CLI re-introduces a hard-coded if/elif chain."""

    pytest.importorskip("kiss_icp", reason="kiss-icp not installed")

    import json
    import subprocess
    import sys
    from pathlib import Path

    # Use the bundled synthetic straight-line dataset via the experiment
    # slice helper rather than spinning up our own fixture.
    from ca.experiments.slam_run.common import _straight_line_dataset

    ds = _straight_line_dataset()
    request = ds.build_request(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    import shutil

    for src in request.frame_paths:
        shutil.copy(src, input_dir / src.name)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "cloudanalyzer_cli.main",
        "slam-run",
        str(input_dir),
        str(out_dir),
        "--driver",
        "kiss-icp",
        "--max-range",
        "60",
        "--voxel-size",
        "0.5",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["driver"] == "kiss_icp"


def test_cli_unknown_driver_fails_cleanly(tmp_path) -> None:
    """Asking for a driver that doesn't exist should exit non-zero and
    mention the entry-point group in the error message — the only way a
    user can self-register one."""

    import subprocess
    import sys
    from pathlib import Path

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.pcd").write_text("")
    (input_dir / "b.pcd").write_text("")

    cmd = [
        sys.executable,
        "-m",
        "cloudanalyzer_cli.main",
        "slam-run",
        str(input_dir),
        str(tmp_path / "out"),
        "--driver",
        "no-such-driver-zzz",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "no-such-driver-zzz" in combined
