"""Phase 29: end-to-end integration test for the canonical external SLAM
driver plugin shipped under ``plugins/cloudanalyzer-driver-example/``.

This test is **the proof** that the entry-point pathway introduced in
Phase 28 actually works on a real, pip-installable package — not just
in-process monkey-patches. When ``cloudanalyzer-driver-example`` is
installed (``pip install -e plugins/cloudanalyzer-driver-example``):

- It registers itself under the ``cloudanalyzer.slam_run_drivers``
  entry-point group with the name ``example``.
- ``ca.core.slam_run.list_drivers()`` now lists it.
- ``ca.core.slam_run.get_driver("example")`` instantiates it.
- ``ca slam-run --driver example`` runs end-to-end on the bundled
  synthetic-figure8 scans.

If the plugin is not installed the whole module is skipped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_example_pkg = pytest.importorskip(
    "cloudanalyzer_driver_example",
    reason="cloudanalyzer-driver-example not installed "
    "(see plugins/cloudanalyzer-driver-example/README.md)",
)


def test_example_plugin_class_satisfies_protocol() -> None:
    """The class re-exported from the plugin package must satisfy the
    ``SlamRunDriver`` protocol (``name`` + ``run``)."""

    from ca.core.slam_run import SlamRunDriver

    drv = _example_pkg.Open3DICPSlamDriver()
    assert isinstance(drv, SlamRunDriver)
    assert drv.name == "open3d_icp"


def test_example_plugin_visible_via_registry() -> None:
    """After install, ``list_drivers()`` must include ``"example"`` and
    ``get_driver("example")`` must instantiate the plugin's class."""

    from ca.core.slam_run import get_driver, list_drivers

    names = list_drivers()
    assert "example" in names, (
        f"plugin not discovered; list_drivers()={names}. "
        "Run `pip install -e plugins/cloudanalyzer-driver-example` first."
    )

    drv = get_driver("example")
    assert drv.name == "open3d_icp"
    assert isinstance(drv, _example_pkg.Open3DICPSlamDriver)


def test_example_plugin_via_cli_end_to_end(tmp_path: Path) -> None:
    """The full CLI path resolves ``--driver example`` through the registry,
    produces the three standard artifacts (``trajectory.tum``, ``map.ply``,
    ``summary.json``), and runs to completion. The driver is a deliberately
    plain ICP and is not expected to clear the synthetic-figure8 gate — we
    only assert the pipeline is wired."""

    repo_root = Path(__file__).resolve().parents[2]
    scans_dir = repo_root / "benchmarks" / "slam" / "synthetic-figure8" / "scans"
    if not scans_dir.is_dir():
        pytest.skip("synthetic-figure8 scans not present")

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "cloudanalyzer_cli.main",
        "slam-run",
        str(scans_dir),
        str(out_dir),
        "--driver",
        "example",
        "--max-range",
        "5",
        "--voxel-size",
        "0.5",
        "--frame-period",
        "0.1",
        "--max-frames",
        "30",  # Open3D point-to-point ICP is slow; cap for CI runtime.
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["driver"] == "open3d_icp"
    assert summary["frames_processed"] == 30
    assert "open3d_icp" in summary["driver_metadata"]
    assert (out_dir / "trajectory.tum").is_file()
    assert (out_dir / "map.ply").is_file()
