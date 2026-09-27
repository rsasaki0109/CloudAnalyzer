"""Record CloudCompare's results on two synthetic surveys, for the parity
tests in ``rust/crates/ca-core/tests/cloudcompare.rs``.

Usage::

    python scripts/make_cloudcompare_fixtures.py \\
        --cloudcompare "C:/Program Files/CloudCompare/CloudCompare.exe"

Writes ``rust/crates/ca-core/tests/cloudcompare/``: the two input clouds
(``before.xyz``, ``after.xyz``) and CloudCompare's outputs, mapped back to the
input order. The inputs are deterministic, so rerunning it only changes the
outputs if CloudCompare does.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rust" / "crates" / "ca-core" / "tests" / "cloudcompare"

# Settings shared with the Rust test.
SOR = (8, 1.0)  # CloudCompare counts the point itself: ours is k = 7.
CSF = ("RELIEF", 1.0, 0.3)  # scene, cloth resolution, class threshold
M3C2 = (2.0, 1.0, 2.0)  # normal diameter, projection diameter, depth (half-length)
VOLUME_STEP = 1.0


def terrain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.8 * np.sin(x / 11) + 0.6 * np.cos(y / 9)


def survey(seed: int, after: bool) -> np.ndarray:
    """A 60 x 60 m site, a point about every 0.4 m with jitter (so no two
    coordinates line up with a grid): rolling ground, a building, trees and a
    few outliers. `after` adds a 3 m mound and a 1.5 m deep pit."""
    rng = np.random.default_rng(seed)
    n = 150
    gx, gy = np.meshgrid(np.arange(n) * 0.4, np.arange(n) * 0.4)
    x = gx.ravel() + rng.uniform(-0.15, 0.15, n * n)
    y = gy.ravel() + rng.uniform(-0.15, 0.15, n * n)
    z = terrain(x, y) + rng.normal(0, 0.01, n * n)
    roof = (x > 20) & (x < 30) & (y > 20) & (y < 32)
    z[roof] += 7.0
    crown = np.hypot(x - 45, y - 15) < 4
    z[crown] += 5.0 + rng.uniform(0, 1.2, crown.sum())
    if after:
        r = np.hypot(x - 12, y - 45)
        z += np.where(r < 6, 3.0 * (1 - (r / 6) ** 2), 0.0)
        z[(np.abs(x - 45) < 4) & (np.abs(y - 45) < 3)] -= 1.5
    outliers = np.c_[rng.uniform(0, 60, 12), rng.uniform(0, 60, 12), rng.uniform(15, 30, 12)]
    return np.round(np.r_[np.c_[x, y, z], outliers], 3)


def run(exe: str, work: Path, *args: str) -> list[Path]:
    before = set(work.iterdir())
    subprocess.run(
        [exe, "-SILENT", "-NO_TIMESTAMP", "-C_EXPORT_FMT", "ASC", "-PREC", "8", "-SEP", "SEMICOLON",
         "-ADD_HEADER", *args],
        cwd=work, check=True, capture_output=True, timeout=1800,
    )
    return sorted(set(work.iterdir()) - before)


def read_asc(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", errors="replace") as f:
        header = f.readline().lstrip("/").strip().split(";")
    data = np.loadtxt(path, delimiter=";", skiprows=1, ndmin=2)
    return {name.strip(): data[:, i] for i, name in enumerate(header)}


def rows_of(points: np.ndarray, cols: dict[str, np.ndarray]) -> np.ndarray:
    distance, index = cKDTree(points).query(np.c_[cols["X"], cols["Y"], cols["Z"]])
    assert distance.max() < 1e-3, distance.max()
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cloudcompare", required=True)
    exe = parser.parse_args().cloudcompare
    OUT.mkdir(parents=True, exist_ok=True)
    before, after = survey(1, False), survey(2, True)
    np.savetxt(OUT / "before.xyz", before, fmt="%.3f")
    np.savetxt(OUT / "after.xyz", after, fmt="%.3f")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ("before.xyz", "after.xyz"):
            (work / name).write_bytes((OUT / name).read_bytes())

        # C2C: distance of every "after" point to "before", in "after" order.
        cols = read_asc(next(f for f in run(exe, work, "-O", "after.xyz", "-O", "before.xyz", "-C2C_DIST")
                             if "C2C" in f.name))
        c2c = np.full(len(after), np.nan)
        c2c[rows_of(after, cols)] = cols[next(k for k in cols if "C2C" in k)]
        np.savetxt(OUT / "c2c.txt", c2c, fmt="%.6f")

        # SOR: indices of "before" removed.
        cols = read_asc(run(exe, work, "-O", "before.xyz", "-SOR", str(SOR[0]), str(SOR[1]))[0])
        kept = np.zeros(len(before), bool)
        kept[rows_of(before, cols)] = True
        np.savetxt(OUT / "sor_removed.txt", np.flatnonzero(~kept), fmt="%d")

        # CSF: indices of "before" classified ground.
        files = run(exe, work, "-O", "before.xyz", "-CSF", "-SCENES", CSF[0], "-CLOTH_RESOLUTION", str(CSF[1]),
                    "-CLASS_THRESHOLD", str(CSF[2]), "-EXPORT_GROUND")
        cols = read_asc(next(f for f in files if "ground" in f.name.lower() and "off" not in f.name.lower()))
        np.savetxt(OUT / "csf_ground.txt", np.sort(rows_of(before, cols)), fmt="%d")

        # M3C2 at every "before" point (core points = cloud 1).
        params = work / "m3c2.txt"
        params.write_text(
            "[General]\nM3C2VER=1\n"
            f"NormalScale={M3C2[0]}\nNormalMode=0\nNormalUseCorePoints=false\nNormalPreferedOri=4\n"
            f"SearchScale={M3C2[1]}\nSearchDepth={M3C2[2]}\nSubsampleEnabled=false\n"
            "UseMedian=false\nUseMinPoints4Stat=true\nMinPoints4Stat=5\n"
            "RegistrationErrorEnabled=false\nRegistrationError=0\nPositiveSearchOnly=false\n"
            "UseSinglePass4Depth=false\nUseOriginalCloud=true\nExportStdDevInfo=false\n"
            "ExportDensityAtProjScale=false\nUsePrecisionMaps=false\nMaxThreadCount=0\n"
        )
        files = run(exe, work, "-O", "before.xyz", "-O", "after.xyz", "-M3C2", str(params), "-SAVE_CLOUDS")
        cols = read_asc(next(f for f in files if f.name == "before.asc"))
        m3c2 = np.full(len(before), np.nan)
        m3c2[rows_of(before, cols)] = cols["M3C2 distance"]
        np.savetxt(OUT / "m3c2.txt", m3c2, fmt="%.6f")

        # 2.5D volume, "before" as the ground.
        run(exe, work, "-O", "before.xyz", "-O", "after.xyz", "-VOLUME", "-GRID_STEP", str(VOLUME_STEP),
            "-GROUND_IS_FIRST")
        report = (work / "VolumeCalculationReport.txt").read_text(encoding="utf-8", errors="replace")

        def number(label: str) -> float:
            return float(re.search(label + r"[^\d]*([\d,.]+)", report).group(1).replace(",", ""))

        (OUT / "volume.txt").write_text(f"{number('Added volume')}\n{number('Removed volume')}\n")

    changelog = Path(exe).with_name("CHANGELOG.md")
    version = re.search(r"^v(\S+)", changelog.read_text(encoding="utf-8"), re.M).group(1) if changelog.is_file() else "?"
    (OUT / "README.md").write_text(
        "# CloudCompare parity fixtures\n\n"
        f"Recorded with CloudCompare {version} by `scripts/make_cloudcompare_fixtures.py`; "
        "checked by `tests/cloudcompare.rs`.\n\n"
        "- `before.xyz`, `after.xyz`: two synthetic 60 x 60 m surveys (the second adds a mound and a pit).\n"
        f"- `c2c.txt`: C2C distance of each `after` point to `before`.\n"
        f"- `sor_removed.txt`: `before` points removed by SOR (k = {SOR[0]}, {SOR[1]} σ).\n"
        f"- `csf_ground.txt`: `before` points CSF classifies as ground ({CSF[0].lower()}, "
        f"resolution {CSF[1]}, threshold {CSF[2]}).\n"
        f"- `m3c2.txt`: M3C2 distance at each `before` point (normal Ø {M3C2[0]}, projection Ø {M3C2[1]}, "
        f"depth {M3C2[2]}), `nan` where not measured.\n"
        f"- `volume.txt`: added and removed volume, `before` as the ground, {VOLUME_STEP} m grid.\n",
        encoding="utf-8",
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
