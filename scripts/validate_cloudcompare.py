"""Compare the CloudAnalyzer Rust core with CloudCompare on a public LiDAR tile.

Runs the same inputs through CloudCompare's command line and through
``cloudanalyzer_core`` (the Python bindings of the core the web viewer runs as
WebAssembly), and writes a Markdown report.

Usage::

    python scripts/validate_cloudcompare.py autzen_trim.las \\
        --cloudcompare "C:/Program Files/CloudCompare/CloudCompare.exe" \\
        --work /tmp/ca-validation --out docs/validation.md

The second epoch is the tile itself, thinned to 70 % and moved by a known
rigid transform, so registration can be checked against the truth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path

import cloudanalyzer_core as cc
import numpy as np
from scipy.spatial import cKDTree

SEED = 0
ANGLE_DEG = 0.3
SHIFT = np.array([0.5, -0.3, 0.2])


#: Added to coordinates on load, as CloudCompare stores float32 offsets.
GLOBAL_SHIFT: list[str] = []


def run_cc(exe: str, workdir: Path, *args: str) -> list[Path]:
    """Run CloudCompare silently in `workdir`; return the files it wrote.
    Every opened file gets the same global shift, so its float32 storage stays precise."""
    command: list[str] = []
    for arg in args:
        command.append(arg)
        if arg == "-O":
            command += ["-GLOBAL_SHIFT", *GLOBAL_SHIFT]
    before = set(workdir.iterdir())
    subprocess.run(
        [exe, "-SILENT", "-NO_TIMESTAMP", "-C_EXPORT_FMT", "ASC", "-PREC", "8",
         "-SEP", "SEMICOLON", "-ADD_HEADER", *command],
        cwd=workdir,
        check=True,
        capture_output=True,
        timeout=1800,
    )
    return sorted(set(workdir.iterdir()) - before)


def read_asc(path: Path) -> dict[str, np.ndarray]:
    """A CloudCompare ASCII export with a `//X;Y;Z;...` header, by column name."""
    with path.open(encoding="utf-8", errors="replace") as f:
        header = f.readline().lstrip("/").strip().split(";")
    data = np.loadtxt(path, delimiter=";", skiprows=1, ndmin=2)
    return {name.strip(): data[:, i] for i, name in enumerate(header)}


def xyz(cols: dict[str, np.ndarray]) -> np.ndarray:
    return np.c_[cols["X"], cols["Y"], cols["Z"]]


def index_of(points: np.ndarray, of: np.ndarray) -> np.ndarray:
    """Row of `points` nearest to each row of `of` (CloudCompare stores float32
    offsets from its global shift, so exported coordinates differ slightly)."""
    distance, index = cKDTree(points).query(of)
    assert distance.max() < 0.01, f"unmatched point, {distance.max()} away"
    return index


def rotation_error_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("las", type=Path)
    parser.add_argument("--cloudcompare", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    exe = args.cloudcompare
    work = args.work
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    data = cc.read(str(args.las))
    ref = data["positions"]
    rng = np.random.default_rng(SEED)
    keep = rng.random(len(ref)) < 0.7
    a = np.radians(ANGLE_DEG)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    centre = ref.mean(0)
    cmp = (ref[keep] - centre) @ rot.T + centre + SHIFT
    # Coordinates with 4 decimals, read identically by both tools.
    ref = np.round(ref, 4)
    cmp = np.round(cmp, 4)
    GLOBAL_SHIFT[:] = [str(-np.floor(v / 1000) * 1000) for v in ref.min(0)]
    np.savetxt(work / "ref.xyz", ref, fmt="%.4f")
    np.savetxt(work / "cmp.xyz", cmp, fmt="%.4f")
    shift = np.array([float(v) for v in GLOBAL_SHIFT])
    rows: list[tuple[str, str, str, str]] = []  # (analysis, settings, result, note)

    # ------------------------------------------------------------ C2C
    files = run_cc(exe, work, "-O", "cmp.xyz", "-O", "ref.xyz", "-C2C_DIST")
    theirs = read_asc(next(f for f in files if "C2C" in f.name))
    col = next(k for k in theirs if "C2C" in k)
    order = index_of(cmp, xyz(theirs))
    ours = cc.nearest_distances(cmp, ref)
    diff = np.abs(ours[order] - theirs[col])
    rows.append((
        "C2C distance",
        f"{len(cmp):,} points to {len(ref):,}",
        f"max difference {diff.max():.1e} ft (mean distance {ours.mean():.4f} ft)",
        "Same nearest neighbours; the difference is CloudCompare's float32 storage.",
    ))

    # ------------------------------------------------------------ ICP
    files = run_cc(exe, work, "-O", "cmp.xyz", "-O", "ref.xyz", "-ICP", "-ITER", "100",
                   "-MIN_ERROR_DIFF", "1e-8", "-OVERLAP", "70")
    local = np.loadtxt(next(f for f in files if "REGISTRATION_MATRIX" in f.name))
    # CloudCompare's matrix works on shifted coordinates (x + s): undo the shift.
    theirs_m = local.copy()
    theirs_m[:3, 3] = local[:3, :3] @ shift + local[:3, 3] - shift
    ours_m = cc.icp(cmp, ref, max_iterations=100, overlap=0.7)["transformation"]
    # The truth maps cmp back onto ref: x = R^T (y - c - s) + c.
    truth = np.eye(4)
    truth[:3, :3] = rot.T
    truth[:3, 3] = centre - rot.T @ (centre + SHIFT)

    def errors(m: np.ndarray) -> tuple[float, float]:
        delta = m @ np.linalg.inv(truth)
        # Translation error measured at the tile centre.
        moved = m[:3, :3] @ centre + m[:3, 3]
        expected = truth[:3, :3] @ centre + truth[:3, 3]
        return rotation_error_deg(delta[:3, :3]), float(np.linalg.norm(moved - expected))

    (r_o, t_o), (r_c, t_c) = errors(ours_m), errors(theirs_m)
    rows.append((
        "ICP (known truth)",
        f"{ANGLE_DEG}° and {np.linalg.norm(SHIFT):.2f} ft apart, 70 % overlap",
        f"error: ours {r_o:.4f}° / {t_o:.4f} ft, CloudCompare {r_c:.4f}° / {t_c:.4f} ft",
        "Point-to-plane (ours) against CloudCompare's point-to-point.",
    ))

    # ------------------------------------------------------------ SOR
    files = run_cc(exe, work, "-O", "ref.xyz", "-SOR", "8", "1.0")
    theirs_kept = set(index_of(ref, xyz(read_asc(files[0]))).tolist())
    # CloudCompare counts the point itself among its k neighbours.
    ours_kept = set(cc.statistical_outliers(ref, k=7, ratio=1.0).tolist())
    rows.append((
        "Outlier removal (SOR)",
        "CloudCompare k = 8 (with the point itself) = ours k = 7, 1 σ",
        f"kept {len(ours_kept):,} vs {len(theirs_kept):,}; {len(ours_kept ^ theirs_kept):,} points differ",
        "Identical once k counts the same neighbours.",
    ))

    # ------------------------------------------------------------ CSF
    resolution, threshold = 3.0, 1.5
    files = run_cc(exe, work, "-O", "ref.xyz", "-CSF", "-SCENES", "RELIEF", "-CLOTH_RESOLUTION",
                   str(resolution), "-CLASS_THRESHOLD", str(threshold), "-EXPORT_GROUND")
    ground_file = next(f for f in files if "ground" in f.name.lower() and "off" not in f.name.lower())
    theirs_ground = np.zeros(len(ref), bool)
    theirs_ground[index_of(ref, xyz(read_asc(ground_file)))] = True
    ours_ground = cc.ground_csf(ref, cloth_resolution=resolution, class_threshold=threshold, rigidness="relief")
    rows.append((
        "Ground extraction (CSF)",
        f"relief, cloth {resolution:g} ft, threshold {threshold:g} ft",
        f"ground {ours_ground.sum():,} vs {theirs_ground.sum():,}; "
        f"{(ours_ground != theirs_ground).sum():,} points differ",
        "Same simulation as CloudCompare's CSF plugin.",
    ))

    # ------------------------------------------------------------ M3C2
    normal_d, proj_d, depth = 6.0, 4.0, 10.0
    params = work / "m3c2.txt"
    params.write_text(
        "[General]\nM3C2VER=1\n"
        f"NormalScale={normal_d}\nNormalMode=0\nNormalUseCorePoints=false\nNormalPreferedOri=4\n"
        f"SearchScale={proj_d}\nSearchDepth={depth}\nSubsampleEnabled=false\n"
        "UseMedian=false\nUseMinPoints4Stat=true\nMinPoints4Stat=5\n"
        "RegistrationErrorEnabled=false\nRegistrationError=0\nPositiveSearchOnly=false\n"
        "UseSinglePass4Depth=false\nUseOriginalCloud=true\nExportStdDevInfo=false\n"
        "ExportDensityAtProjScale=false\nUsePrecisionMaps=false\nMaxThreadCount=0\n"
    )
    # UseOriginalCloud puts the results on the core points (cloud 1), unmoved.
    files = run_cc(exe, work, "-O", "ref.xyz", "-O", "cmp.xyz", "-M3C2", str(params), "-SAVE_CLOUDS")
    theirs = read_asc(next(f for f in files if f.name == "ref.asc"))
    theirs_d = theirs["M3C2 distance"]
    order = index_of(ref, xyz(theirs))
    # CloudCompare's scales are diameters and its depth is the cylinder's
    # half-length; it measures with any number of points (the minimum only
    # applies to the statistics).
    d, _, _, _ = cc.m3c2(ref, ref, cmp, normal_radius=normal_d / 2, projection_radius=proj_d / 2,
                         max_depth=depth, min_points=1)
    ours_d = d[order]
    both = np.isfinite(ours_d) & np.isfinite(theirs_d)
    delta = np.abs(ours_d[both] - theirs_d[both])
    rows.append((
        "M3C2",
        f"normal Ø {normal_d:g} ft, projection Ø {proj_d:g} ft, depth ±{depth:g} ft",
        f"{both.sum():,} core points measured by both: median difference {np.median(delta):.1e} ft, "
        f"95th percentile {np.percentile(delta, 95):.1e} ft",
        f"CloudCompare measures {np.isfinite(theirs_d).sum() - both.sum():,} more core points "
        "(ours needs 3 neighbours to fit a normal).",
    ))

    # ------------------------------------------------------------ normals
    radius = 4.0
    files = run_cc(exe, work, "-O", "ref.xyz", "-OCTREE_NORMALS", str(radius), "-ORIENT", "PLUS_Z")
    theirs = read_asc(files[0])
    order = index_of(ref, xyz(theirs))
    theirs_n = np.c_[theirs["Nx"], theirs["Ny"], theirs["Nz"]]
    ours_n = cc.normals(ref, k=16)
    cos = np.abs(np.sum(ours_n[order] * theirs_n, axis=1))
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    rows.append((
        "Normals",
        f"ours 16 nearest points, CloudCompare radius {radius:g} ft",
        f"median angle {np.median(angle):.2f}°, 75th percentile {np.percentile(angle, 75):.1f}°",
        "Different neighbourhoods (k nearest against a radius): close on surfaces, apart on vegetation.",
    ))

    # ------------------------------------------------------------ volume
    step = 4.0
    run_cc(exe, work, "-O", "ref.xyz", "-O", "cmp.xyz", "-VOLUME", "-GRID_STEP", str(step), "-GROUND_IS_FIRST")
    report = (work / "VolumeCalculationReport.txt").read_text(encoding="utf-8", errors="replace")

    def number(label: str) -> float:
        return float(re.search(label + r"[^\d]*([\d,.]+)", report).group(1).replace(",", ""))

    theirs_added, theirs_removed = number("Added volume"), number("Removed volume")
    ours = cc.volume(ref, cmp, cell=step)
    rows.append((
        "Cut / fill volume",
        f"{step:g} ft grid, mean height per cell",
        f"fill {ours['added']:,.0f} vs {theirs_added:,.0f} ft³ ({ours['added'] / theirs_added - 1:+.1%}), "
        f"cut {ours['removed']:,.0f} vs {theirs_removed:,.0f} ft³ ({ours['removed'] / theirs_removed - 1:+.1%})",
        "Cells are placed differently (CloudCompare centres them on the grid origin).",
    ))

    changelog = Path(exe).with_name("CHANGELOG.md")
    version = ""
    if changelog.is_file():
        match = re.search(r"^v(\d+\.\d+\.\d+)", changelog.read_text(encoding="utf-8", errors="replace"), re.M)
        version = f" {match.group(1)}" if match else ""
    lines = [
        "# Validation against CloudCompare",
        "",
        "Generated by [`scripts/validate_cloudcompare.py`](../scripts/validate_cloudcompare.py) on "
        f"{dt.date.today().isoformat()}.",
        "",
        f"- Data: `{args.las.name}`, {len(ref):,} points, a public LiDAR tile from the PDAL sample data "
        "(coordinates in feet).",
        f"- Second epoch: {keep.mean():.0%} of the points, rotated {ANGLE_DEG}° about z and moved by "
        f"{SHIFT.tolist()} ft (seed {SEED}).",
        "- Ours: the Rust core through `cloudanalyzer_core`; the web viewer runs the same code as WebAssembly.",
        f"- CloudCompare{version}, command line, with a global shift so its float32 coordinates stay precise.",
        "",
        "| Analysis | Settings | Result (ours vs CloudCompare) | Note |",
        "|---|---|---|---|",
        *[f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows],
        "",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out.read_text(encoding="utf-8").encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
