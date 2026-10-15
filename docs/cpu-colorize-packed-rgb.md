# Packing heatmap RGB rows before Open3D conversion

Heatmap coloring now packs Matplotlib's RGB slice into contiguous rows before
constructing Open3D's owned color vector. Six real-source comparisons that
produce PNG snapshots are **20.16–35.84% shorter** by median paired duration,
with **29/30 faster timed pairs**. Full distance/color buffers, numeric reports
and PNG bytes match the preceding implementation.

## Change and scope

Matplotlib returns an RGBA array. Selecting its first three columns leaves a
four-double row stride, although Open3D receives three components per point.
`np.ascontiguousarray` packs those rows before `Vector3dVector` conversion.
Normalization, colormap selection and RGB values retain the original behavior.
The input cloud is still modified in place; points, normals and covariances
are preserved. Empty distances still return immediately.

This applies to all callers of `colorize`, including registered comparisons.
The measured interval below is specifically the complete unregistered
`run_compare` function with a snapshot. It does not establish default GICP,
interactive Open3D renderer or GPU performance. Numeric-only comparisons
already [skip unused coloring](cpu-compare-requested-colors.md).

## Real-source measurements

The existing [real-source corpus](c2c-real-source.md) retains its measured
coordinates, preparation and licensing. Each reference/compared pair is made
from one measured source cloud, rather than independent surveys or accuracy
ground truth. There is no generated geometry, added perturbation or additional
input thinning; Tokyo retains its preceding original-point selection.

Both policies run the current `run_compare(method=None, threshold=.5)` with
a PNG destination. Actual loading, native NN construction/search, statistics,
threshold/report construction, normalization, colormap, owned Open3D colors,
Matplotlib projection, PNG encoding/write and function cleanup are timed.
Only the coloring function changes. No solver or distance result is replaced.
Distances and coloring use all compared points: 531,908 for PandaSet,
878,920 for Autoware and 941,933 for Tokyo.

The renderer keeps its existing deterministic 200,000-point display cap and
1920×1080 output. Directory creation, imports/hashing, validation/capture writes,
CLI startup/serialization and extra JSON/Markdown file writes are outside
the timing summaries. Both policies have logging disabled.

| Source/pass | Strided median, ms | Packed median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 2,285.20 | 1,715.05 | +29.81% | 5/5 |
| PandaSet final | 2,097.73 | 1,565.91 | +24.53% | 5/5 |
| Autoware initial | 4,758.71 | 4,697.19 | +20.16% | 5/5 |
| Autoware final | 3,042.83 | 2,122.68 | +31.15% | 4/5 |
| Tokyo initial | 3,519.00 | 2,183.00 | +35.84% | 5/5 |
| Tokyo final | 2,635.06 | 1,735.31 | +34.62% | 5/5 |

Each source/pass starts a fresh process. An excluded traced parity pair,
including JSON/Markdown output, precedes a separate untraced pair, two warm-up
pairs and five alternating timed pairs, starting packed. Sources run PandaSet
→ Autoware → Tokyo, then reverse. No edits, builds, tests or scans run during
timing, and all observations are retained.

Shortening is `median(100 * (1 - packed_ms / strided_ms))`, rather than a ratio
of marginal medians. Autoware initial illustrates why these differ: both
policies vary substantially during the pass. Autoware final pair 5 takes
3,652.9638 ms packed versus 2,822.6593 ms strided and remains in the results
without an assigned cause. Scheduling, clocks, other applications and
allocator/page state are uncontrolled. These are warmed-page observations,
not a per-call guarantee. No peak RSS, cold-process, registered-pipeline,
independent-drive-frame or GPU speed claim is made.

## Verification

Every one of the 108 benchmark calls checks its complete report and PNG bytes.
Excluded traced calls also check 12 complete f64 distance arrays and 12 complete
RGB arrays against the preceding real-source records. The 24 additional
JSON/Markdown files match byte-for-byte between policies.

An independent audit uses the original Open3D reader, checks 12 full coordinate
arrays against the preceding snapshots, runs six actual native NN queries,
and rebuilds the original normalization/colormap and reports. It verifies all
24 captured arrays, all 108 reports and all 108 PNGs, including six separately
rendered images using the archived original visualization module. It also
checks source/input/library fingerprints and nine invalid/conflicting CLI
cases without overwriting sentinel files or directories.

Twenty-nine new regression cases cover exact RGB bytes across float32,
float64 and integer distances; contiguous, strided and reversed input; two
colormaps; constant and near-constant ranges; nonfinite/extreme values; custom
bad colors; empty inputs; ownership; point attributes; and two byte-identical
PNG comparisons against the original mapping. Related Python/CLI tests pass
159 cases, with 49 related cases also passing when the optional Rust core is
disabled. Type checking passes all 166 application files.

## Reproduction and records

The [controller](../cloudanalyzer/scripts/benchmark_snapshot_colors.py) requires
the source root, original reference/compared clouds and manifest, preceding
color record, archived visualization module and hash, native build index,
matching archived native source root, and fresh work/output paths. Duplicate
options, invalid counts and existing destinations are rejected. Defaults are
two warm-up pairs and five timed pairs.

Records pin 86 current Python/source files, the baseline module, original data,
coordinate snapshot, preceding records, installed extension, installed wrapper
and Open3D CPU library. The 31 native source fingerprints are checked against
the archive for the installed binary; they are not represented as the later
checkout's native sources. No extension is rebuilt during this study.

The environment is Windows, Python 3.12.10, NumPy 2.3.4, Open3D 0.19.0 and
Matplotlib 3.10.8, with 12 OpenMP/Rayon threads and one OpenBLAS thread. Production
does not force these thread settings. No new worker pool or public option is
introduced.

- PandaSet: [initial](../benchmarks/cpu/visualization/real-source/windows-pandaset019-packed-rgb-initial.json), [final](../benchmarks/cpu/visualization/real-source/windows-pandaset019-packed-rgb-final.json)
- Autoware: [initial](../benchmarks/cpu/visualization/real-source/windows-autoware-planning-packed-rgb-initial.json), [final](../benchmarks/cpu/visualization/real-source/windows-autoware-planning-packed-rgb-final.json)
- Tokyo: [initial](../benchmarks/cpu/visualization/real-source/windows-tokyo-takanawadai-packed-rgb-initial.json), [final](../benchmarks/cpu/visualization/real-source/windows-tokyo-takanawadai-packed-rgb-final.json)
