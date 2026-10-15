# Distance reuse in scan-match diagnostics

`ca scan-match-debug --artifact-dir ...` now uses the same before/after distance
arrays for statistics, threshold checks and colored PLY output. It performs
two nearest-neighbour searches instead of four. The measured diagnostic,
coloring and artifact-writing interval shortens by 9.75–17.27% across six
real-source comparisons, with 54/60 faster timed pairs. All tested distances,
report fields and complete PLY bytes agree with the preceding workflow.

The change applies automatically to ICP and GICP diagnostics, including those
called by `ca slam-debug`. Public options and result fields stay unchanged.
The existing Rust/Rayon nearest-neighbour path is used when installed; the
Open3D fallback also reuses results. Registration, distance arithmetic, point
order, thread dispatch, threshold rules, colormap and PLY writer are unchanged.
This is CPU work elimination using already computed parallel-search results.

This report records the distance-array reuse stage. The subsequent
[scoped reference reuse](cpu-scan-match-reference.md) shares one tree across
the before/after searches and measures its preparation/cleanup separately.

## Lifetime and behavior

Previously the workflow calculated before/after distances for summaries,
discarded those arrays, and repeated both searches for artifact colors.
The new workflow keeps each result until its corresponding colored cloud is
written. After the first artifact is saved, its distance array is released;
the after array is released after the second artifact. Results stay local to
one invocation; there is no cross-job cache or retained search tree.

Without an artifact directory, each array is released immediately after its
summary/threshold block, so the before array is not kept during registration.
Registration failure still creates no artifact directory. Input clouds are
copied before coloring and original input files remain unchanged.

| Measured source | Query points | Additional before-distance payload held during registration, bytes |
| --- | ---: | ---: |
| PandaSet 019 | 531,908 | 4,255,264 |
| Autoware planning | 878,920 | 7,031,360 |
| Tokyo Takanawadai | 941,933 | 7,535,464 |

These are `8 * query_count` payload bytes, not process RSS or a peak-memory
bound. With artifacts enabled, both before/after arrays are alive before the
first coloring step; the before array is the additional live result compared
with the preceding one-result-at-a-time artifact stage. Caller clouds, normals,
colored copies, native search scratch, allocator overhead and file buffers are
separate. Avoiding searches also avoids their transient allocations; this
measurement does not quantify a net RSS reduction.

## Measured interval

[Source preparation](c2c-real-source.md) provides full original f64 XYZ and
source-index subsets from PandaSet, Autoware and Tokyo. No new thinning,
generated points, artificial displacement or noise is added. Tokyo retains
its preceding original-point selection. These subsets come from one measured
source, not independent surveys or accuracy ground truth. Small generated
fixtures are used only for correctness tests.

Each comparison loads the full reference/query PLY, performs one actual
full-source Open3D ICP registration, and saves its complete initial/aligned
points, normals and transformation. Both production workflow versions receive
that same registration result. Input loading, normal estimation and ICP solving
are outside measured clocks. The benchmark replaces the loader and registration
entry points within its process to supply these fixed inputs/results; production
code uses the usual loader and solver. No voxel reduction or crop is used in
this measurement. Threshold checks use 0.5 m.

The timed interval calls the production `run_scan_match_debug` and includes
its preprocessing/cloud copies, frozen-result/normal assignment, nearest-neighbour
searches, NumPy summaries/threshold counts, color mapping and all three binary
PLY writes. Output hashing/checks are outside clocks. This is a controlled
diagnostic/artifact interval, not a full CLI or registration speed measurement.
Full command benefit depends on the remaining loading/registration work.

Windows, 12 logical CPUs, Python 3.12.10, NumPy 2.3.4, Open3D 0.19.0 and
Matplotlib 3.10.8 use the preceding frozen release Rust extension. Its 27
source/configuration fingerprints remain unchanged; nine Python source/config
fingerprints and the archived preceding workflow are recorded separately.
`RAYON_NUM_THREADS=12` and `OMP_NUM_THREADS=12` are set before imports.
Each source/pass starts a new process. An excluded correctness pair tracing
complete distance arrays and five warm-up pairs precede ten alternating timed
pairs. Passes run PandaSet → Autoware → Tokyo, then reverse the source order.
First correctness timings retain traced arrays and are not performance claims.
No builds, tests, scans or edits run alongside measured comparisons.

| Source/pass | Previous median, ms | Reuse median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 3,456.01 | 2,624.40 | +17.27% | 10/10 |
| PandaSet final | 3,170.63 | 2,513.63 | +9.75% | 9/10 |
| Autoware initial | 5,706.73 | 4,689.43 | +12.85% | 8/10 |
| Autoware final | 4,656.58 | 4,082.16 | +10.87% | 8/10 |
| Tokyo initial | 3,994.56 | 3,843.86 | +11.16% | 10/10 |
| Tokyo final | 5,636.91 | 4,770.75 | +12.01% | 9/10 |

Shortening is `median(100 * (1 - reuse_ms / previous_ms))`, not a ratio of
marginal medians. All observations remain, including six slower reuse pairs:
final PandaSet's sixth (5,002.43 / 5,104.60 ms), initial Autoware's first and
sixth (5,423.24 / 6,020.09 and 6,699.87 / 9,272.84 ms), final Autoware's sixth
and ninth (4,665.12 / 5,109.99 and 4,272.30 / 4,491.02 ms), and final Tokyo's
eighth (4,657.67 / 4,763.91 ms). Absolute times vary between passes.
Scheduling, clocks, power, other applications, allocator/page retention and
filesystem buffering are uncontrolled. Generated PLY files are overwritten
inside each policy's fresh private work directory between trials; writes use
the production writer without a separate durable-flush benchmark. Results
are not isolated kernel speedups or guaranteed improvements for another workload.
They do not measure GPU/browser/WASM performance or a new registration algorithm.

## Validation and reproduction

Thirty-six full f64 distance arrays agree between the preceding workflow's
summary/color searches and the new shared searches. All 192 diagnostic reports
agree in registration/preprocessing fields, both sets of five distance statistics,
improvement values and threshold counts; artifact paths are normalized to their
filenames for comparison. All 576 complete PLY outputs (three files per invocation,
both policies, first/warm-up/timed calls) agree in every byte, including coordinates,
normals and quantized RGB. Independent audit recalculates full-source distances
and statistics, verifies all six real registration snapshots by applying their
recorded matrices to the original source, checks the final files against their
hashes, and recomputes every timing summary.

Forty-six targeted tests pass across scan-match diagnostics, metrics, visualization,
`slam-debug` and the CLI invocation. New regressions cover ICP/GICP, Rust/Open3D,
artifacts on/off, optional thresholds, changed aligned coordinates, two search
calls, complete color/coordinate parity, distance-array lifetime and failure
without publishing artifacts. Existing preprocessing and related workflow tests
remain in place. Syntax and whitespace checks pass. The Rust kernel is unchanged.

The [benchmark script](../cloudanalyzer/scripts/benchmark_scan_match_distance_reuse.py)
requires `--source-root`, `--reference`, `--compared`, `--input-manifest`, an archived
`--baseline` and `--baseline-sha256`, plus new `--work-dir` and `--output` paths.
Defaults are `--threads 12 --repeats 10`. Use the preceding
`cloudanalyzer/ca/scan_match_debug.py` as baseline in an environment with the frozen
release extension and recorded dependencies. The script fingerprints original
source data, inputs, implementation and extension before/after measurement.
Unknown/duplicate options, invalid counts, mismatched baseline/inputs and existing
output/work paths fail without replacing existing files. Twelve negative controls
verify these cases. Raw timings and metadata are preserved:

- PandaSet: [initial](../benchmarks/cpu/scan-match/real-source/windows-pandaset019-distance-reuse-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-pandaset019-distance-reuse-final.json)
- Autoware: [initial](../benchmarks/cpu/scan-match/real-source/windows-autoware-planning-distance-reuse-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-autoware-planning-distance-reuse-final.json)
- Tokyo: [initial](../benchmarks/cpu/scan-match/real-source/windows-tokyo-takanawadai-distance-reuse-initial.json), [final](../benchmarks/cpu/scan-match/real-source/windows-tokyo-takanawadai-distance-reuse-final.json)
