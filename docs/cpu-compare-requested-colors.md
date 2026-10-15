# Avoiding unused colors in numeric comparisons

Comparisons now compute heatmap colors only when a snapshot is requested.
Without an image, the returned report, JSON and Markdown do not consume those
colors. Six full real-source comparisons with registration disabled are
**78.45–82.21% shorter**, with **60/60 faster timed pairs**.

## Behavior

`run_compare` retains loading, optional registration, nearest distances,
statistics, threshold handling and report generation. A truthy `snapshot_path`
still triggers the original coloring call before output writing and rendering.
`None` and an empty snapshot path omit the unused coloring work. The public
`colorize` function, renderer and registration implementations are unchanged.

The optimization applies to numeric-only comparisons with or without
registration. The measured speedups below apply specifically to registration
disabled. They do not establish a speedup for default GICP comparisons or
snapshot-producing calls. The existing
[binary XYZ PLY loader](cpu-ply-xyz-loading.md) is active in both policies.

## Real-source comparison

The [real-source corpus](c2c-real-source.md) retains measured XYZ coordinates
and its existing licensing/attribution. Each pair uses the full reference and
compared PLY exports, without generated geometry, perturbation or additional
thinning. These are subsets of one measured cloud per source, not independent
surveys or map-accuracy ground truth. Tokyo retains its preceding original-point
selection.

The clock measures the complete `run_compare` function with `method=None`,
threshold 0.5, no snapshot and no report-file destination. Actual file loading,
native NN construction/search, statistics, returned-report construction and
cleanup run normally. No solver result or distance array is replaced. The
control also computes unused colors; the candidate omits that call.

JSON/Markdown file parity, tracing/capture writes, imports/hashing and CLI
serialization are outside timing summaries. Logging is disabled for both
policies. This measures a complete unregistered comparison function, not a
cold CLI process, registered comparison or renderer.

| Source/pass | Always-color median, ms | Requested-color median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 1,327.82 | 277.11 | +80.58% | 10/10 |
| PandaSet final | 1,042.07 | 210.56 | +78.45% | 10/10 |
| Autoware initial | 2,967.45 | 535.27 | +81.27% | 10/10 |
| Autoware final | 2,090.44 | 395.85 | +81.03% | 10/10 |
| Tokyo initial | 2,647.35 | 489.66 | +82.21% | 10/10 |
| Tokyo final | 2,571.54 | 500.53 | +80.07% | 10/10 |

Each source/pass starts a fresh process. An excluded traced parity pair,
separate file-output pair and first-use pair precede two warm-up pairs and
ten alternating timed pairs, starting with the always-color control. Sources
run PandaSet → Autoware → Tokyo and then reverse. No edits, builds, tests or
scans run during timing. Every observation is retained.

Shortening is `median(100 * (1 - requested_ms / always_ms))`, not a ratio of
marginal medians. Tokyo final pair 10 takes 4,960.1558 ms versus 1,183.6305 ms
and remains in the results without an assigned cause. Scheduling, clocks,
other applications and allocator/page state are uncontrolled. These are
warmed-page observations, not a universal guarantee. No peak RSS, independent
drive-frame, full registered pipeline or GPU performance claim is made.

## Output verification

The excluded parity calls check 12 complete f64 distance arrays against the
preceding distance oracle. Six baseline color buffers are retained for audit;
the candidate calls coloring zero times. JSON and Markdown outputs match
byte-for-byte in six additional file-output pairs. All 180 reports produced
during the benchmark, including timed calls and file-output checks, match
the expected numeric result.

An independent audit reads coordinates with the original Open3D reader,
compares 12 complete coordinate arrays with preceding snapshots, and computes
six full native distance arrays. It verifies all 12 captured distance arrays,
six full RGB arrays containing 4,705,522 color vectors, 36 stored reports and
24 report files. It also rebuilds six production reports while rejecting any
unexpected coloring call, verifies source/input/library fingerprints, and
checks 13 malformed/conflicting CLI input cases without overwriting sentinels.

Thirteen new regression cases cover numeric-only comparisons with no
registration, ICP and GICP; optional report files; empty snapshot paths;
snapshot colors and point fields; constant-distance inputs; CLI outputs; and
four byte-identical PNG comparisons against the original color/render pipeline.
Their JSON/Markdown files also match the corresponding numeric-only calls.
Related Python/CLI tests total 130 passes. With the preceding Rust extension,
40 related Python tests pass and one existing feature check skips.

For the point-only measured inputs, skipping coloring avoids constructing
an otherwise unused owned f64 RGB vector of 12,765,792 bytes for PandaSet,
21,094,080 for Autoware or 22,606,392 for Tokyo. Normalization/colormap scratch,
cloud geometry, reference trees, existing input attributes and allocator
overhead are separate; these payload sizes are not RSS savings.

## Reproduction and records

The [controller](../cloudanalyzer/scripts/benchmark_compare_color.py) requires
the original reference/compared clouds and manifest, preceding compare module
and hash, native build index, preceding distance record, source root and new
work/output paths. Defaults are two warm-up pairs and ten timed pairs. Existing
destinations and duplicate/conflicting options are rejected.

Records pin 16 Python/source fingerprints, 31 unchanged native Rust
fingerprints, the installed extension and Open3D CPU library. Timings use
Windows Python 3.12.10, NumPy 2.3.4, Open3D 0.19.0, 12 OpenMP/Rayon threads
and one OpenBLAS thread. Production does not impose these thread settings.
No public option, GPU dispatch or extra worker pool is introduced.

- PandaSet: [initial](../benchmarks/cpu/compare/real-source/windows-pandaset019-requested-colors-initial.json), [final](../benchmarks/cpu/compare/real-source/windows-pandaset019-requested-colors-final.json)
- Autoware: [initial](../benchmarks/cpu/compare/real-source/windows-autoware-planning-requested-colors-initial.json), [final](../benchmarks/cpu/compare/real-source/windows-autoware-planning-requested-colors-final.json)
- Tokyo: [initial](../benchmarks/cpu/compare/real-source/windows-tokyo-takanawadai-requested-colors-initial.json), [final](../benchmarks/cpu/compare/real-source/windows-tokyo-takanawadai-requested-colors-final.json)
