# Bounded temporary rows for CSV exports

CSV export now materializes Python float rows in batches of 65,536, while
retaining a full numeric XYZ snapshot. On three measured point clouds, peak
allocations tracked by `tracemalloc` decrease **72.54–77.95%**. All CSV bytes
match the preceding writer. Wall-clock results are mixed; this study does
**not establish a consistent export speedup**.

## Behavior and scope

The preceding writer calls `xyz.tolist()` for the whole cloud before passing
the rows to `csv.writer`. The new writer copies the coordinates into an owned
numeric array, then converts each slice to Python rows immediately before
writing it. This bounds temporary row objects without retaining all Python
float objects and row lists at once. The numeric snapshot prevents changes to
the source cloud between data writes from changing later output rows.

The canonical `x,y,z` header, Python float formatting, point order, UTF-8 text
and CRLF records retain their preceding bytes. Empty clouds still produce
only the header. Nonfinite values are still written, source attributes are
preserved, existing destinations are overwritten, parent directories are
created and write errors propagate with the file closed. No public options,
worker pools or format/precision changes are introduced.

This remains an API for a resident Open3D cloud. It retains an additional
full XYZ snapshot of 24 bytes per point, plus temporary row objects and file
buffers. Bounded row lists do not make it a bounded-memory streaming exporter.
CSV export does not consult the optional native core. Other output formats
retain their existing implementations.

## Measured inputs

The [existing input manifest](../benchmarks/cpu/io/real-source/csv-batch-inputs.json)
pins the full compared PLY clouds and little-endian f64 coordinate oracles
from the [real-source corpus](c2c-real-source.md): PandaSet 019 (531,908 points),
Autoware planning (878,920 points), and Tokyo Takanawadai (941,933 points).
No CSV input is loaded for this study; each unchanged PLY cloud is exported
with the writer's canonical header.

All coordinate bits and point order match the existing oracles. No points
are generated, perturbed or further thinned. Tokyo retains the corpus's prior
first-original-point voxel selection. These are subsets of measured clouds,
not independent surveys or native sensor CSV logs. The manifest retains the
source attribution, licensing and preceding preparation details.

## Traced allocation measurements

| Source | Whole-row peak, bytes | Batched peak, bytes | Reduction |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 85,272,087 | 23,418,063 | 72.54% |
| Autoware planning | 140,795,183 | 31,747,525 | 77.45% |
| Tokyo Takanawadai | 150,881,279 | 33,263,855 | 77.95% |

Each source/policy runs once in a fresh process, separately from all timing
trials. Imports, source loading, validation and a garbage collection precede
tracing. `tracemalloc` covers the complete `save_point_cloud` call, including
the candidate's numeric copy, Python rows, float formatting, actual writes
and file close. Every output matches its preceding complete CSV fingerprint.
The [six observations and summaries](../benchmarks/cpu/io/real-source/windows-csv-write-traced-memory.json)
retain the controller and timing-record hashes.

These peaks include Python allocations and NumPy's registered array buffers.
They exclude source-cloud storage, untraced native/OS buffers and tracer
metadata. They are **not process peak RSS or total application memory**.
Tracing adds overhead; no durations from these runs are used as speed evidence.

## Complete save durations

| Source/pass | Whole-row median, ms | Batched median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 2,886.87 | 2,757.63 | +5.63% | 3/5 |
| PandaSet final | 4,131.19 | 3,666.06 | +6.74% | 5/5 |
| Autoware initial | 4,904.65 | 5,022.73 | -5.45% | 1/5 |
| Autoware final | 8,378.01 | 8,545.98 | +5.97% | 3/5 |
| Tokyo initial | 5,085.60 | 4,633.50 | +11.65% | 5/5 |
| Tokyo final | 4,554.60 | 4,937.51 | -8.41% | 2/5 |

The clock covers the complete CSV save: path/parent handling, file open,
header, all rows, the candidate's numeric snapshot, internal cleanup, flush
and close. It does not include an `fsync` durability guarantee. Source loading,
imports, fingerprints, output checks/removal and CLI serialization are excluded.

Each source/pass starts a fresh process. An excluded full-byte parity pair,
separate first pair and one warm-up pair precede five alternating timed pairs.
Sources run PandaSet → Autoware → Tokyo, then reverse; the first policy is
balanced across all 30 timed pairs. No edits, tests, builds or scans accompany
timing. All 96 saves are fingerprinted, including the excluded observations.

Shortening is `median(100 * (1 - batched_ms / whole_rows_ms))`, rather than a
ratio of marginal medians. This explains the differing signs of Autoware's
final paired and marginal comparisons. All observations are retained, including
an Autoware initial pair of 5,593.23 → 10,034.74 ms and a Tokyo final pair of
4,859.67 → 8,092.41 ms. Scheduling, other applications, allocation/page state
and system clocks are uncontrolled. These warmed-page results span
**-8.41% to +11.65%**, with **19/30 faster pairs**. The reproducible allocation
reduction is the reason for adopting the change; the timings do not establish
a consistent speedup, a per-call guarantee, cold CLI performance, complete
format-conversion performance or GPU/parallel acceleration.

## Verification and reproduction

All 96 saves match the preceding output's full SHA-256 and size. Twelve retained
parity files also compare byte-for-byte. Each process parses its entire original
parity CSV and checks every f64 coordinate bit against the pinned oracle.

An independent audit parses all 2,352,761 points in 131,071-record batches,
checks them against both the coordinate oracles and the original Open3D PLY
reader, and compares six additional complete saves through the old/current
implementations. It verifies all 96 recorded fingerprints, all six timing
summaries, the six allocation observations, balanced first-policy counts and
source/input/library hashes. Sixteen invalid benchmark CLI cases preserve
their sentinel destinations.

Nineteen new regression cases cover one-row, partial/full/multiple batch
boundaries, exact special float text, empty exports, Unicode/uppercase paths,
replacement of existing files, source attributes, mutation between data writes
and write-failure propagation/closure. Related I/O/comparison/CLI/conversion
checks pass 231 cases with the installed native core; the related set without
the five conversion tests passes 226 with the core disabled. Type checking
passes all 166 application files.

The [timing controller](../cloudanalyzer/scripts/benchmark_csv_writer.py)
requires the source root, archived preceding `ca/io.py` and hash, existing
input manifest, dataset key and fresh work/output paths. It pins nine
application/test/controller files, the baseline, inputs/provenance, Open3D CPU
library, NumPy core binary, CSV source and parser binary. The Windows parser
is built into the Python runtime DLL. The environment is Windows, Python
3.12.10, NumPy 2.3.4 and Open3D 0.19.0, with 12 OpenMP/Rayon threads, one
OpenBLAS thread and the optional core disabled. Production does not force
these settings.

The separate [allocation controller](../cloudanalyzer/scripts/benchmark_csv_writer_memory.py)
requires a matching timing record, source root, policy and fresh work/output
paths. Run each dataset/policy in a new process after the timing study.
Duplicate/abbreviated options and existing destinations are rejected by both
controllers. Local provenance paths in a recreated manifest must resolve;
preserve the published file and coordinate hashes.

- PandaSet: [initial](../benchmarks/cpu/io/real-source/windows-pandaset019-csv-write-initial.json), [final](../benchmarks/cpu/io/real-source/windows-pandaset019-csv-write-final.json)
- Autoware: [initial](../benchmarks/cpu/io/real-source/windows-autoware-planning-csv-write-initial.json), [final](../benchmarks/cpu/io/real-source/windows-autoware-planning-csv-write-final.json)
- Tokyo: [initial](../benchmarks/cpu/io/real-source/windows-tokyo-takanawadai-csv-write-initial.json), [final](../benchmarks/cpu/io/real-source/windows-tokyo-takanawadai-csv-write-final.json)
