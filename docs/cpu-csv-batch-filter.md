# Batch finite validation for CSV point clouds

Whole-file CSV loading now validates finite coordinates in batches instead of
constructing a tiny NumPy array for each point. Six complete real-source CSV
loading comparisons are **46.61–60.15% shorter** by median paired duration,
with **30/30 faster timed pairs** and identical full XYZ outputs.

## Behavior and scope

The existing `csv.reader`/`DictReader` parsing, Python `float` conversions and
header detection remain in place. Temporary Python row lists are limited to
65,536 parsed records. Each batch becomes a numeric array and uses the existing
finite-point filter; accepted batches join in their original order before
Open3D takes an owned copy. The temporary batch arrays are released before that
final conversion.

Column names, `x_m,y_m,z_m` aliases, duplicate-column selection, quoted/multiline
fields, ignored extra columns, short headerless rows, strict numeric errors,
nonfinite filtering and the original empty-array conversion retain their
existing behavior. Malformed rows still fail even after earlier batches have
been validated. No partial cloud is returned on failure.

CSV dispatch does not consult the optional Rust reader, so this optimization
applies with or without the extension installed. Other formats and the separate
`iter_point_chunks` streaming API retain their existing implementations. This
whole-file API still retains the full accepted cloud in memory; bounded Python
row lists do not make it a bounded-memory streaming reader.

## Measured inputs

The [input manifest](../benchmarks/cpu/io/real-source/csv-batch-inputs.json) pins
CSV exports of the existing [real-source corpus](c2c-real-source.md). Each uses
all compared-subset points in their existing order. Every f64 coordinate bit
matches the preceding binary PLY geometry. There are no generated coordinates,
new perturbations or additional thinning. Tokyo retains the corpus's preceding
first-original-point selection. These are format exports of measured clouds,
rather than native sensor CSV logs or independent surveys/accuracy ground truth.
Existing source attribution and licensing are retained in the manifest.

| Source | CSV layout | Points | CSV bytes |
| --- | --- | ---: | ---: |
| PandaSet 019 | `x,y,z` header | 531,908 | 30,915,566 |
| Autoware planning | Headerless XYZ | 878,920 | 43,410,905 |
| Tokyo Takanawadai | `x_m,y_m,z_m` header | 941,933 | 34,855,677 |

CSV exports use UTF-8, CRLF records and Python's round-trip float strings.
All measured coordinates are finite. Nonfinite/malformed input behavior is
verified separately; these timings do not characterize every possible CSV
dialect or metadata layout.

## Complete loader measurements

The clock surrounds `load_point_cloud`: actual CSV file reading/parsing,
Python float conversion, finite checks, XYZ assembly, owned Open3D conversion,
empty check and internal cleanup. Returned-cloud release, coordinate hashing
and byte checks, imports, fingerprints, CSV export and CLI serialization are
outside timing. No parser or read result is replaced.

| Source/pass | Per-row median, ms | Batched median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 6,625.41 | 3,244.50 | +46.61% | 5/5 |
| PandaSet final | 7,118.39 | 3,388.92 | +51.75% | 5/5 |
| Autoware initial | 9,564.24 | 3,330.67 | +55.34% | 5/5 |
| Autoware final | 8,550.64 | 3,243.36 | +60.15% | 5/5 |
| Tokyo initial | 13,061.81 | 6,645.31 | +52.25% | 5/5 |
| Tokyo final | 11,099.20 | 5,271.33 | +54.23% | 5/5 |

Each source/pass starts a fresh process. An excluded full byte-parity pair,
separate first pair and two warm-up pairs precede five alternating timed pairs.
Sources run PandaSet → Autoware → Tokyo, then reverse. Initial passes start
per-row and final passes start batched: each policy leads 15 timed pairs.
No edits, builds, tests or scans run alongside timing. All observations remain
in the records, including the substantial variation in both policies.

Shortening is `median(100 * (1 - batched_ms / per_row_ms))`, rather than a ratio
of marginal medians. These are warmed-page observations. Scheduling, clocks,
other applications and allocator/page state are uncontrolled, and these are
not per-call guarantees. No peak RSS, cold CLI, whole comparison, registration,
GPU or streaming-throughput speedup is established by this study.

## Verification

All 108 observed loads check their full coordinate-buffer hash, shape, dtype
and empty color/normal/covariance attributes against the original geometry's
coordinate oracle. Twelve excluded parity outputs also compare every coordinate
byte directly.

An independent audit parses all 2,352,761 CSV points with `csv.reader` in
131,071-record batches and checks them against both the original Open3D PLY
reader and the full coordinate oracles. It loads all three CSVs through the
original and current readers, comparing 4,705,522 coordinate vectors byte-for-byte.
It verifies all 108 stored output fingerprints, recomputes all six summaries,
checks source/input/library hashes and verifies the balanced first-policy counts.
Seventeen parser-contract cases match original outputs or exception types/messages,
including two failures after 65,537 actual measured rows. Thirteen invalid or
conflicting benchmark CLI cases preserve their sentinel destinations.

Forty new regression cases cover the CSV grammar and filtering behaviors above,
one-row and multiple-batch boundaries, signed zero/subnormal/extreme values,
source-file preservation, owned storage, bounded temporary row batches and
unchanged optional-core dispatch. Related I/O/comparison/CLI tests pass 207
cases with the installed native core and 207 with it disabled. Type checking
passes all 166 application files.

## Reproduction and records

The [controller](../cloudanalyzer/scripts/benchmark_csv_filter.py) requires the
source root, archived preceding I/O module and its hash, input manifest, dataset
key, CSV, full coordinate oracle and fresh work/output paths. Defaults are two
warm-up pairs and five timed pairs. Duplicate/abbreviated options, invalid counts
and existing destinations are rejected. Local file paths can be supplied for
the CSV and oracle; provenance paths in a recreated manifest must also resolve.

To reproduce the format export, read the pinned compared PLY with Open3D,
write its full XYZ as little-endian f64 for the oracle, and use `csv.writer`
over those unchanged Python float values. Emit the header listed above, or
omit it for Autoware. Open the text file with `newline=""` and UTF-8 so the
writer's CRLF records remain stable. The coordinate hashes must match the
published manifest; no numeric rounding or geometric conversion is needed.

Records pin 11 application/test/controller files, the baseline and input/provenance
files, Open3D's CPU library, Python's CSV source and the CSV parser binary.
For the Windows built-in parser, the latter is the Python runtime DLL. The
environment is Windows, Python 3.12.10, NumPy 2.3.4 and Open3D 0.19.0, with
12 OpenMP/Rayon threads, one OpenBLAS thread and the optional core disabled.
Production does not force those settings or introduce a worker pool/public option.

- PandaSet: [initial](../benchmarks/cpu/io/real-source/windows-pandaset019-csv-batch-initial.json), [final](../benchmarks/cpu/io/real-source/windows-pandaset019-csv-batch-final.json)
- Autoware: [initial](../benchmarks/cpu/io/real-source/windows-autoware-planning-csv-batch-initial.json), [final](../benchmarks/cpu/io/real-source/windows-autoware-planning-csv-batch-final.json)
- Tokyo: [initial](../benchmarks/cpu/io/real-source/windows-tokyo-takanawadai-csv-batch-initial.json), [final](../benchmarks/cpu/io/real-source/windows-tokyo-takanawadai-csv-batch-final.json)
