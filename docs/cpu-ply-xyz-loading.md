# Bulk loading of binary XYZ PLY files

Point-only binary PLY files with a verified layout now use a bounded bulk
read. Across six real-source comparisons, the loading interval is
**78.01–83.25% shorter**, and the measured diagnostic interval is
**23.08–31.70% shorter**. Both comparisons improve in **30/30 timed pairs**.
Actual ICP iterations are excluded from the diagnostic interval.

## Eligible files and fallback

The fast path accepts only a little-endian binary PLY 1.0 file containing one
positive-count vertex element, with exactly three `double` properties in
`x`, `y`, `z` order. The file payload must be exactly `24 * vertex_count`
bytes, no larger than 64 MiB. Comments and LF/CRLF line endings are supported.
Header inspection is bounded to 128 lines, 4,096 bytes per line and 64 KiB
overall. Size is checked on the opened file descriptor before bulk allocation.

The payload is read with an explicit little-endian f64 dtype using
[NumPy `fromfile`](https://numpy.org/doc/2.3/reference/generated/numpy.fromfile.html),
then copied into an owned Open3D point vector. Coordinates are not filtered,
scaled or reordered. A short read or I/O error defers to the existing reader.
There is no cache across calls and no public CLI option.

ASCII, float32, big-endian and larger files, extra/reordered properties,
colors, normals, faces and other elements continue through the preceding
Open3D loader. Unrecognized, oversized and malformed headers also defer.
This preserves existing attribute handling and empty/error behavior. The
fast path changes neither PCD/LAS/LAZ/CSV readers nor the finite/bounds filtering
performed by streaming consumers.

Open3D 0.19's [PLY reader](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/io/file_format/FilePLY.cpp)
assigns decoded scalar values through property callbacks. For the verified
plain XYZ layout, the bulk path reads the same f64 values together. Complex
layouts retain those existing callbacks and attribute conversions.

## Workloads and timing

An exploratory instrumented diagnostic profile found loading to account for
roughly 30% of the tested interval. That profile was used to select this
candidate; it is not a speed comparison.

The [real-source corpus](c2c-real-source.md) provides attributed/licensed,
unchanged measured coordinates in point-only PLY exports. No generated
geometry, perturbation or additional thinning is used. Each diagnostic run
repeats the same full scan/map three times, using a CSV dispatch fixture.
These are not independent driving frames, recorded SLAM telemetry or map
accuracy ground truth. Tokyo retains its preceding original-point selection.

The control is the preceding I/O module. Both policies retain the
[unused-normal optimization](cpu-scan-icp-normals.md), scoped map/reference
reuse and all independent working-cloud copies; the
[rejected copy-borrowing candidate](cpu-scan-preprocess-borrow.md) remains off.

The diagnostic clock includes actual file loading, cloud copies,
preprocessing, native NN preparation/search, statistics, ranking/report
assembly, transformed-result construction and cleanup. A saved actual
full-source ICP result is supplied at the native solver boundary after
preparation. ICP iterations, artifact/PLY output, parity tracing/hashing/writes
and CLI serialization are excluded.

| Source/pass | Open3D diagnostic median, ms | Bulk diagnostic median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 940.43 | 730.61 | +26.21% | 5/5 |
| PandaSet final | 936.97 | 707.59 | +24.44% | 5/5 |
| Autoware initial | 1,506.54 | 1,158.84 | +23.08% | 5/5 |
| Autoware final | 1,656.93 | 1,121.06 | +31.70% | 5/5 |
| Tokyo initial | 1,635.37 | 1,229.75 | +25.23% | 5/5 |
| Tokyo final | 1,586.90 | 1,192.29 | +23.67% | 5/5 |

The separately measured loading interval sequentially loads and releases the
same four inputs: scan, map, scan, scan. It includes file handling, decoding,
owned cloud construction and cleanup, with no diagnostic or solver work.

| Source/pass | Open3D loading median, ms | Bulk loading median, ms | Median paired shortening |
| --- | ---: | ---: | ---: |
| PandaSet initial | 318.33 | 64.53 | +79.49% |
| PandaSet final | 254.61 | 55.67 | +78.01% |
| Autoware initial | 513.77 | 95.13 | +83.25% |
| Autoware final | 510.48 | 93.35 | +80.82% |
| Tokyo initial | 496.30 | 98.38 | +80.78% |
| Tokyo final | 550.07 | 94.51 | +82.82% |

Each source/pass starts a fresh process. Each interval has an excluded parity
pair and first-use pair, one warm-up pair and five alternating timed pairs,
starting with bulk loading. Diagnostic trials precede loading-only trials.
Sources run PandaSet → Autoware → Tokyo and then reverse. No edits, builds,
tests or scans run during timing. Every observation remains.

Shortening is `median(100 * (1 - bulk_ms / open3d_ms))`, not the ratio of
marginal medians. Scheduling, clocks, other applications and allocator/page
state are uncontrolled. Five pairs per case provide limited evidence, not a
universal guarantee. The two intervals overlap and their gains cannot be
added. These warmed-page measurements do not establish cold-ingestion,
full iterative ICP/SLAM, other file-format or GPU performance gains.

## Correctness and memory

The benchmark checks 48 complete loaded-cloud states, another 48 loading-only
states and 36 complete prepared solver states. Point counts and XYZ bytes
match, with absent attributes verified. All 72 captured full f64 distance
arrays match the preceding oracle. Ninety-six complete run reports, containing
288 frame reports, are compared during the benchmark, including timed calls.

The independent audit compares raw binary payloads, the original Open3D
reader, the new reader and preceding coordinate snapshots for 12 full input
array checks. It independently computes 12 full native distance arrays,
checks all 72 captured arrays and statistics/threshold blocks, verifies 24
stored reports and rebuilds six production reports. It also verifies source,
input and library hashes, released references and 16 negative CLI controls.

Thirty-four new tests cover bit equality for finite values, signed zeros,
subnormals, infinities and NaN payloads, owned storage, Unicode paths, line
endings, provided colors/normals, alternate valid formats, header/payload
bounds, short reads and fallback/error behavior. Related I/O, diagnostic,
streaming, evaluation, batch, loop and CLI tests total 434 passes. The main
selection also yields 257 passes and 11 existing feature skips with the
preceding Rust extension.

The NumPy payload array is temporary; the Open3D cloud owns its coordinates.
For the measured source files the temporary payload is 12,765,792,
21,094,104 or 22,606,392 bytes, depending on source. The absolute fast-path
limit is 64 MiB of temporary payload, separate from the owned cloud, reference
tree, remaining cloud copies and allocator overhead. Peak RSS is not measured;
this change trades temporary storage for less scalar decoding work.

## Reproduction and records

The [controller](../cloudanalyzer/scripts/benchmark_ply_xyz_loading.py) requires
the original reference/compared PLY files and manifest, preceding I/O module
and hash, native build index, saved actual registration record, source root
and new work/output paths. Defaults are three attempts and five timed pairs.
It refuses existing destinations and conflicting options. Files, sources and
libraries are fingerprinted; the frozen archive preserves the preceding I/O
module and candidate sources.

Windows Python 3.12.10, NumPy 2.3.4 and Open3D 0.19.0 use 12 OpenMP/Rayon
threads and one OpenBLAS thread for timing. Records pin 18 Python/source
fingerprints, 31 unchanged native Rust fingerprints, the installed extension
and Open3D CPU library. Production does not impose benchmark thread counts.
This step adds no GPU dispatch or additional parallel worker pool.

- PandaSet: [initial](../benchmarks/cpu/io/real-source/windows-pandaset019-ply-xyz-initial.json), [final](../benchmarks/cpu/io/real-source/windows-pandaset019-ply-xyz-final.json)
- Autoware: [initial](../benchmarks/cpu/io/real-source/windows-autoware-planning-ply-xyz-initial.json), [final](../benchmarks/cpu/io/real-source/windows-autoware-planning-ply-xyz-final.json)
- Tokyo: [initial](../benchmarks/cpu/io/real-source/windows-tokyo-takanawadai-ply-xyz-initial.json), [final](../benchmarks/cpu/io/real-source/windows-tokyo-takanawadai-ply-xyz-final.json)
