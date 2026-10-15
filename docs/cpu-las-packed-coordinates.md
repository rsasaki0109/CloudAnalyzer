# Packed XYZ rows in the LAS/LAZ reader

The laspy-based `load_point_cloud` path now constructs contiguous XYZ rows
before creating Open3D's owned point vector. Four complete real-source loading
comparisons are **85.79–95.94% shorter** by median paired duration, with
**20/20 faster timed pairs** and identical full XYZ outputs.

## Behavior and scope

The previous `np.vstack([las.x, las.y, las.z]).T` produces a column-contiguous
array. `np.column_stack((las.x, las.y, las.z))` produces row-contiguous storage
for Open3D's bulk conversion. The same laspy coordinate scaling executes in
both policies; point order, f64 coordinate bits, ownership and empty-input
errors are preserved. Both paths continue to return coordinates without
importing LAS color, intensity or classification attributes into Open3D.

This affects the standard laspy path when the optional Rust core is unavailable
or disabled. The installed native reader, streaming point-chunk reader,
spatial COPC queries and other formats retain their existing implementations.
These measurements establish full loading performance for this path, rather
than whole comparison, registration, GPU or native-reader speedups.

## Real inputs

The [input manifest](../benchmarks/cpu/io/real-source/las-packed-inputs.json)
pins two existing measured source files and their full decoded coordinate
oracles. No new geometry, thinning, perturbation or file conversion is used
for timing. Both policies read the same existing files unchanged.

| Input | File size, bytes | Full decoded points |
| --- | ---: | ---: |
| Autoware planning LAS export | 63,282,651 | 1,757,841 |
| Autzen classified COPC/LAZ | 81,123,042 | 10,653,336 |

Autoware uses the existing `planning-signal-source.las` export of the official
[planning sample map](https://docs.autoware.org/main/demos/planning-sim/),
retaining original point order and the existing header quantization. Every
coordinate matches the original PCD after rounding to that header's integer
grid. Three Z coordinates differ from the PCD, with maximum absolute difference
`4.76837158203125e-7`; all remaining coordinate values match. No further
rounding is introduced by this change. The original PCD hash matches the
preceding [real-source corpus](c2c-real-source.md).

Autzen uses the original full public classified COPC example, with the same
file hash and point count as its existing download record. See the
[existing source description](large-point-clouds.md) and the
[public source file](https://s3.amazonaws.com/hobu-lidar/autzen-classified.copc.laz).
This is an ordinary whole-file laspy read of its LAZ point stream, rather than
a COPC spatial query or LOD subset. Source data is not included in these results;
the existing attribution is retained and no additional dataset license is asserted.

## Measurements

The timer surrounds the complete `load_point_cloud` call: actual LAS/LAZ file
decoding, header scaling, XYZ assembly, owned Open3D conversion, empty-input
check and internal cleanup. Returned-cloud release, coordinate validation and
hashing, imports, fingerprints and CLI serialization are outside the timer.
No decoder, point buffer or read result is replaced.

| Source/pass | Strided median, ms | Packed median, ms | Median paired shortening | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Autoware initial | 3,585.45 | 151.93 | +95.94% | 5/5 |
| Autoware final | 1,933.42 | 101.13 | +94.74% | 5/5 |
| Autzen initial | 17,338.77 | 2,370.28 | +85.79% | 5/5 |
| Autzen final | 12,035.41 | 1,615.59 | +87.03% | 5/5 |

Each source/pass starts a fresh process. An excluded full byte-parity pair,
separate first pair and two warm-up pairs precede five alternating timed pairs.
Sources run Autoware → Autzen, then reverse. The initial pass starts strided
and the final pass starts packed, balancing the first policy across timed
pairs. No edits, builds, tests or scans run during timing. All observations
are retained, including the substantial variation in baseline duration.

Shortening is `median(100 * (1 - packed_ms / strided_ms))`, rather than a ratio
of marginal medians. File pages are warm; scheduling, clocks, other applications
and allocator/page state are uncontrolled. These are not cold-process or
per-call guarantees. Both layouts contain the same 24-byte-per-point f64 XYZ
payload; no peak RSS or memory-saving claim is made.

## Verification

All 72 observed loads check the full coordinate-buffer hash, shape, dtype and
empty color/normal/covariance attributes against independent coordinate oracles.
Eight excluded parity outputs also compare every coordinate byte directly.
The oracles are decoded in 250,000-point chunks from raw signed integer
X/Y/Z, multiplied by each header scale and then added to its offset.

An independent audit decodes all 12,411,177 input points again in 131,071-point
chunks and checks every axis against those oracles. It also loads both sources
with the original and current readers, comparing 24,822,354 coordinate vectors
byte-for-byte; verifies all 72 recorded output fingerprints and source/library
hashes; rechecks Autoware's complete quantization relationship to its PCD; and
checks 12 invalid/conflicting CLI cases while preserving sentinel destinations.

Twenty-one new regression cases cover LAS and LAZ, point formats 3/7, one and
multiple points, signed integer extrema, survey and binary scales, large and
negative offsets, source-file preservation, owned storage, empty-input errors
and unchanged native dispatch. Related I/O/comparison/CLI tests pass 167 cases
both with the installed native core and with it disabled. Type checking passes
all 166 application files.

## Reproduction and records

The [controller](../cloudanalyzer/scripts/benchmark_las_coordinates.py) requires
the source root, archived preceding I/O module and its hash, input manifest,
dataset key, full coordinate oracle and fresh work/output paths. Defaults are
two warm-up pairs and five timed pairs. Duplicate/abbreviated options, invalid
counts and existing destinations are rejected.

An oracle can be rebuilt from either pinned source with the following logic,
then supplied using `--coordinate-oracle`. Its full hash must match the manifest:

```python
with laspy.open(source) as reader, oracle.open("xb") as stream:
    for chunk in reader.chunk_iterator(250_000):
        xyz = np.empty((len(chunk), 3), dtype="<f8")
        for axis, field in enumerate(("X", "Y", "Z")):
            xyz[:, axis] = (
                np.asarray(getattr(chunk, field)).astype(np.float64)
                * reader.header.scales[axis] + reader.header.offsets[axis]
            )
        stream.write(xyz.tobytes())
```

Records pin eight current application/test/controller files, the baseline,
input manifest/files and full coordinate oracles, 39 laspy source files,
Open3D's CPU library, the lazrs extension and its wrapper. Timings use Windows,
Python 3.12.10, NumPy 2.3.4, Open3D 0.19.0, laspy 2.7.0 and lazrs 0.8.2,
with 12 OpenMP/Rayon threads and one OpenBLAS thread. Production does not
force those thread settings or introduce a worker pool or public option.

- Autoware LAS: [initial](../benchmarks/cpu/io/real-source/windows-autoware-planning-las-packed-initial.json), [final](../benchmarks/cpu/io/real-source/windows-autoware-planning-las-packed-final.json)
- Autzen LAZ: [initial](../benchmarks/cpu/io/real-source/windows-autzen-laz-packed-initial.json), [final](../benchmarks/cpu/io/real-source/windows-autzen-laz-packed-final.json)
