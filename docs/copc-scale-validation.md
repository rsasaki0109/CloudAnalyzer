# Reproducing COPC scale measurements

Run `scripts/benchmark_copc_scale.py` from the repository root with a local,
complete COPC file. Install the native core built from the same checkout, plus
`laspy`, `lazrs` and `psutil`. The script creates a new JSON report and a sibling
`.artifacts` directory; it refuses to overwrite either. Keep real sources,
reports and packs under the ignored `demo_data/` and `notes/` directories.

```sh
python -m pip install "laspy[lazrs]" psutil
python scripts/benchmark_copc_scale.py demo_data/copc-scale/sofi.copc.laz \
  --bounds 376299 3757799 -84 376301 3757801 139 \
  --grid-size 100 --halo 2 \
  --http-source https://hobu-lidar.s3.amazonaws.com/sofi.copc.laz \
  --full-tiles --output notes/sofi-scale.json
```

The [COPC example list](https://copc.io/#example-data) links
[SoFi Stadium](https://hobu-lidar.s3.amazonaws.com/sofi.copc.laz), courtesy of
the US Army Corps of Engineers Remote Sensing & GIS Center of Expertise and
the National Center for Airborne Laser Mapping. On 2026-10-02 its actual LAS
header reported **364,384,576 points** in **2,029,696,615 file bytes**, point
format 6 with 30-byte records. The file size is measured from HTTP
`Content-Range`, rather than the example page's approximate download size.
The data are not redistributed with CloudAnalyzer.

Coordinates, grid width and halo are in the source's units. The example box
straddles two XY grid boundaries. It is an IO/ownership test, not evidence of
lane or signal extraction accuracy.

## What the script measures

Each operation runs in a fresh child process. The parent samples that child's
RSS every 2 ms and also uses Windows `peak_wset` when available. The report
records the child's post-import baseline, operation wall time, full child
lifetime, source SHA-256, package versions, batch settings and logical IO.
Sampling can miss short peaks on platforms without a peak counter. This is an
observation, not an enforced process memory limit. Cache and concurrent machine
load affect runtime; the script does not flush the filesystem cache.

1. A **sequential laspy/LAZ scan** independently decodes the whole physical
   source and checks its header count. It retains only the halo-expanded small
   oracle box, with a default cap of 200,000 points. Larger selections fail
   explicitly; choose a smaller box or set `--max-oracle-points` up to 1,000,000.
2. **Local and optional HTTP full-density box streams** visit overlapping octree
   levels. They retain the same capped selection for later raw-record equality
   checks. HTTP uses the existing validated byte-range/source-identity reader.
3. A **whole-source stream** decodes and counts all points, discarding each batch.
   Its peak RSS and runtime do not include saving a complete output cloud.
4. A **box tile job** pauses after two new nodes and resumes in another process.
   A separate verification process checks complete raw-record multisets, XYZ64
   coordinates, unique source identities and halo flags against the sequential
   oracle, including duplicate records. Oracle/verification measurements are
   reported separately from the tile writes. Small core LAS exports must also
   match the oracle, with unchanged coordinate scales, offsets and non-index
   source VLR metadata.
5. With **`--full-tiles`**, the script persists the whole source. It terminates
   only its own benchmark worker after the first pack is published and before
   its SQLite node commit. Another worker recovers that uncommitted pack and
   finishes. A third resumes the completed job: all committed packs are hashed,
   and no new point nodes may be decoded or written. This is a process
   interruption test, not a power-loss guarantee.

Plan disk space before enabling full tiles. Each original record adds a uint64
ordinal and one halo byte to the raw packs. For this 30-byte source, uniquely
owned point payloads alone need 14.21 GB, before halo copies, pack/fragment
headers, SQLite fragment rows and exports. This is format arithmetic; the original compressed
file's 2.03 GB size is not an estimate of persisted tile storage.

The JSON is saved after each completed phase. A failed phase leaves its stderr
and artifacts for inspection. The benchmark itself requires a new output path
on another run; existing tile jobs can be resumed separately through
[`ca copc-tile --resume`](commands/copc-tile.md) with matching options. No network
data are downloaded by the benchmark. The optional HTTP box only requests
needed ranges of the same source.

## Measured SoFi result, 2026-10-02

The command above completed on Windows 11 with an i7-9750H (12 logical CPUs)
and approximately 32 GB RAM. It used a release native core from main `4d64557`,
benchmark commit `31eced2`, Python 3.12, NumPy 2.3.4, laspy 2.7.0, lazrs 0.8.2
and psutil 7.2.2. Each child's post-import baseline was approximately 56 MB.
The source was hashed before workers started, so its filesystem cache was
already warm. No GPU processing was used.

| Operation | Result | Operation seconds | Peak child RSS (MB) | Logical source bytes read |
|---|---|---:|---:|---:|
| Independent sequential oracle | 364,384,576 counted; 13,546 retained in expanded box | 72.90 | 124.2 | Not instrumented |
| Local full-density box | 1,519 selected; 10 nodes / 227,580 decoded | 0.21 | 65.7 | 2,615,955 |
| Real HTTP full-density box | Same 1,519 complete raw records | 17.02 | 69.8 | 2,615,955 |
| Whole-source count stream | 364,384,576 counted; no retained cloud | 218.87 | 73.3 | 2,029,702,398 |
| Whole tile job after interrupted publication | 364,384,576 owned; 29,945,087 halo copies | 1,408.85 | 84.4 | 2,029,702,398 |
| Resume completed whole job | All packs verified; 0 new nodes / 0 new pack bytes | 162.88 | 62.2 | 490,524 |

MB in the RSS column means decimal megabytes. These are separate-process
observations under varying load, not hard memory caps or controlled speed
comparisons. The independent oracle uses a different decoder/batch consumer;
its time is not a like-for-like comparison with CloudAnalyzer. Process RSS does
not measure total machine memory, filesystem cache, the parent or other apps.

The whole job processed **13,163 nodes**, produced **137 tiles** and **98,233
fragment rows**, and wrote **15,379,304,399 pack bytes** (15.38 GB), excluding
SQLite and source metadata. This matches 39 bytes per core/halo record plus
34 bytes per node pack header. A separate metadata-only traversal found a
maximum of 93,876 points / 697,707 compressed bytes / 2,816,280 raw bytes per
node, well within the default item limits. This particular node layout is part
of the measured conditions; larger nodes can need different limits.

The box job paused after two new nodes (0.11 s), then resumed its remaining
eight nodes (0.30 s). Its four cells contained 1,519 uniquely owned records and
36,115 flagged halo copies. Independent verification checked each cell's
complete raw-record/halo multiset and source identities exactly. Core LAS
exports totaled the same 1,519 unchanged records, with exact XYZ64, source
scales/offsets and non-index VLR metadata. HTTP used 14 validated ranges and a
pinned strong ETag, and matched the same raw oracle records.

Completed resume hashed all **15.38 GB** of committed packs, while reading only
490,524 source bytes in 114 metadata ranges. It decoded/wrote no new nodes.
Resume still has output-verification cost; its source IO count must not be
mistaken for all IO or a near-instant operation. The publication interruption
left the first pack on disk with zero committed nodes, and recovery completed
in a new process. This demonstrates process recovery on the measured source.

A separate 30 s nonblocking Python-stack sample during tile creation collected
872 active samples (28 sampling errors). Of those, 586 contained the tile
membership/grouping function; this suggests CPU grouping is worth profiling
before introducing GPU work. It is a short Python-stack observation, not a
whole-run kernel/wall-time breakdown or a GPU comparison. This external sample
may have perturbed the measured job time.

Source SHA-256:
`fd6848a0eaee3e21f7ce3a91af44ed3ae06ae8439b7f65828b4a5d956685f699`.
Expanded oracle raw-record multiset SHA-256:
`e2085ddebd4070c89290f80af1d6666e9d88211a912678fc8ba723a5256ffad6`.
Original scales are `(0.001, 0.001, 0.001)` and offsets are
`(376331.56, 3757833.285, 27.238)`; no CRS transformation was applied.
The full JSON and artifacts remain local and ignored under `notes/sofi-scale*`.

## What remains unverified

Full-source point counts and completed pack checks do not constitute an
independent byte-for-byte comparison of every persisted record. Exact raw
multiset/halo verification is limited to the explicitly capped oracle box.
Synthetic CI checks cover the measurement/recovery machinery, not large-cloud
throughput. Physical **10,000,000,000-point** input remains unbenchmarked; do not
extrapolate runtime or a total memory guarantee from a smaller source.

See [the processing limits and earlier measurements](large-point-clouds.md).
