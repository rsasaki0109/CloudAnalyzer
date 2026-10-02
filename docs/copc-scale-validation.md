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
headers, SQLite and exports. This is format arithmetic; the original compressed
file's 2.03 GB size is not an estimate of persisted tile storage.

The JSON is saved after each completed phase. A failed phase leaves its stderr
and artifacts for inspection. The benchmark itself requires a new output path
on another run; existing tile jobs can be resumed separately through
[`ca copc-tile --resume`](commands/copc-tile.md) with matching options. No network
data are downloaded by the benchmark. The optional HTTP box only requests
needed ranges of the same source.

## What remains unverified

Full-source point counts and completed pack checks do not constitute an
independent byte-for-byte comparison of every persisted record. Exact raw
multiset/halo verification is limited to the explicitly capped oracle box.
Synthetic CI checks cover the measurement/recovery machinery, not large-cloud
throughput. Physical **10,000,000,000-point** input remains unbenchmarked; do not
extrapolate runtime or a total memory guarantee from a smaller source.

See [the processing limits and earlier measurements](large-point-clouds.md).
