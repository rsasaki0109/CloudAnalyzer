# Large point clouds

## Web full-density working box

Open a COPC file, then use **COPC full-density box** to select its original
source coordinates and a point limit (default 200,000; maximum 1,000,000).
**Use clipping box** copies the current clipping bounds. **Read full-density
box** reads all overlapping octree levels, filters exact inclusive XYZ bounds,
and adds a working cloud. The original display cloud is hidden; Undo restores
it. The new cloud can be used by the existing signal measurement and other
map tools. The ordinary clipping crop only copies displayed points.

Traversal accepts one page/node at a time, with limits of 8 MiB for the header,
1 MiB per hierarchy page, 16 MiB per compressed node, 32 MiB per raw node,
16,384 pending entries and 32 nested pages. Oversized selections fail without
adding a partial cloud or thinning their points. Cancel aborts in-flight HTTP
requests and discards pending results; synchronous node decoding cannot be
interrupted. Moving/removing the source invalidates its original-coordinate
reader. Changing form inputs during a request discards its result.

Remote full-density selection requires a strong ETag exposed through CORS and
`If-Match` allowed in requests, in addition to valid byte ranges. A changed ETag
or source size fails before consuming the response body. Without an exposed
strong ETag, use the local file. Display loading can still work without one.

These are per-operation buffer/output limits, **not** a browser memory ceiling:
display clouds, worker pools, indexing, draw buffers and Undo history also retain
memory, and WASM memory may keep its high-water allocation. Remove unused clouds
to release their data. The result has the normal viewer's XYZ64, intensity,
classification and normalized 8-bit RGB attributes; it does not retain GPS time,
all raw LAS fields or original CRS records. Use Python tile exports below for
original LAS records and CRS metadata.

The 42,000-point fixture's box `[8,8,-1]` to `[20,24,10]` selects 1,965 points.
Browser regression tests compare its exported XYZ, intensity and classification
against a separate full-source load, including duplicate records. Loading only
the 2,000-point display root still yields the same full-density box. Overflow,
empty selection, changed HTTP identity and cancellation must publish no cloud.

On 2026-10-01 a local Web check used the public Autzen COPC file below
(10,653,336 source points). The display loaded only its 61,201-point root.
The box `[636800,851200,400]` to `[636900,851300,700]` read 9 nodes and
selected 3,708 points, matching a separate sequential laspy scan's XYZ,
intensity and classification multiset exactly. Selection transferred
3,119,416 bytes, excluding the already cached header; this explains its
65,536-byte difference from the Rust example's range IO below. One warm-cache
browser selection/index/display request took 0.40 s. The main worker reported
21 MB WASM memory before and after; this is not peak process/browser memory
or a GPU comparison. Undo restored the original cloud, and the working cloud
was available to the signal tool. This is a real ten-million-point source test,
not physical ten-billion-point validation.

LAS 1.4 point counts remain `u64` in the Rust readers, including WASM builds.
The extended count is authoritative even when a legacy count is also present,
as required by the [LAS 1.4 public header definition](https://github.com/ASPRSorg/LAS/blob/main-1.4/source/02.04_header.sub).
Local buffer sizes and record spans are checked before narrowing to an address
size. Uncompressed whole-file reads validate the available record bytes before
allocating from the header. Whole-file LAZ reads reserve a small initial output
and decompress in blocks of at most 4 MiB; the retained output still grows with
the requested points. LAS/LAZ chunk layouts reject more than two million chunks.

Remote LAS/COPC reads require HTTP 206 and a valid `Content-Range` matching the
request. Servers that ignore ranges are rejected before their full body is
buffered. Encoded, truncated and oversized responses are rejected. HTTP byte
ranges are limited to 64 MiB per request; this is a response limit, **not** a
total process memory limit. Browser CORS configuration must expose
`Content-Range` (and `Content-Length` when available). Python and Web readers
pin the total size and a strong ETag, when provided, and send `If-Match` on
later requests. Local reads do not require an HTTP server.

The Web regression with a 10,000,000,000-point header uses a 42,000-point fixture
body to check WASM metadata without allocating ten billion points. It is **not**
a ten-billion-point processing benchmark. JavaScript numeric counters are exact
only through `Number.MAX_SAFE_INTEGER`; ten billion is within that range.

Current Web COPC loading and Python `read_copc` choose whole octree levels for
display. The root is always loaded even when it exceeds the requested point
budget. They accumulate the chosen output in memory; LOD is not full-density
analysis. Existing native vector-map operations load the input cloud in memory.
These APIs should not be described as bounded processing of an entire
ten-billion-point cloud.

A real HTTP check on 2026-10-01 read the header and root hierarchy page of
[the public Autzen COPC example](https://s3.amazonaws.com/hobu-lidar/autzen-classified.copc.laz)
linked by the [COPC specification](https://copc.io/). The header reports
10,653,336 points in 81,123,042 bytes. Two validated ranges transferred 74,432
bytes (65,536-byte prefix plus an 8,896-byte hierarchy page), with a pinned
strong ETag. This checks actual range IO and metadata, not full-cloud processing
time, peak memory, or ten-billion-point throughput.

The Rust core also provides `CopcQuery` for full-density spatial traversal.
It visits all overlapping levels, accepts one hierarchy page at a time, and
keeps a node pending until its caller acknowledges it. It checks page alignment,
file byte spans, octree keys, subtree scopes, duplicate keys, page cycles and
per-item limits. It does not retain the whole hierarchy or selected point cloud.
The subtree layout follows the additive hierarchy described by
[COPC](https://copc.io/) and [EPT](https://entwine.io/en/latest/entwine-point-tile.html#ept-hierarchy).

Defaults are 1 MiB per page, 16 MiB per compressed node, 32 MiB per raw node,
16,384 pending entries and 32 pages along a traversal path. Supported octree
levels are 0–31, records are at most 1,024 bytes, and coordinate scales must be
finite and nonzero. A necessary item that exceeds a limit is rejected before
reading it; it is never silently thinned. These bound data items and metadata,
**not** total process RSS, caller-retained output, or decoder/runtime overhead.

`CopcHeader::decode_records` keeps the original LAS records for point formats
6/7/8, including GPS time, flags, 16-bit color, NIR and extra bytes. It checks
the layered chunk count and total encoded layer sizes before invoking the
decoder, preventing allocations based on impossible layer sizes. Header/VLR
metadata must accompany raw records when publishing a normal LAS artifact.

A native Rust example refuses to overwrite its output, selects an inclusive
box at full density, and writes raw point records or counts them with `-`:

```sh
cd rust
cargo run --release -p ca-core --example copc_query -- \
  ../demo_data/copc-scale/autzen-classified.copc.laz ../notes/selected.raw \
  636800 851200 400 636900 851300 700
```

Raw records are a verification artifact, not a standalone LAS file. The example
may leave a partial raw output on an error; it is not a resumable tile job.
The published Web/Python display APIs still use their existing LOD behavior.
Python full-density batches and Web box selections are available separately as
described below. Python tile processing with halo and durable
checkpoints is described below. Accuracy-sensitive
operations must specify their neighborhood requirements; a display subsample
cannot substitute for the full-density input. GPU work follows measured kernel
costs after the IO and memory limits are in place.

On 2026-10-01 the native example read the real Autzen file using the default
limits. The selected box contained 3,708 points: 9 nodes / 312,842 decoded
points required 3,184,952 bytes of IO. Its unordered multiset of complete raw
records matched an independent sequential laspy scan exactly, including GPS
time, flags, intensity, classification and 16-bit RGB. The enclosing-box run
decoded and counted all 10,653,336 points in 278 nodes. It read 81,185,474 bytes;
the prefix probe overlaps some later node reads, so IO exceeds file size slightly.

Two warm-cache measurements on an i7-9750H Windows laptop observed about
10–11 MB peak process memory for the Rust example, sampled externally with
psutil every 2 ms including Windows `peak_wset`. Whole-file wall times were
11.84 s and 24.96 s under varying machine load; the latter attributed 24.53 s
to decoding and 0.09 s to range IO. These are single-machine observations, not
guaranteed throughput, a complete application memory limit, or a physical
ten-billion-point benchmark. The whole run counts points without retaining
them or writing a full output cloud. The raw ROI artifact contains only the
selected records. Larger nodes, deeper pages, consumers and persisted tile/halo
outputs have different memory/storage costs.

The Python native wheel exposes `CopcStream` / `iter_copc_batches`. Install
`cloudanalyzer-core[copc]` for the optional laspy schema dependency, or use
CloudAnalyzer with an updated core (`cloudanalyzer[fast]`). HTTP(S) URLs, including
presigned URLs, and local COPC files use the same full-density iterator:

```python
from cloudanalyzer_core import CopcStream, CopcLimits

with CopcStream("survey.copc.laz", bounds=(636800, 851200, 400, 636900, 851300, 700),
                chunk_size=10000, limits=CopcLimits()) as stream:
    count = 0
    for batch in stream:
        count += len(batch.positions)
        # positions: XYZ64; records: laspy ScaleAwarePointRecord with all fields.
        # node_offset + ordinals identify original records, including duplicates.
        # Consume/discard here rather than collecting a global array.
```

It decodes one node, then filters its raw records in batches; there is no pool
of outstanding reads or global output list. The Rust-to-Python byte transfer
temporarily copies the raw node. XYZ, filtering masks, selected records and
identities add batch-sized buffers. Limits bound each item, not their combined
RSS or output retained by the consumer. Python limits additionally cap each
page/compressed/raw item at 64 MiB, pending entries at 65,536, path depth at 64,
metadata at 8 MiB and requested batches at 1,000,000 points.

Source VLRs and non-index EVLRs retain CRS/extra-field metadata. EVLR headers
are followed by checked offsets; COPC hierarchy payloads are skipped, so a large
hierarchy EVLR is not materialized. At most 16,384 EVLR headers are inspected;
non-index metadata that exceeds the combined limit fails explicitly. The original
header prefix is retained as `raw_header`; raw records alone are not a standalone
LAS artifact. Publishing ordinary LAS requires removing COPC/compression index
VLRs and supplying the source schema/metadata through an appropriate writer.

Use the context manager to close on errors/early exit. Alternatively close an
`iter_copc_batches` generator explicitly after an early break. `cancel=event.is_set`
raises `CopcCancelled` between ranges/nodes/batches. Blocking HTTP IO has a timeout
(60 s default); it cannot be interrupted midway through a blocking request. Local
ranges pin the open descriptor's file identity, size and timestamps. HTTP ranges
pin size and an available strong ETag. Without a strong ETag, size checks alone
cannot prove immutable contents for resumption. The low-level `nodes()` iterator
can skip committed nodes without reading their point bytes, but this API does not
itself persist a tile checkpoint; the separate job API below does.

`ca.io.iter_point_chunks` uses this path for local `.copc.laz` and remote COPC
HTTP(S) inputs. It yields only XYZ under its existing inclusive-box/empty-result
contract. It requires the updated Rust core and does not use PDAL's full-output
array pipeline. `s3://` inputs need an HTTP(S)/presigned URL. Local ordinary LAS/LAZ
still uses sequential laspy chunks; PCD/PLY still load via Open3D before chunking.

On 2026-10-01, the Python iterator selected the same 3,708 complete raw Autzen
records as the independent laspy oracle through both local and real HTTP input.
Each read 3,185,012 bytes in 12 ranges (including a 60-byte EVLR header), visiting
9 nodes / 18 selected batches with a 10,000-point batch setting. Local ROI wall
time was 0.29 s and HTTP ROI 15.10 s; observed peak process RSS was 65.2 MB and
69.2 MB, including Python/NumPy/laspy and a roughly 54.6 MB baseline. The local
whole-file count visited 278 nodes / 1,210 batches and counted all 10,653,336 points
in 7.57 s, reading 81,185,534 bytes, with 66.0 MB peak RSS. Sampling used external
psutil every 2 ms plus Windows `peak_wset`; each mode ran in a separate process.
Only the small ROI records were retained for equality verification. These are
single-machine observations under varying load, not controlled comparisons with
the earlier Rust run, hard process memory caps, or a ten-billion-point benchmark.

[`ca copc-tile`](commands/copc-tile.md), its Python API and MCP tool now persist
bounded full-density XY tile jobs with unique core ownership, flagged halo
copies and source node/ordinal identities. One node pack is published atomically
before its SQLite transaction commits. Resume replays bounded hierarchy pages,
hashes committed packs and skips their compressed point reads. Output metadata,
tile fragments and counters remain on disk. The default batch has 10,000 points,
frame payloads are capped at 8 MiB, SQLite's cache at 8 MiB, and per-node output
at 256 MiB / 65,536 fragments. The underlying stream retains its separate item
limits and temporary raw-byte copy; these are working-data limits, not an RSS cap.

The real Autzen test used grid width 100, halo 2 and origin `(0,0)` in source
units. The ROI retained all 3,708 uniquely owned records, plus 976 halo copies
across four target cells. Independent sequential laspy queries checked each
cell's complete raw-record/halo multiset, and no cell duplicated a source
identity. Core LAS export returned 3,708 original records; halo LAZ export
returned 4,166 records with explicit identities. CRS VLR bytes were preserved.
Non-index EVLR byte snapshots and export are covered separately by regressions.
Actual HTTP pause after two nodes and subsequent resume also returned the same
3,708 original records / 976 halo copies with the source's strong ETag pinned.

On the same Windows i7 laptop, final ROI saving took 0.39 s, wrote 211,086 pack
bytes and observed 62.1 MB peak process RSS. The full job wrote 518,545,937 pack
bytes / 82,631 fragment rows across 278 nodes and 1,580 cells. Core ownership
totaled all 10,653,336 source points; halo copies totaled 869,697. A test worker
was forcibly terminated after publishing its first pack but before committing
the node. A separate process resumed the job, recovered the uncommitted pack,
and completed the full output in 48.33 s with 75.4 MB peak RSS (about 54.6 MB
baseline) and 81,185,534 source bytes read. This tests process interruption;
it is not a power-loss or storage-failure guarantee.

Resuming the completed full job took 3.53 s with 58.5 MB peak RSS. It read
74,492 source bytes in three ranges and verified all 518.5 MB of output packs;
it committed/decoded no new nodes. External psutil sampled every 2 ms including
Windows `peak_wset`; independent oracle/export checks ran separately from the
timed worker. Cache and machine load affect these observations. Raw inputs,
outputs and measurement scripts remain ignored. The full job checks actual
point ownership and writes its output, while the earlier whole-query benchmarks
only counted points; their wall times are not interchangeable.

Physical ten-billion-point input remains unbenchmarked. Node/page shape must
fit configured limits, and persistent output/storage must fit the chosen disk
and filesystem. With 36-byte original records, raw packs use 45 bytes per
record before halo: ten billion records alone require about 450 GB plus pack,
SQLite, halo and exported-artifact costs. This is format arithmetic, not a
measured storage result or runtime extrapolation. Current Web loading remains
LOD output in memory, and native vector-map whole-cloud paths still need a
bounded spatial selection. GPU work is reserved for measured kernel needs.
