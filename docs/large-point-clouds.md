# Large point clouds

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
`Content-Range` (and `Content-Length` when available). The Python reader also
pins the total size and a strong ETag, when provided, and sends `If-Match` on
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
Connecting this core to bounded Python/Web selections and tile processing with
halo, cancellation and durable checkpoints is the next stage. Accuracy-sensitive
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
