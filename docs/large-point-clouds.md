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

The next processing path will read full-density spatial ranges, yield bounded
batches, and process tiles with halo and durable checkpoints. Accuracy-sensitive
operations must specify their neighborhood requirements; a display subsample
cannot substitute for the full-density input. GPU work will follow measured
kernel costs after the IO and memory limits are in place.
