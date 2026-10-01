# ca copc-tile

Create full-density XY tiles from local or HTTP(S) COPC with bounded reads,
raw LAS attributes, optional halo and a durable SQLite journal. Install an
updated `cloudanalyzer[fast]` core; ordinary `ca split` still loads through
Open3D. Coordinates, grid width, origin and halo use the **source's units**.
This command does not transform CRS or assume those units are metres.

```sh
ca copc-tile survey.copc.laz --out tiles --grid 100 --halo 2 \
  --box 636800,851200,400,636900,851300,700 --stop-after-nodes 10

# Repeat with identical source/grid/halo/box/origin/chunk settings:
ca copc-tile survey.copc.laz --out tiles --grid 100 --halo 2 \
  --box 636800,851200,400,636900,851300,700 --resume

# Export uniquely owned points from a committed tile:
ca copc-tile-export tiles --i 6368 --j 8512 --out tile.las
# Include neighbor points and explicit halo/source identity dimensions:
ca copc-tile-export tiles --i 6368 --j 8512 --out tile-with-halo.laz --include-halo
```

Initial output must be a new directory. `--resume` verifies the source identity,
coordinate/processing options, metadata snapshots and committed pack hashes.
`--stop-after-nodes` commits that many new nodes and returns `paused`; an already
completed node is checked/skipped without decoding its point bytes. Ctrl+C
returns `cancelled` after rolling back the current node. Python additionally
accepts `cancel=event.is_set`; cancellation occurs at range/node/batch/frame
boundaries. A blocking HTTP request has the stream's 60 s default timeout.

| Option | Default | Meaning |
|---|---|---|
| `--out` | Required | New job directory, or existing job with `--resume` |
| `--grid` | Required | Positive finite XY cell width in source units |
| `--origin` | `0,0` | XY grid origin in source coordinates |
| `--halo` | `0` | Nonnegative neighbor width; candidate fan-out must fit 64 |
| `--box` | Whole root cube | Inclusive original-coordinate min XYZ, max XYZ |
| `--chunk-size` | `10000` | Requested batch points, additionally capped by 8 MiB frame bytes |
| `--stop-after-nodes` | All remaining nodes | Commit a bounded prefix per call |
| `--max-node-output-mib` | `256` | Per-node pack quota, up to 2 GiB |
| `--max-fragments-per-node` | `65536` | Per-node fragment quota, up to 1,000,000 |

Output/fragment quotas can be increased on resume. Other options and the source
must agree; a different grid/halo/query needs a new job. If an input node cannot
fit the stream's raw/compressed/page limit, the job fails explicitly rather than
thin points. Python can supply `limits={...}` using `CopcLimits` field names;
these limits must remain the same on resume. A larger grid, smaller halo or an
externally retiled source can reduce output fan-out for a new job.

Core ownership uses `floor((XY-origin)/grid)`, so every selected source record
belongs to one half-open cell. Duplicate coordinates remain distinct records.
Halo copies carry an explicit flag and do not increase `core_points`. Only cells
intersecting the original inclusive ROI are targets. The query expands the ROI
by halo in XY; for a partial cell, neighbor support is limited to that expanded
ROI, not the unselected remainder of its entire cell. Z stays within the original
query bounds. Finite halo is suitable only for algorithms whose required radius
fits it; it does not establish exact global nearest-neighbor or ICP results.

The output contains `manifest.sqlite3`, bounded source metadata snapshots and
one internal `.pack` per processed node. Packs keep unchanged LAS records plus
uint64 source ordinals and halo flags. SQLite indexes tile fragments and counts
on disk, with an 8 MiB page cache; no global tile/node list is retained. The
returned JSON is a small summary. `source_read_bytes` counts source ranges;
`verified_pack_bytes` counts committed output bytes hashed on resume and
`written_pack_bytes` counts newly published packs. These are logical IO counts,
not physical device/cache traffic or SQLite journal write volume.

A node's frames are written into a staged pack, flushed, then published before
the node transaction commits. A later resume repairs only uncommitted files
whose complete pack header matches this job UUID and node. Missing/corrupt
committed artifacts and unrelated existing files are refused. An OS writer lock
is released automatically when a process exits; a stale PID lock needs no deletion.
Abandoned tiny unique seed files from termination during header seeding are
unreferenced. A job interrupted before initialization finishes is refused as
incomplete; start in another new directory. Filesystem durability depends on
file `fsync`, POSIX directory synchronization and SQLite FULL/WAL support.

HTTP jobs require a strong exposed ETag. Resume may refresh presigned query
credentials for the same URL path/object identity; source size, header and ETag
must still match. Without that ETag, a resumable HTTP job is refused. Local jobs
pin file identity, size and timestamps. Jobs use SQLite signed 64bit offsets and
totals; source point counts times the maximum candidate fan-out must fit that
range. Ten billion points are within it. Grid indices must fit the exact supported
range near ±2^52. These conditions do not promise a physical ten-billion-point
benchmark or a cap on whole-process RSS.

LAS export streams core records with original schema, scales, offsets, CRS/VLRs
and non-index EVLRs. COPC/LAZ index VLRs are removed. `--include-halo` adds
`ca_halo`, `ca_source_node_offset` and `ca_source_ordinal`; conflicting existing
dimension names are rejected. Export checks committed hashes and publishes
without overwrite using a same-directory hard link (the filesystem must support
it). LAZ needs a laspy compression backend; its encoder also retains chunk-table
metadata. LAS avoids that encoder metadata. Neither export retains a global cloud.

Export requires a completed job, so an unfinished prefix is not published as
a complete tile. `iter_tile_batches` can inspect committed frames of a paused
job; those frames are only the committed prefix until the job finishes.

Python uses `ca.copc_tiles.tile_copc`, `iter_tile_batches` and `export_copc_tile`.
Close a tile generator after early exit. MCP tools are `tile_copc` and
`export_copc_tile`, with the same source/ownership/resume contracts. The viewer
opens the exported LAS/LAZ, not the internal packs.

See [real-data measurements and remaining limits](../large-point-clouds.md).
