# Resume browser editing

Use **Save project** under Clouds to download `project.cloudanalyzer.json`. It stores the vector map's editable geometry and rules, the pose graph's current and initial poses, fixed nodes, loop/odometry edges, gravity and floor constraints, dynamic point flags, display and loading settings, and lane review decisions and notes.

Keep the original cloud files, graph/trajectory files and scan files alongside the project. Their raw point data is external. Local files are identified by a SHA-256 digest covering every byte, computed in 8 MiB blocks without loading a second full copy into memory. Range-loaded URLs need a strong ETag; the URL, byte length and ETag must still match on reopening. Servers must expose the ETag through CORS.

Use **Open project**, then open the requested original files together or in later batches. A file with the same name and size but different contents cannot satisfy the project. Each original cloud retains the loading point limit used when it was opened, even if the global preset changes later; an already-loaded source with a different limit is reopened from its verified original. Graph scans are rebound to the saved editing state after the original inputs have been checked. Files recorded from multiple merged sessions are required too. Cloud names must be unique when saving. Reopening starts a fresh Undo history.

The project does not include derived clouds, user-computed scalar fields, transient build-evidence overlays, discovery previews or Undo history. Export derived results separately if needed. Metadata files are limited to 64 MiB. Existing **Save session** files remain supported for view-only sharing.

## Automatic browser recovery

**Auto save editing state in this browser** stores one recent project in this site's browser storage after editing settles. It includes verified source references, map/graph editing state, saved review decisions and the current selected lane's unsaved review text. It does not store raw source files, derived outputs or Undo history. Save a lane review before moving to another lane or exporting its decision. The checkbox's preference is retained in this browser.

When a browser copy is found on reopening, automatic saving pauses to protect it. Choose **Resume saved work**, then reselect the matching original files. Contents are verified as for a portable project, including graph inputs and each cloud's loading limit. Same-name files with different contents cannot restore the copy. The previous copy remains protected until restoration finishes, or you explicitly choose **Discard browser copy** to continue with the current workspace. **Download browser copy** exports its saved project decisions; unsaved lane-review text remains in browser recovery until you save that review.

The save status reports waiting, saving, saved and failure states separately from analysis progress. Changes during snapshot creation cancel obsolete fingerprinting; changes during a storage commit remain pending for the next save. Storage updates are atomic and refuse to overwrite a copy changed by another tab. Capacity/access failures preserve the previous copy and keep a closing warning for edits not protected by a browser save or portable project export. Use **Retry browser save** after a temporary failure, or **Save project** for a portable backup. Closing warnings cannot guarantee recovery during a crash or forced termination; wait for the saved status before closing.

Browser copies belong to this browser profile and site origin. Private browsing, cleared site data or browser storage eviction can remove them. Keep original files and downloaded projects separately for durable backups.

## Lane reviews

The vector map's **Lane review** queue can visit unreviewed, reviewed, deferred, low-coverage or all lanes. Select a lane, choose **Reviewed**, **Needs fixes**, **Deferred** or **Unreviewed**, write notes and press **Save lane review**. Run **Check source coverage** before using the low-coverage queue; missing or limited coverage is a reason to inspect a lane, not an automatic approval.

The lane table uses the same filter as **Next lane** and shows 25 lanes per page, ordered by lane ID. Click a lane ID to select it, open its review editor and focus the map on its geometry. **Next lane** also moves the table to the selected lane's page. Changed decisions show **Review again**; notes remain available in the editor.

Use **Export all reviews CSV** to share every current lane's saved state and notes, including untouched lanes as **Unreviewed**. Export includes all lanes regardless of the active filter or page. Save changes in the editor before exporting. Columns are `lane_id`, `status`, `notes`, `updated_at`, `previous_status` and `stale_reason`; untouched lanes have no review timestamp. The UTF-8 file includes a BOM for spreadsheet compatibility, preserves quoted/multiline notes and prefixes text that could be interpreted as a spreadsheet formula with an apostrophe. Deleted lanes and decisions from another map are excluded. CSV is a report; use a project to resume editing.

Geometry, lane attributes, shared boundaries and associated equipment rules are compared against the reviewed state. A relevant change returns affected lanes to **Unreviewed**, preserving their notes and displaying why they need another look. Changing the review source, or changing its points, also requires another review. Deleting a lane removes its review. Review state is saved in projects; plain Lanelet2 exports retain map geometry and rules only.

## Memory and Undo

**Memory and Undo** reports allocated main/pool WASM memory, retained JavaScript arrays and an estimate of geometry GPU buffers. Backing arrays are counted once when multiple fields share them. Undo's cloud, pose and map histories each have a configurable step limit and estimated byte budget (defaults: 20 steps and 128 MiB per history). Oldest steps are discarded when either limit is exceeded; a step larger than its budget is not retained. Cloud history includes a conservative proxy for the native points, attributes and indexes held by referenced clouds, including clouds still in the list; shared cloud IDs are counted once. Map sizes use a conservative serialized-size proxy. The warning threshold does not silently thin point data or block operations.

**Clear Undo and release idle workers** keeps the current clouds, poses and map, releases detached redo clouds, clears the three histories, drops unused display/detail caches and terminates idle parallel workers. Workers are created again when needed. Active computations must finish first. The main WASM heap can reuse freed allocations but does not shrink until the page reloads. Save a project before reloading to reclaim it. The displayed estimates exclude browser overhead, render targets and transient operation buffers; they are not process RSS.

## Validation

In `web/`, run `npm run test:unit`, `npm run build`, then `npx playwright test`. Project tests cover restoration of maps, graph constraints, source validation, review invalidation and memory release. Rust snapshot and map-history tests run with `cargo test --locked -p ca-wasm` in `rust/`.

Opt into the real-data roundtrip and memory tests with `CLOUDANALYZER_REAL_DATA=1 npx playwright test project-real-data.spec.ts --workers=1`. They use the bundled NCLT MCAP and RELLIS-3D samples and attach JSON measurements to the Playwright results. The stress cloud consists of sixteen translated copies of one RELLIS-3D frame (2,097,152 points, approximately 32 MiB), not additional surveyed frames. It verifies 24 edits, a three-step history cap, memory release and project restoration; it does not establish limits for multi-gigabyte captures. See the sample attribution files before redistributing data.

## Keep processed records in browser recovery

Enable **Include current records and original inputs (64 MiB)** under the automatic
save controls to retain the currently loaded point clouds and meshes, including
processed results and computed attributes, alongside the editing state. This
option is off by default and is remembered in this browser. Wait for **Current
point/mesh records are included in this browser copy** before closing. After
reopening, **Resume saved work** restores those records without choosing their
original source files or rerunning filters. **Download browser copy** exports
that saved state as a portable `project.cloudanalyzer.zip`.

The recovery record and its ZIP are committed atomically; capacity or cross-tab
failures preserve the preceding copy and the closing warning. ZIP member hashes
and the project identity are checked before restoration. A cloud already open
with the same display name must be exported and closed before resuming this copy;
its current records are not silently replaced. Unsaved review text is restored
from browser recovery, but is not included in the portable ZIP until the lane
review is saved and a new snapshot captured.

This stores currently loaded geometry, not unloaded original density. The whole
browser recovery record is capped at 64 MiB, including its metadata and stored
ZIP. Original pose-graph source/scan files and the original verified generated-map
review archive are included in new snapshots; Undo starts fresh. Browser storage is not a durable backup:
keep a downloaded workspace snapshot and the original attributed review bundle.
When this option is off, browser recovery retains metadata/source references and
processed results still require their own export or manual workspace snapshot.

## Keep graph inputs and original map evidence in one file

New workspace snapshots include the original pose-graph/trajectory and scan files
needed to rebind its saved poses and constraints. Original scan basenames survive
the ZIP roundtrip, including distinct same-name scans from merged sessions; file
fingerprints resolve their content identities. The same snapshot includes the
original verified generated-map review ZIP if one was opened in this workspace.
This preserves its attribution, decisions, proposals and frozen audits alongside
the current edited map and point records.

Manual workspace snapshots allow 256 MiB of uncompressed content, including the
manifest, metadata, current records and original inputs. Browser recovery remains
capped at 64 MiB including its stored ZIP. There are
at most 127 clouds/meshes, 2,048 ZIP members and 10 MiB project metadata. Generated
review imports keep their original smaller member limit. Oversized workspaces
report failure rather than omit inputs; the previous browser copy remains intact.
Older v1/v2 snapshots remain readable under their original 64 MiB content limit;
snapshots with external graph references still request those source files.

The real-data roundtrip and independent verification are recorded in
[larger NCLT workspace](../benchmarks/vector-map/nclt-large-workspace/README.md).
Run `e2e/large-workspace.spec.ts` with `CLOUDANALYZER_LARGE_WORKSPACE_BASE` set to
the previous complete workspace ZIP and `CLOUDANALYZER_LARGE_WORKSPACE_CLOUD`
set to the second-session point map. The ordinary synthetic test checks a
million-point workspace above 64 MiB and the unchanged browser-save budget.

New v3 snapshots use the existing `sha256-chunks-v1` file identity to verify every
stored member in 8 MiB SHA-256 chunks. ZIP writing checks CRC32 in 1 MiB chunks;
opening retains stored member Blob slices rather than reading each entire member
into another buffer. Legacy v1/v2 members retain their original whole-file SHA-256
verification. This bounds archive verification buffers, not total browser memory:
loaded geometry, native exports, staging and recovery still need memory. The
256 MiB limit does not guarantee a workspace fits on every device. Generated-map
review ZIP imports retain their separate 64 MiB content and smaller member limits.

After reopening, **Download original map package** retrieves the exact archived
ZIP. Saved audit controls remain disabled for the current edited workspace. Reopen
that archive to review its original map pair, or run a new source check on your
current edits. A display-preview archive continues to identify its external full
point map; archiving it does not include that missing full density or align separate
drives. Metadata-only projects keep their existing source-selection workflow.

## Validate a workspace before changing current work

Workspace ZIP imports and record-inclusive browser recovery first verify every
member and original source fingerprint, then prepare the saved map, graph and
all point/mesh records in the worker. They switch the map and graph and register
the prepared records only after all inputs pass. A malformed later PLY, invalid
graph, cancellation during preparation, or an edit to current work during staging
discards the temporary geometry and leaves current points, graph, map and cloud
Undo history available. Duplicate current cloud names are rejected before staging.

After a successful import, Undo starts from the saved state. Legacy ZIPs with
external graph inputs wait for matching files before native preparation begins.
Cancel is cooperative between parsing/indexing steps; one native call may finish
before cancellation is handled. Preparation temporarily needs memory for both
workspaces. Freed WASM allocations remain reusable in the heap; its displayed
allocated size does not shrink. This is an input/preparation transaction, not a
crash-recovery guarantee or rollback of later recomputed analyses and rendering.

The fault tests run in `workspace-transaction.spec.ts`. Opt into the real NCLT
restore using `CLOUDANALYZER_TRANSACTION_ZIP=/path/to/nclt-complete-workspace.zip`
with that test file; it reuses the previous immutable workspace and compares
exported point records, saved graph and HD map exactly without regenerating them.

## Apply and keep a reusable filter recipe

Open **Reusable filter recipe** in Filters. Choose an operation and its parameters
in the existing filter controls, then press **Add current filter step**. Steps run
in the displayed order; remove a step to change the sequence. Give it a name and
use **Save recipe** to share its JSON. **Open recipe** loads instructions for
inspection; it does not execute them or select source clouds.

Select source checkboxes, or press **Select visible clouds**, then **Run recipe on
selected clouds**. Version 1 supports voxel subsampling, minimum-distance
subsampling, octree subsampling, SOR and splat cleanup. Random sampling and CSF
remain in the individual-cloud controls. Splat cleanup requires splat attributes
in every selected source. Recipes have at most eight steps and sixteen point-cloud
sources, with at most four million loaded points per batch. The selected-source
native memory estimate is limited to 256 MiB and each canonical point-record
export to 64 MiB; these are bounds/estimates, not a process-memory guarantee.

All results are prepared before originals are hidden. Sources remain in the
workspace, and one cloud Undo step restores the whole batch; Redo restores its
results and processing records. Cancellation, a later step/source failure, or a
current-work edit before commit discards temporary results and retains original
points and the preceding Undo history. Cancellation is cooperative between native
calls and indexing steps. Only final outputs are registered; intermediate clouds
are freed as the sequence progresses. Preparation still needs extra memory.
The final step must fit the configured cloud Undo byte budget and at least one
history step. If it does not, the recipe refuses to hide sources and reports the
required estimated MiB. Increase **Memory and Undo** limits before retrying.
Existing older steps can be evicted to fit a successful new batch, as with other
cloud edits; a refused batch does not evict them.
Voxel sizes below the supported grid range for a cloud's extent are refused in
both individual filters and recipes; the status gives a safe minimum.

Each result has a **Recipe** button to download its processing record: ordered
parameters, actual WASM-build SHA-256, source name/count, and complete SHA-256
digests of the input and output native PLY exports. These exports contain current
loaded records, current coordinates and native attributes. They do not identify
unloaded source density, certify a map, or guarantee matching future kernels.
The record describes the result when generated; later point edits do not rewrite
that historical output hash.

Use **Save workspace snapshot** or record-inclusive browser recovery to keep the
results and records together. Plain metadata projects keep only source-backed
clouds; export derived point records separately or choose a workspace snapshot.
Display-preview sources keep their preview flag on recipe outputs: filtering does
not turn a preview into full-density evidence. Filters do not modify the HD map.
