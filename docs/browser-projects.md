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
