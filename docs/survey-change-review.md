# Review changes between two passes

Use **Pose graph → Analyse the map → Compare parts (M3C2)** after joining and
aligning two recordings. The split node separates the first pass from the
second; the default for joined recordings is the second session's first node.
This also works with two passes through the same place in one recording.

The existing comparison builds voxel maps from keyframes within 50 m of the
other path. A zero map-voxel setting uses 0.3 m for this comparison. M3C2 uses
normal radius 1 m, projection radius 0.5 m, depth 2 m and core spacing 0.5 m.
Significant same-sign distances of at least 0.3 m are grouped within 1 m, with
at least eight core points per candidate.

## Inspect alignment and observations

**Changes · survey review** shows the number of cross-part loop constraints,
the RMS and maximum of their weighted residuals (√χ²), and measured/unmeasured
M3C2 core counts. Sequential odometry and loops wholly within one pass are
excluded from this constraint summary. Missing/non-finite residuals are
unavailable, not zero. The residuals are information weighted, **not metres**;
low residuals do not independently verify registration accuracy.

Select a candidate to frame its original-coordinate bounds with surrounding
context. Switch **Review view** between **Before**, **After** and **Difference**
to inspect exactly the same camera view. This changes visibility of the three
comparison clouds only. Other loaded clouds remain under the cloud list's
visibility controls; hide unrelated overlays if they obscure the comparison.
Inspect stable surfaces before deciding whether a candidate is an actual change
or remaining misalignment.

Candidates are largest first, with 50 per page. **Previous candidates** and
**Next candidates** expose the complete list, including candidates beyond the
first 50. Selecting another candidate changes the active editor; changing pages
keeps the current selection and its note.

Use **Candidate filter** for all candidates, those with a decision (including
Deferred and Insufficient observations), or one specific decision state. Counts
and pages follow the filter while candidate numbers retain their original rank.
**Next unreviewed** selects and frames the next unreviewed candidate in original
rank order, crossing page boundaries and wrapping to the beginning. It switches
the filter to Unreviewed and excludes the currently selected candidate; it is
disabled when no other unreviewed candidate remains.

Applying a decision can remove the selected row from the current filter. The
editor keeps that candidate selected so you can finish its note without an
unexpected jump. **Show selected candidate** returns to its page in the complete
list without moving the camera. An empty filtered list does not clear the
selection or imply there are no changes. Filtering never changes analysis,
decisions or geometry. JSON exports and global decision totals always include
all candidates. Filters are presentation settings, not stored in review JSON;
resuming saved decisions refreshes the active filter against the restored states.

## Record and save judgments

For each candidate choose **Actual change**, **Misalignment**,
**Insufficient observations**, **Deferred**, or leave it **Unreviewed**. Notes
accept up to 2,000 characters. These are human judgments; they do not alter the
point clouds, graph or Lanelet2 map.

**Save review JSON** writes `survey-change-review.json` with schema
`cloudanalyzer.survey_change_review.v1`. It contains the graph/session names,
split and selected frame IDs, comparison cloud names and counts, processing
settings, alignment/measurement diagnostics, all candidate bounds/centroids,
signed mean distances, decisions, notes, timestamps and decision totals.
Cloud IDs are browser-session identifiers. New exports additionally carry an
`identity` using the `sha256-ply-graph-v1` protocol. It binds complete binary PLY
exports of the three comparison clouds, graph arrays and plane/gravity counts,
processing settings, source names/counts, diagnostics and candidate geometry.
PLY exports are hashed sequentially in 1 MiB chunks; computing the identity adds
export/hashing time and requires memory for one serialized cloud at a time, in
addition to the existing working data. No sampled display hash substitutes for
these exports. The final digest excludes runtime cloud IDs, analysis/export
timestamps and human decisions so a matching rerun can resume a saved review.
Resumable exports require Web Crypto in a secure browser context (HTTPS or
localhost); otherwise comparison and historical JSON export remain available
without an identity. Exact serialized content and graph array bytes must match; different builds or
point order can prevent resumption even when a scene looks similar.

The JSON contains neither the recordings nor a self-contained reproducible dataset.
Its digest is a matching aid, not proof of reviewer identity or file authenticity.

Save before rerunning, refreshing or leaving the page. Reviews are not included
in browser project saves/autosave. A successful new comparison starts a fresh
review and hides the earlier comparison clouds, which remain in the cloud list.
A failed rerun leaves the earlier record available as historical evidence.
Changing graph poses/constraints, closing/replacing the graph or modifying/removing
one of the three comparison clouds marks the review historical and locks further
judgments/view switching. Its original judgments and diagnostics remain exportable
with the reason they became historical. Presentation changes do not invalidate it.

## Reopen and resume

Expand **Saved survey reviews** and choose **Open review JSON…**. Files up to
16 MiB with at most 100,000 candidates are accepted. The loader checks schema,
finite geometry, ranks, counts, decisions, note limits, timestamps and diagnostic
consistency before replacing the displayed historical file. Invalid files leave
the earlier file and current decisions intact. Closing a file also cancels its
pending read. The historical list supports 50 candidates per page and renders
notes as text.

Opening a file does not apply decisions or load clouds. You can inspect a saved
review without a graph open. Original v1 files without an identity remain
readable as history but cannot resume decisions.

To continue editing, run **Compare parts (M3C2)** on the same inputs with the same
poses/constraints and settings, then open the saved JSON. When content identity
and analysis evidence match, **Resume saved decisions** becomes available.
Clicking it replaces **all** current candidate decisions and notes, preserving
the saved review timestamps. It keeps the current comparison's analysis time,
cloud IDs, point clouds, graph and camera. Opening or resuming does not rerun
M3C2. Matching uses the complete evidence, never candidate rank or filenames alone.

A changed input, setting, graph or candidate geometry prevents resumption.
Historical reviews (including those invalidated by closing/changing a graph)
cannot resume decisions. The earlier judgments stay available for reference.
This is an explicit JSON save/open workflow; browser projects/autosave still do
not contain these reviews.

## Inspect related map lanes

Open a Lanelet2 `.osm` or vector-map IR `.json` in the **Vector map** panel.
Under the selected change candidate, expand **Related map lanes**. Confirm that
the map and comparison use the same Cartesian metre frame; no coordinate
alignment, CRS transformation or unit conversion is inferred or performed.

The check intersects the candidate's 3D bounding box with lane surface triangles.
The triangles use the same normalized XY arc-length boundary resampling as the
viewer, with linearly interpolated Z. This is a geometric strip approximation,
not a source audit or proof that a lane changed. The full candidate box can cover
areas between disconnected change points. Lane semantics, legality, clearance
and equipment associations are not checked.

**XY margin m** (default 0) and **Height margin m** (default 1) expand the box;
both accept 0–10 m. The height test helps separate stacked roads and checks
sloping surfaces locally. Larger margins can include additional nearby surfaces.
**Inspect lane ID** selects and frames an existing lane using the map editor.
It does not edit map geometry, lane review states or survey judgments.

Each selection checks at most 100,000 lanes, 200,000 source boundary vertices and
200,000 strip triangles. Counts distinguish checked lanes, invalid geometry and
lanes left unchecked by limits. The first 50 matching IDs are shown, with a
notice when additional matches are omitted. No matches among checked surfaces
cannot establish that no map update is needed.

Opening/replacing/editing the map clears these hints and requires coordinate
confirmation again, including when lane IDs are reused. A fresh point-cloud
comparison also clears confirmation. A historical comparison cannot inspect
current lane links. Existing point-cloud decisions remain unchanged by map edits.
The geometric hint list is transient and is not included in survey JSON,
resumed judgments or browser projects. Explicit follow-up records can be saved
separately as described below.

## Inspect individual changed cores

Each of the first 50 related lanes also shows **supported / candidate cores**.
The worker reads the candidate's `change_object` labels and original double XYZ
from the comparison cloud. It checks XY inclusion in the resampled lane strip
and absolute Z distance from its locally interpolated surface, using **Height
margin m**. **XY margin m** affects box links only; it does not widen the lane
for this count. Shared triangle edges count a core once.

**Box overlap only** means the count completed with zero supported cores. The
lane remains inspectable and recordable. **Unavailable** means missing/stale
labels, invalid geometry or a processing limit, never an inferred zero. Limits
are 2,000,000 source cores, 100,000 cores per candidate, 50 lanes, 200,000 boundary
vertices and 10,000,000 core/triangle tests across one request. An oversized lane
is not partially counted. Candidate counts and bounds must match the current
review before its points can be used.

**Show changed cores** frames the candidate: cyan marks cores supporting that
lane; amber marks its other changed cores. The preview samples at most 10,000
points of each class; displayed counts always use the complete candidate.
**Clear core overlay** removes these markers. Selection, map and comparison
changes also clear them, and late asynchronous results cannot restore an old
overlay or count. The original map, graph and clouds are unchanged.

These are geometric inspection cues. Nonzero support does not establish a
physical change or a map update; zero support does not establish no update is
needed. The live hints and preview settings are transient. Explicitly recording
a lane saves a separate core-count snapshot as described below; point positions,
masks and preview settings are not embedded in either JSON format.

## Record lane follow-up tasks

In **Related map lanes**, choose **Record lane ID** for an overlapping lane.
This explicitly creates a **Needs review** task in **Lane follow-up tasks**.
Recording the same candidate/lane again while its source remains current selects
the existing record, preserving its saved decision and note.

Choose **Needs review**, **Update needed**, **No update needed** or **Deferred**,
enter a task note (up to 2,000 characters), and click **Save task decision**.
These task decisions are separate from point-cloud judgments and native lane
reviews. They do not modify geometry or perform a map update. **Inspect task
lane** selects the lane only while the source snapshot remains current.

Each task owns copies of the original candidate (including its survey decision,
note and timestamp), selected lane boundaries, overlap margins and checked/
unavailable counts. It references the survey comparison identity and saves a
SHA-256 of the complete native vector-map JSON at recording time when Web Crypto
is available. The map JSON itself and original recordings are not embedded.
The hash identifies an exact serialized snapshot, not a reviewer or authenticated
file. A missing hash is explicitly stored as null.

New records also carry `coreSupport`: a `xy-lane-z-v1` protocol identifier,
complete/unavailable status, candidate core total, supported count (null when
unavailable), height margin in metres and unavailable reason. Recording re-reads
the original float64 candidate cores for that lane; it does not copy sampled
preview points or assume a pending live count is zero. Its XY inclusion is
strictly within the saved lane; the box's XY margin does not widen it. Source
comparison/map identities and copied geometry bind this count to the same task
snapshot. This is saved evidence, not independent verification of a physical
change or an authenticated measurement.

If core collection fails or exceeds its limits, a task can still record human
follow-up with an explicit unavailable reason. If its map, candidate, selection
or margins change during collection, recording is aborted. Existing duplicate
records keep their original evidence. Later decisions, notes and source changes
also preserve the saved count. Queue and historical file views display it.
The file remains schema v1 with an additive `coreSupport` field. Older records
with an absent/null field are readable as **Core support not recorded**, never
as zero. Imports validate protocol, integer counts, candidate total, height
margin and complete/unavailable consistency; invalid files leave the earlier
history untouched. Older app versions may ignore the new field.

Editing a source candidate's judgment/note, invalidating or rerunning the survey
comparison, or editing/replacing the map makes affected tasks **historical**.
Their saved states, notes and geometry remain exportable. Historical tasks cannot
change decisions or select a current lane, even when IDs are reused or an edit
is undone. Merely changing the selected candidate, filter or overlap margins
does not change an already recorded snapshot. Pending recording is aborted when
its source or selection changes before it finishes.

**Save lane tasks JSON** writes `survey-lane-tasks.json`, schema
`cloudanalyzer.survey_lane_tasks.v1`, with all current and historical queue records
and their status totals. Lists page at 50 records; totals include historical
records. Limits are 1,000 tasks, 100,000 copied boundary vertices and 16 MiB of
exported JSON. Exceeding a limit leaves the queue/saved decision unchanged.
Save before leaving the page: this queue is not part of browser project autosave.

**Open lane tasks JSON…** validates a saved file and opens it as a separate,
read-only historical list. It never merges into the live queue, changes a map,
or resolves current lane IDs from a saved file. An invalid file leaves the
previously displayed history and live queue untouched. Notes render as text.

## Interpretation limits

No candidates does not mean no scene change: unmeasured areas, insignificant
changes and clusters smaller than eight core points are outside the list.
Measured-core fraction is not full scene completeness. Signed distances follow
reference normals and cannot by themselves label added/removed objects. This
comparison is directional; newly visible surfaces may require a reverse
comparison. This first workflow does not automatically classify causes, repair
SLAM or update Lanelet2 elements.

## Real-data verification

The [NCLT April/June workflow packet](../benchmarks/vector-map/nclt-survey-change-review/README.md)
records a complete comparison, same-frame draft generation, Lanelet2 round trip
and saved follow-up, plus an identical-recording control. An independent PLY
check verifies all candidate objects and counts individual changed cores near
each box-linked lane. The packet documents unsupported box links, source-quality
holds and the absence of independent physical-change labels or manual timings.
