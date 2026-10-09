# Agent-driven generation of both maps

Use `start_mapping_run`, `inspect_mapping_run` and `advance_mapping_run` over
[MCP](mcp.md) to let Codex or another calling agent generate a point-cloud map
and a Lanelet2 road draft from a recording. Supply the log and **one explicit
layout hypothesis**. The agent examines candidates and chooses source ranges;
the runner binds that layout to every retained piece, executes geometry/lane
generation and returns diagnoses. The operator does not prepare candidate IDs,
per-piece lane JSON or individual processing commands.

The calling agent supplies reasoning. CloudAnalyzer does not embed an LLM or
make adoption decisions using a hidden ranking. No additional model API key is
required by the MCP server. The output is a draft with recorded unresolved
intervals and semantics, rather than a certified road network.

## Ask the agent to run

Install a current native core and register `ca mcp` with your agent. A request
can specify:

> Generate a point-cloud map and an HD road draft from `drive.mcap` into a new
> `runs/drive` directory. Use a single forward driving-lane hypothesis with
> one-way use, 40 km/h, fraction 1 and minimum profile width 2.5 m. Treat the
> source span and every traffic attribute as unverified. Use a six-attempt
> budget and keep the 90% full-input extent goal. Inspect candidate evidence,
> record your choices and retry autonomously within those fixed assumptions.
> Return the point map, editable IR, Lanelet2 and remaining review holds.

Use assumptions appropriate to the task. Direction, lane count, speed and
complete road width are not inferred from a trajectory or ground support.
There is no default legal traffic interpretation. If those assumptions cannot
be supplied, the [lane-free geometry workflow](mapping-job.md#adopt-source-geometry-while-lane-semantics-remain-unresolved)
can still preserve source curves for review.

## The loop

1. `start_mapping_run(source, out_dir, layout_hypothesis)` creates a new frozen
   job, generates the point map and corridor proposals and returns revision 0,
   a paged candidate index and decision guidance. Only raw inputs are accepted:
   MCAP, ROS1 bag or rosbag2 SQLite. Native processing and relevant recording
   readers must be installed. Initial point-map/proposal failures stay visible.
2. `inspect_mapping_run(job_dir, offset=0)` resumes saved observations without
   processing. Candidate index pages contain 16 entries. Page through them and
   issue `advance_mapping_run` inspect actions for relevant candidates.
3. An inspect action returns up to eight candidates with original-frame curves,
   surface evidence and observed stations. Previews contain up to 128 sections
   per candidate with explicit total/truncation flags. The inspection receipt
   persists. Draft ranges must have inspected endpoints; unreviewed tails stay
   unresolved. There is no automatic adoption or candidate ranking.
4. A draft action supplies **complete** include/defer decisions and reasons.
   It automatically saves source geometry, binds the fixed layout to each
   included piece, generates Lanelet2/IR/projector and reads all four source
   audits. It uses up to two shared HD attempts. Each draft is a replacement
   hypothesis; it does not append roads to an earlier map. Separate pieces stay
   disconnected. Previous maps and any explicit selection remain intact.
5. The agent reads width/structural failures, original input extent and both
   ground estimators, then decides another draft, inspects short connections,
   inspects raw-source gaps to test less point thinning, or finishes. Explicit
   connection choices spend one additional shared HD attempt.
   The runner cannot
   change the layout or lower the extent goal through an action. Invalid
   choices spend no processing attempt. Failed native stages retain their
   reason and consume their attempt. Full source support is not accuracy truth.
6. Finish with a retained audited lane candidate, or `null` if none could be
   generated. The output includes point-map/trajectory paths, HD artifact and
   audit paths when available, a compact diagnosis, fixed assumptions and the
   complete decision-history path. It never calls `select_mapping_candidate`.
   A partial export remains `draft_needs_review`; no HD output is explicitly
   `hd_unavailable`. `deployment_ready=false` remains unconditional.

Every action supplies the inspected `expected_revision` and a reason. Stale
retries are rejected before execution. Inspection and finish actions spend no
HD attempts. The run allows 128 decisions; finishing remains possible after
that action budget is exhausted. The shared HD budget is 2–8 attempts, default
6, sufficient for up to three geometry/lane pairs when every stage succeeds.
Do not shrink required lane width, lane count or retained extent to improve a
source-support score. Semantic assumptions remain fixed hypotheses throughout.

## Inputs and actions

When off-path bands overlap a path-containing band and break its unique match,
the calling agent can issue one explicit source extraction experiment:

```json
{"type": "refine", "association": "trajectory_containing"}
```

The extractor keeps only source bands containing the recorded trajectory in
each endpoint profile for matching. Every original profile, off-path band and
source-level observation is retained. Search reach, height/width thresholds,
support spacing, query budgets, point map, trajectory, initial layout and full
input denominator stay unchanged. Centre and both edges still need continuous
source support. Absent path bands, missing anchors and source gaps remain deferred;
no road junction, legal branch choice or full-width clearance is inferred.

The original `corridor-proposals.json` stays immutable, while successful refinement
saves `corridor-proposals-trajectory.json` and changes the active proposal.
`corridor_refinement` returns both reports' summaries and paths; compare the
path-associated extent and longest candidate length, not total off-path coverage.
Candidate IDs can be reused with different geometry. All current review receipts
are reset: inspect the new candidates before drafting a replacement. The history
binds each inspection to its proposal hash. Earlier geometry/lane outputs and the
selected draft remain intact; finishing with an earlier run draft is allowed.

Refinement spends no HD attempt and can be attempted once per run. Failure retains
the original active proposal and its error, so the agent can still draft or finish.
Interrupted draft or refinement actions support `resume`: completed extraction
is verified and reused, running native stages require manual inspection. A failed
extraction is not silently rerun. This action addresses geometric association;
it does not repair the point map or guarantee a continuous road across the drive.
The [two NCLT refinement runs](../../benchmarks/vector-map/nclt-path-refinement/README.md)
retain all previous exported intervals and record actual before/after maps:
102 → 106 m and 88 → 112 m, with June's longest piece increasing 60 → 66 m.

### Connect short, inspected source gaps

After an audited lane draft, the calling agent can inspect connection candidates:

```json
{"type": "inspect_connections", "candidate_id": 2, "offset": 0}
```

Use the actual shared attempt ID from your run. This first version supports one
forward, one-way driving lane per disconnected source piece, at most 256 lanes.
It considers only consecutive pieces in original drive-station order. Both the
endpoint distance and original drive gap must be at most 10 m. Native heading,
grade and endpoint checks remain active. Ambiguous native branches are withheld.
Centre and both actual boundaries must have 100% low-quantile source support;
recorded XY path samples at no more than 0.5 m spacing must lie inside the exact
connector polygon, including endpoints. Minimum distance between equal normalized
XY-arc positions on the piecewise-linear boundaries must meet the fixed width
hypothesis. This is a geometric width check, not observed physical road width.

The proposal is saved with parent, cloud, trajectory, source and native hashes.
Pages show at most eight complete candidates; overlong geometry is withheld.
Inspection costs no HD attempt and never changes an exported map. Each chosen
pair must have been seen through this run and have its own reason:

```json
{"type": "connect", "candidate_id": 2, "pairs": [{"from": 3, "to": 6, "reason": "Inspected consecutive gap; exact geometry encloses the recorded path and retains fixed minimum width. Test both ground estimators and reopened graph."}]}
```

Use lane IDs from the observation, rather than corridor proposal IDs. The runner
retains every existing lane, boundary, rule and coordinate value; it adds only
explicitly chosen connections in a new candidate. It rechecks the proposal and
source hashes, requires the actual added boundaries to match the inspected
geometry, audits IR and reopened OSM with **both** ground estimators, and requires
100% support and supported endpoints on all new centre/left/right traces.
Incomplete audits, changed source or altered OSM reachability withhold publication.
Failed attempts and any completed failed audit remain recorded; the previous
candidate and selection stay intact. Existing lanes' estimator disagreement is
still visible and is not repaired by adding supported connections.

`routes.before` and `routes.after` report directed lane chains, component count,
and original drive-station spans, including connector gaps. Native `turn_direction`
tags classify geometric headings; they remain unverified manoeuvre hypotheses.
These spans are not
physical centreline length or certified legal routes. Read the affected local
chain and global longest route separately. Original corridor extraction length,
station dispositions and the 90% extent goal stay unchanged: connectors do not
silently turn unresolved source intervals into generated source corridors.
Finish may return the new connected draft or an earlier retained draft.
Interrupted inspection/connection actions support `resume` without replaying
completed work. One connection attempt uses the existing shared HD budget. Every connection
trial starts from its inspected audited parent, including connected or gap-patched
drafts. Existing lanes and directed edges remain fixed; supply only new pairs.
Only consecutive original-station pieces with open endpoints are eligible.
All retained source samples must keep their support in both estimators and both
saved formats; new connectors require complete support. Saved connection checks
and four full audits remain available even when the trial fails.

The [NCLT connection evidence](../../benchmarks/vector-map/nclt-route-connections/README.md)
records a local chain changing 6 → 14 m in April and 4 → 22 m in June, with
components 12 → 11 and 14 → 12. Global longest routes remain 34 m and 66 m;
these connections do not make the full recordings routable.

After a local gap patch, inspect connections on the combined candidate rather
than its isolated addition draft. Supply only new seen pairs; retained connectors
and endpoint links are inherited. A new trial consumes one remaining shared
attempt, so reserve it when starting the run. Finished runs and budgets remain
fixed. The [NCLT patch connections](../../benchmarks/vector-map/nclt-patch-connections/README.md)
join June's repaired 22–28 m interval to the retained 14–18 m piece across a
4 m gap, yielding one 14 m local chain and components 13 → 12. April offers
no connection and keeps the patch. Global longest spans remain 34/66 m.

### Extend only the HD connection region after a local patch

Default connection inspection keeps new connectors inside the local point-update
box. When a supported link from a repair lane to the existing map crosses that
box, its rejection includes `geometry_bounds_xy`. The agent can explicitly inspect
a separate **HD-only** envelope on the combined patched candidate:

```json
{"type": "inspect_connection_region", "candidate_id": 4, "bounds_xy": [9, -7, 22, 5], "offset": 0}
```

Use observed IDs and coordinates. The envelope must contain the frozen point box,
have positive sides of at most 20 m, and finite coordinates. It is a closed XY
envelope at all heights, without voxel alignment. This is available only for a
combined local-retry patch, including its later connected descendants. It does
not regenerate or expand point replacement, change density or poses, move
existing HD geometry, alter the layout, or infer permitted turns.

Only station-consecutive links touching a repair lane/connector are offered.
Every offered center and boundary curve stays inside the HD envelope; unrelated
original-lane links remain held. Heading, path containment, minimum width,
ambiguity and original directed-edge checks remain in effect. Preview pages
contain eight candidates and a hash-pinned region artifact. Different envelopes
produce different cached previews, and pairs must be seen in the latest exact
preview before adoption:

```json
{"type": "connect", "candidate_id": 4, "pairs": [{"from": 51, "to": 9, "reason": "Inspected the HD-only envelope and exact geometry; test both estimators and OSM topology"}]}
```

Inspection spends no attempt; `connect` still spends one transferred HD attempt.
Plan for it when starting the run: geometry/lane draft, optional height edit,
combined patch and connection all share the child allocation. The native export
must preserve every original entity/edge, exactly match inspected boundaries,
and retain support/failure locations on old traces. New connectors need full
support under both estimators in IR and reopened OSM, against the actual complete
local point map, including unchanged outside points. Failed links retain their
checks/audits and the prior draft; completed previews support `resume`.

Finish explicitly after comparison. Outputs include `hd_connection_proposal`
with both region boundaries, alongside source checks, so the HD-only scope is
reviewable. Connector station spans improve graph reachability; they do not
increase source-corridor extent or establish road identity or legal routing.

The [NCLT HD-region trial](../../benchmarks/vector-map/nclt-hd-connection-region/README.md)
explicitly connects June's new lane to an existing lane across the point-box
boundary: 51 → 54 → 9 spans 22–36 m. Point files and retained HD entities remain
fixed, components fall 13 → 12, and all four audits pass. Source extent stays
118 m and the global longest route stays 66 m; full-drive continuity is unmet.

### Investigate missing intervals and retry point generation

After an audited lane draft, inspect missing source intervals:

```json
{"type": "inspect_gaps", "candidate_id": 3, "offset": 0}
```

Pages contain eight gaps with stable IDs, source-profile reasons and bounded
neighborhood observations. The runner freshly decodes the **original recording**,
aligns returns from the original retained keyframes using the corrected graph,
and records every decoded frame hash. Counts, occupied XY cells and local height
quantiles compare raw returns with the fused point map before a density trial.
These are descriptive observations: repeated returns are not independent ground
evidence, and a missing trajectory band is not a proven thinning failure. The
inspection includes nearby profile context and explicit truncation totals.
It is limited to 4096 frames and five million matched raw returns. Larger logs
need a smaller input recording. Inspection spends no HD attempt.

For an inspected thinning hypothesis, explicitly choose gaps and resolutions:

```json
{"type": "retry_pointcloud", "candidate_id": 3, "gap_ids": [1, 4], "options": {"scan_voxel_m": 0.2, "map_voxel_m": 0.1}}
```

Use the IDs actually returned by your run. Scan voxels can decrease from the
initial 0.4 m down to 0.1 m; fused-map voxels from 0.2 m down to 0.05 m. At least
one must change. Corrected motion, retained keyframes, dynamic-filter policy,
native implementation, source thresholds, association, lane layout, extent goal
and original drive denominator stay fixed. Numerical pose roundtrips are checked;
the child keeps the exact original trajectory and graph bytes. Filtering results
can change with denser returns even though the filter policy remains fixed.
This first retry does not repair motion, choose another ground level, include
excluded non-keyframes or invent missing raw returns.

One root retry transfers **all remaining shared HD attempts** to a child run
at `pointcloud-retry/`, with at least two required. The root cannot spend that
allocation again, even after a failed trial; children cannot retry recursively.
The transfer is journaled before processing and never silently refunded.
The response includes the new point map and child run. Use the child's `job_dir`
and revision, inspect its fresh proposal IDs, explicitly draft the fixed layout,
and optionally test inspected connections within its budget. Nothing adopts
parent decisions or ranks child geometry automatically.

Then call the root with the child's audited attempt ID:

```json
{"type": "compare_retry", "candidate_id": 2}
```

The retained comparison includes **both gained and lost original source
intervals**, point counts, exported extent, all four source-audit totals/protocols
and actual graph route station spans. All extraction/quality protocols and the
original denominator must match. Longer routes and more points alone do not
establish accuracy or improved road semantics. Response lists are bounded;
the hashed report retains complete intervals and chains. No automatic adoption
occurs. Finish the child with its chosen candidate, review the root comparison,
then explicitly deliver that child's point map and HD draft:

```json
{"type": "finish_retry", "candidate_id": 2}
```

Alternatively use the root's normal `finish` with its retained baseline candidate
when the trial worsens, fails or does not justify replacement. Earlier point and
HD maps stay unchanged. Completed retry stages support `resume` without another
allocation; running native stages and interrupted raw extraction require retained
state inspection before manual recovery. Failed retry directories are preserved.
An interrupted inspection can also be abandoned with normal `finish` to deliver
an already audited baseline; finishing never restarts the failed processing stage.
The [two NCLT density trials](../../benchmarks/vector-map/nclt-pointcloud-retry/README.md)
retained their original maps: more points recovered some intervals but lost more,
and global route spans decreased. The saved comparison exposes that outcome.

### Reuse inspected non-keyframe observations

New jobs hash their original odometry report and timestamped trajectory before
point-map correction. After `inspect_gaps`, inspect excluded source frames:

```json
{"type": "inspect_unused_frames", "candidate_id": 3, "offset": 0}
```

Pages contain eight frames in original recording order, their gap-return counts,
pose hypotheses, both neighboring scan checks, eligibility and explicit holds.
The full hashed report retains all observations. Original odometry must have one
finite pose per freshly decoded frame, with matching strictly increasing timestamps.
Older jobs without hashed original motion need a new run for this trial; existing
density trials and retained outputs remain available.

For a frame between two original corrected keyframes, the runner interpolates
their corrections to original odometry: translation linearly and rotation with
SLERP. It applies that correction to the frame's original pose. It never extrapolates
past either end or across brackets longer than 4 s or 3 m. All existing corrected
poses stay fixed. At least three raw returns must fall within 0.75 m of a saved
missing-interval profile; these counts are descriptive, not proof of coherent ground.

The hypothesis must agree with **both** original bracket scans. Each check uses
at most 6000 deterministic sample points, 0.75 m nearest-neighbor reach and at least
65% overlap both before and after ICP. Trimmed ICP uses 60% overlap and 30 iterations,
must converge, have RMS at most 0.5 m without worsening, and suggest at most 0.25 m
translation at the sensor origin and 1.5 degrees rotation. The ICP correction is
**not applied**. Checks establish geometric consistency with correlated observations
from this log; they do not prove independent pose accuracy or resolve weakly
constrained scene geometry. Thresholds are fixed in the reported protocol.

Inspection is limited to 4096 raw frames, 256 excluded frames and five million
total raw returns. Larger recordings need a smaller input. Completed inspection
is cached without another registration pass. Reader/processing failures remain
visible, and the root can finish with its retained audited baseline.

Explicitly adopt IDs that were returned as eligible through this run:

```json
{"type": "retry_frames", "candidate_id": 3, "frame_ids": [6, 12, 51]}
```

Use actual IDs from the observations, not this example. Choose 1–64 distinct frames;
withheld, duplicate or unseen IDs cannot consume a retry allocation. The trial
keeps original scan/map thinning, dynamic-filter policy, native binary, original
corrected keyframes, source thresholds, association, initial lane layout and extent
goal fixed. Filtering outcomes can change with the added observations. Density and
unused-frame strategies share **one** root retry and the transferred HD budget;
children cannot retry recursively. There is no automatic frame ranking or adoption.

The child fuses exactly the original retained frames plus the explicitly chosen
new IDs, retaining original frame names and checked return-byte hashes. Its
`fusion_graph`, `fusion_trajectory` and `fusion_scans` artifacts describe the actual
expanded fusion. Its `graph` and `trajectory` remain the **byte-identical original
reference**, used for HD extraction and the original station denominator. They
do not describe the expanded fusion set. The report verifies every original and
added pose survived native roundtrips and records all excluded raw IDs.

Inspect fresh child proposals and draft the fixed layout there; examine connections
within the child budget. Then use the root's existing `compare_retry` to read
actual gained/lost source intervals, all four audits and global routes. Finish the
child, then explicitly `finish_retry` to deliver its point/HD pair, or normal
`finish` to retain the baseline. More observing frames alone do not establish an
improvement. Failed fusion and interrupted completed stages retain the same
allocation and cannot silently replay processing.

The [NCLT unused-frame trials](../../benchmarks/vector-map/nclt-unused-frames/README.md)
retain both actual point/HD comparisons: April's partial improvement was explicitly
adopted, while June's baseline was kept because its trial shortened the longest
route and increased fragmentation despite gaining net source extent.

### Update points only inside an inspected region

After inspecting the root's gap pages and unused-frame pages, preview an explicit
spatial update instead of replacing the whole point map:

```json
{"type": "inspect_local_points", "candidate_id": 3, "gap_ids": [1], "bounds_xy": [14, -3, 23, 6]}
```

Use observed IDs and coordinates from your run. Each chosen gap must have an
inspected trajectory profile inside the box. Bounds expand outwards to the
original map voxel grid; each effective side must be positive and at most 20 m.
This is an XY column **at all heights**, with lower edges included and upper
edges excluded. It can include another pass through the same physical location.
The preview returns effective bounds, inside/outside counts, an outside-record
hash and eligible frames with at least three raw returns inside the box.

Choose only eligible frame IDs already inspected through this run, and pass the
exact `local_point_observation.file` artifact as `preview_file`:

```json
{"type": "retry_local_frames", "candidate_id": 3, "frame_ids": [199, 217], "preview_file": {"path": "/absolute/run/local-points-03-REQUEST_HASH.json", "sha256": "RETURNED_SHA256", "bytes": 1234}}
```

The runner still generates a **full fusion candidate** from the original frames
plus these explicit unused frames. Only candidate records inside the frozen box
replace baseline records. Every outside vertex record keeps its exact XYZ,
scalar attributes and relative order. The PLY vertex count/header and global
record indices can change. This version accepts canonical binary little-endian
PLY with double XYZ and up to 16 float attributes, at most two million points
and 128 MB per map. It does not establish a local processing speedup.

Before generating child HD proposals, four full audits require the unchanged
baseline lanes to retain their supported samples, endpoints and failure locations
against the local point map. A regression stops the trial and retains its map,
full fusion candidate, report, checks and audits for inspection. The root can
still `finish` with its original point/HD pair.

This strategy shares the **one root retry** with density and full-map unused-frame
trials. It transfers all remaining HD attempts and requires at least three:
geometry, lane draft and combined gap patch. Inspect fresh child proposals, draft
only chosen missing intervals and use `inspect_patch` / `patch_gaps` below. Added
HD boundaries must stay inside the closed effective XY box; patch gaps must be
among those chosen in the local preview. Existing lanes and directed edges remain
fixed. Default connection previews withhold new connectors leaving the box;
`inspect_connection_region` explicitly inspects a separate HD-only envelope.
The root
accepts `compare_retry` and `finish_retry` only for a combined patch (or its
subsequent connection candidate), preserving the original HD map.

Finish requires an explicit choice; successful audits never adopt a trial.
Returned artifacts include `pointcloud_trial_local_report`,
`pointcloud_trial_local_checks` and `pointcloud_trial_local_audits` after a completed
local source gate, including a rejected gate. Completed interrupted stages resume
without another allocation. Failed stages retain their shared allocation.

The [NCLT local point trials](../../benchmarks/vector-map/nclt-local-points/README.md)
verify exact outside records in both scenes. April adds a 4 m HD interval while
retaining existing geometry/routes; June rejects a new failure on a retained lane
and returns the unchanged original pair. Point counts and source support do not
establish independent accuracy or road semantics.

### Increase density only inside an inspected region

When an inspected gap suggests testing less thinning, use the original retained
frames for a local density trial. No unused-frame inspection or added frame is
needed. First inspect the root's relevant gap pages, then preview the box:

```json
{"type": "inspect_local_density", "candidate_id": 3, "gap_ids": [3], "bounds_xy": [14, -3, 23, 6]}
```

Pass the exact returned `local_point_observation.file` to the density retry:

```json
{"type": "retry_local_density", "candidate_id": 3, "preview_file": {"path": "/absolute/run/local-points-03-REQUEST_HASH.json", "sha256": "RETURNED_SHA256", "bytes": 1234}, "options": {"scan_voxel_m": 0.2, "map_voxel_m": 0.1}}
```

Use actual coordinates, IDs and artifact fields from your observations. Options
follow the existing density bounds: scan voxels at least 0.1 m, map voxels at
least 0.05 m, neither larger than the baseline, and at least one strictly smaller.
Invalid options or a preview from the unused-frame strategy spend no allocation.
The box stays aligned to the **original** voxel grid and remains frozen while
candidate density changes. Preview hashes include the strategy, so identical
boxes for density and unused-frame trials are distinct inspected decisions.

The full density candidate still uses exactly the original retained frame IDs
and corrected poses. It adds no non-keyframes, changes no motion and retains the
original graph/trajectory bytes. Only inside candidate records replace baseline
points; outside attributes/relative record order are exact. Finer returns can
change dynamic-filter outcomes even though the filter policy is fixed.

The same four retained-HD audits, one-root-retry allocation, minimum three
remaining HD attempts, combined local HD patch requirement, explicit comparison
and adoption, and failed-trial baseline delivery apply. Density, unused-frame and
both local strategies share the same single retry; they cannot be chained or
automatically reallocated within one run. Source observations do not establish
that thinning caused the gap or that more points improve accuracy.

The [NCLT local density trials](../../benchmarks/vector-map/nclt-local-density/README.md)
keep all original frames and outside records. April explicitly delivers a 4 m
HD addition with no lost interval. June's existing-HD audits pass, but its new
boundary has a height mismatch; the agent stops before a known-held patch and
returns the original pair. The full-drive extent goals remain unmet.

### Adjust an interior height on a local HD addition

An isolated gap-only draft in a **local** retry child can have enough returns
but a boundary height mismatch. Inspect it before the combined patch:

```json
{"type": "inspect_heights", "candidate_id": 2, "offset": 0}
```

The returned `height_observation` pages contain at most eight affected interior
vertices, their XYZ and observations from both estimators in IR/reopened OSM.
Only vertices next to observed boundary height mismatches, inside the frozen
point-update box and on unshared boundaries are offered. Fully supported drafts
have no offered edits. Original failures and source curves remain saved.

An agent can propose an explicit bounded Z hypothesis using the exact returned
`height_observation.file`. Use actual observed IDs and **zero-based** indices:

```json
{"type": "edit_heights", "candidate_id": 2, "preview_file": {"path": "/absolute/child/heights-02.json", "sha256": "RETURNED_SHA256", "bytes": 1234}, "edits": [{"boundary_id": 2, "vertex_index": 1, "delta_z_m": 0.075, "reason": "Test a bounded height hypothesis at the inspected mismatch; require both estimators"}]}
```

Choose 1–16 distinct seen vertices, each with a finite, nonzero delta of at most
0.1 m in magnitude and an individual reason. Boundary XY, endpoints, unchosen Z,
IDs, metadata, lane semantics, directed relations, projector and point-map files
stay fixed. The original map and addition draft are untouched. Derived centerlines
and sample locations may move because native resampling uses 3D arc length.
Acceptance conservatively requires unchanged trace sample counts, **full support
at every addition sample and endpoint in all four audits**, complete reports at
unchanged protocols, and matching IR/reopened OSM geometry and routes.

One height trial is allowed per child. It spends one transferred HD attempt and
requires at least two remaining attempts so the combined patch still fits.
Invalid or unseen edits spend none. A failed hypothesis keeps its trial geometry,
checks and audits without publishing an edited draft; finish the root baseline
when it cannot be adopted. Interrupted completed exports resume without another
export or attempt. A successful edit produces a new candidate ID: inspect and
patch **that** candidate, then compare and explicitly adopt the resulting pair.
The combined patch still independently checks every retained and new lane.

There is no automatic fitting, estimator preference, endpoint/XY adjustment,
retained-root geometry edit, recursive height search or independent accuracy claim.

The [NCLT height trials](../../benchmarks/vector-map/nclt-hd-height/README.md)
repair June's one new-boundary mismatch with a +0.075 m interior Z hypothesis,
then deliver a 6 m combined HD addition without lost intervals. April has no
affected vertices and delivers its unchanged 4 m addition. Both preserve outside
point records and original HD entities; full-drive extent goals remain unmet.

### Repair HD gaps while retaining the existing map

After a point-fusion retry, the child can add source-supported missing HD
intervals without replacing the root's retained geometry. This first patch is
limited to one forward one-way lane per piece, 1–32 new lanes and 256 total lanes.
The point map remains the **complete fusion trial** for ordinary density and
unused-frame retries. `retry_local_frames` and `retry_local_density` use the spatial
replacement above.

Inspect the root's actual baseline gap pages, then the child's fresh source
sections. Draft **only** additions inside the gaps you choose. Each added lane
must lie entirely in the union of selected missing station intervals and cannot
overlap any retained lane or connector interval. A whole-drive replacement draft
is not a patch. Source thresholds, original reference poses/denominator, extent
goal, minimum widths and lane hypotheses remain fixed.

In the child, preview that audited gap-only draft:

```json
{"type": "inspect_patch", "candidate_id": 2, "gap_ids": [3], "offset": 0}
```

Use current IDs; the gap IDs belong to the **root's** audited baseline. Preview
pages contain eight geometric endpoint pairs, with `baseline:ID` and `addition:ID`
references and endpoint distances. New links require original station adjacency
and both boundary endpoints within 0.01 m in XYZ, matching the OSM reader's
inference tolerance. The preview holds nonadjacent or ambiguous links. It is
cached for this addition candidate and exact gap choice; a different gap selection
needs a new draft. No HD attempt is spent by preview.

OSM readers can infer links at coincident endpoints even without an explicit IR
relation. Inspect and explicitly adopt **every** offered pair, or revise the draft;
the runner does not silently accept these links. Pairs must have been returned
through this child run. Use `pairs: []` when no endpoint pairs were offered:

```json
{
  "type": "patch_gaps",
  "candidate_id": 2,
  "gap_ids": [3],
  "pairs": [{"from": "baseline:6", "to": "addition:3", "reason": "Inspected station-adjacent coincident boundary endpoints"}]
}
```

The patch spends **one** remaining transferred HD attempt. It retains original
lane/boundary IDs, geometry, lane semantics, metadata and directed connections;
new entities receive unused IDs. Only the explicitly inspected endpoint pairs
can add edges, without moving or extending original boundaries. Native export
and OSM reload must preserve geometry, projection, turn labels and the expected
route graph. These geometric relations do not establish permitted manoeuvres.

All retained and new traces are re-audited against the child point map, using
both ground estimators for IR and reopened OSM at unchanged protocols. Every new
trace and join endpoint needs full source support. Retained traces must keep
their sample counts, supported endpoints and nonworsening support/failure totals.
Complete failure-location reports also ensure **no previously supported sample
becomes a new failure**, even if net totals improve elsewhere. Limited audits or
failure-location previews hold the patch. Earlier source failures remain visible;
they are not treated as repaired or as a quality pass.

Failed patches retain full audits/checks and spend their original allocation;
they cannot publish an audited replacement or be replayed as a new trial. The
root can still explicitly finish with its baseline pair. Completed interrupted
patches resume without another native export or allocation. A child can make
only one patch; patches do not retry recursively.

After a successful patch, use the root's `compare_retry` on the **patched** child
candidate. Inspect gained/lost intervals, four audits and global route spans,
finish the child with that candidate, then explicitly `finish_retry`, or retain
the root baseline. Existing routes are preserved, but filling an isolated source
interval need not make the whole drive connected. Original-station coverage is
not unique physical road length, independent accuracy or deployment readiness.

The [NCLT local patches](../../benchmarks/vector-map/nclt-partial-repair/README.md)
retain actual before/addition/combined maps and four audits. They add 4/6 m with
zero lost source intervals, preserve original geometry/edges and the 34/66 m
longest routes, and explicitly deliver both partial pairs. The new intervals
remain isolated; whole-drive connectivity and the original 90% goal remain unmet.

`layout_hypothesis` / `layout.json`:

```json
{
  "boundary_policy": "source_span_hypothesis",
  "reason": "Explicit unverified test layout; legal traffic semantics remain unresolved",
  "speed_limit_kmh": 40,
  "lanes": [
    {"direction": "forward", "kind": "driving", "one_way": true, "fraction": 1, "minimum_width_m": 2.5}
  ]
}
```

Lanes are left-to-right along original input stations; fractions sum to one.
Only driving lanes are accepted because the source-audit protocol assesses that
kind. [Lane export](mapping-job.md#export-explicit-lane-hypotheses-from-adopted-geometry)
describes supported values, profile-width constraints and virtual boundaries.
The initial policy file is hashed; editing it invalidates inspection/advancement.

Example action objects, using IDs read from the current run rather than fixed
dataset-specific IDs:

```json
{"type": "inspect", "candidate_ids": [12, 13]}
```

```json
{"type": "draft", "decisions": [{"candidate_id": 12, "action": "include", "reason": "Observed source band follows the input path; complete width remains unresolved"}]}
```

```json
{"type": "finish", "candidate_id": 2}
```

All tool calls use the `job_dir` and current revision returned by the runner.
The draft decisions use proposal IDs; finish uses the shared job attempt ID.
Geometry curve IDs and per-piece lane specifications are supplied automatically.

## Resume and inspect results

An agent can stop between actions and resume with `inspect_mapping_run`. Each
draft journals planned geometry/lane attempt IDs before processing. If a call
is interrupted after saving geometry or completing lane generation, inspect
the state and issue `{"type":"resume"}` with its current revision. Completed
stages are reused after checking their identity and hashes, without spending
attempts twice. Running native stages are not replayed blindly.

Hard process termination can leave `.mapping-lock` or `.mapping-run-lock` files.
Inspect the retained job/action and confirm its process has stopped before
manual recovery. Do not remove a live lock or retry startup into an existing
directory. Initial preparation interrupted before `run.json` exists is visible
through `inspect_mapping_job`; retain it and start a new run. This runner does
not reconstruct missing odometry/point-map work after a hard crash.

Response histories show the last eight actions, with explicit totals/limits.
Compact diagnoses retain full sample totals and at most 16 lane summaries per
audit; failed sample locations are omitted from these responses. Use
`diagnose_mapping_candidate` for the existing bounded problem-location preview,
or read the hashed complete source audit. The history file retains observations,
all action reasons, failed outcomes and the output choice. Full original station
dispositions remain in each draft report.

## CLI equivalents

```sh
ca mapping-run drive.mcap --out runs/drive --layout layout.json
ca mapping-run-inspect runs/drive
ca mapping-run-advance runs/drive --action action.json --revision 0 --reason "Inspect current source candidates"
```

These CLI commands execute one stage/agent decision and return JSON; the calling
agent continues the loop through MCP. Starting alone does not finish HD mapping.
A failed draft exits nonzero while preserving its run state and prior artifacts.
See the [two NCLT calling-agent runs](../../benchmarks/vector-map/nclt-agent-run/README.md)
for full live decision traces, unchanged layout priors and original-input extent.
