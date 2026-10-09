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
   unresolved. Source-proposal inspection preserves original candidate IDs and does not adopt a draft automatically.
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

## Deliver a portable review package

After `finish` or `finish_retry`, call
`export_mapping_run(finished_job_dir, bundle_path, attribution)` over MCP, or:

```sh
ca mapping-run-export runs/drive --out drive-review.zip \
  --attribution "Source dataset, license terms and required credit"
ca mapping-bundle-inspect drive-review.zip
```

Supply the actual source-data license and attribution. The exporter copies the
**exact delivered pair**, including an adopted child's maps when applicable.
The ZIP contains the complete point map, graph, trajectory, editable IR,
Lanelet2, projector, fixed layout, retained proposal, final four source audits,
final checks/comparisons and root/owner decision records. Duplicate files are
stored once. `manifest.json` identifies each role with a relative member path,
SHA-256 and byte count; `review` keeps the output diagnosis and adoption decision.
Read the full audits for problem locations omitted from the compact diagnosis.

Use `inspect_mapping_bundle(bundle_path)` on the receiving machine to check every
member before opening the map. It needs neither the original directories/logs
nor a native core. Its module uses only Python's standard library; an existing
CloudAnalyzer installation exposes the CLI above. After verification, extract
the ZIP and use the manifest's `map` member with CloudCompare or CloudAnalyzer;
load `hd_editable_map` for editing, or `hd_map` plus `hd_projector` for Lanelet2.
All geometric coordinates and source-audit thresholds are retained.

In CloudAnalyzer Web, choose **Generated maps → Open generated maps…** and
select the ZIP. The browser verifies every member before loading the point map
and editable HD map together, using their original coordinates. Opening replaces
the current HD map and adds the point cloud. The initial solid display makes
low-intensity points visible; the original fields remain available for coloring.
The source-extent goal and dataset credit remain visible in the generated-map panel.

Choose any of the four **Saved source audit** protocols to inspect frozen
editable-IR/reopened-OSM results from each estimator. Existing source-review
buttons and problem intervals focus the original failed samples over the point
cloud. These are saved checks of the full exported pair; your loading limit and
view budget can show fewer points. Editing/replacing the map or changing/removing
the imported point source disables saved-audit display and clears its overlays.
Use a fresh source check for edited geometry, or reopen the unchanged exported pair.

Browser review accepts at most **64 MiB uncompressed**, 128 artifact members and
a 10 MiB manifest. It supports stored/deflated ZIP members, including the
exporter's ZIP64 local headers, with bounded decompression and full SHA-256
verification. Larger packages use CLI verification and individual-file loading.
No mapping generation or attempt spending occurs when opening the package.

For a large delivered point map, call
`export_mapping_preview(finished_job_dir, bundle_path, attribution,
max_preview_points=200000)` over MCP, or:

```sh
ca mapping-run-preview runs/drive --out drive-display.zip \
  --max-preview-points 200000 \
  --attribution "Source dataset, license terms and required credit"
ca mapping-bundle-inspect drive-display.zip
```

This streams the canonical binary PLY (double XYZ, optional float attributes) and
retains every kth complete record, starting at record zero. The stride is
`ceil(source_count / max_preview_points)`. Kept coordinates and attributes retain
their original bytes, with no quantization or frame change. The full original
point map stays outside the ZIP. The v2 manifest names the subset `preview_map`,
records the original map's path/hash/size, point counts and stride, and retains
the exact HD files and all four saved audits of the **original full source**.
Sampling is for display; it makes no claim about feature coverage. A source
already within the cap uses the exact full-map v1 format instead.

Open this ZIP through the same browser control. The panel distinguishes loaded
preview records from original source points and lets you inspect all four saved
audits. New source-coverage checks are disabled for the imported display preview;
load and select the original full point map for a fresh check. Projects/sessions
retain this preview flag. Standalone extracted or re-exported PLY files do not
carry that application flag; preserve the ZIP/manifest to retain provenance.
Preview export defaults to a 64 MiB **uncompressed** package limit; large HD/audit
evidence can still exceed it. Use a smaller point cap, or the full-map CLI path
when evidence alone does not fit. It spends no mapping attempts and changes no
original map or saved source check.

Export spends no attempts, changes no run revision and refuses existing output
files. Total **uncompressed** content is bounded by `max_bundle_bytes` / CLI
`--max-bundle-bytes` (default 1 GiB; allowed 1024 bytes–4 GiB), with at most 128
artifact members and a 10 MiB manifest. The inspector rejects unlisted,
duplicate, unsafe, encrypted or symlink members and changed hashes without
extracting files. A matching manifest establishes byte integrity, not a trusted
signature or independent map accuracy.

This package supports portable review. Original histories and archived manifests
retain historical paths for provenance; referenced raw logs, native binaries and
prior-run dependencies are excluded. It is **not a resumable mapping job**.
Existing source failures, incomplete extent and unverified traffic semantics
remain visible; export does not promote the draft to deployment readiness.

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

## Preserve retained HD neighborhoods during density repair

An ordinary box replacement can alter the ground estimate around an existing lane,
even when it inserts more points. `inspect_protected_density` uses the same seen gap
IDs and explicit box as `inspect_local_density`, with a separate frozen preview:

```json
{"type":"inspect_protected_density","candidate_id":1,"gap_ids":[1],"bounds_xy":[7,5,15,10]}
```

Inspect the returned protected/editable counts, then use the exact preview artifact
with `retry_local_density`. This is an explicit alternative; ordinary replacement
and unused-frame previews retain their existing behavior. The new preview consumes
no attempts and has a distinct request hash; density retry retains the single child,
transferred HD budget, bounded strictly reduced thinning and frozen original poses.

The protection region is the union of each retained lane's **convex hull** of its
left/right boundaries and any explicit centerline, expanded by the maximum ground
query radius in the four complete saved audits (**0.75 m** in the current native
protocol, plus a **1 µm** numerical margin). Derived centerlines lie in these hulls.
The region covers every continuous retained trace's source-query disk, including
existing holds and exported geometry; it is deliberately more conservative than
protecting individual sampled disks. All heights are included.

Inside protected regions, existing PLY records stay byte-identical and all fusion
candidate points are excluded. Only the unprotected part of the explicit box is
replaced. Every retained record, attribute and relative order is checked after
writing/reloading the map; four native audits still require complete evidence and
no newly failing retained source locations. A fully protected box or no unprotected
candidate points cannot become a ready repair. No support threshold or HD geometry
is modified to obtain a pass.

Protection can exclude useful new evidence near held lanes or bend interiors. Point
count increases do not prove improved ground/accuracy or an eligible HD addition.
Continue the child, inspect fresh corridor evidence, and adopt only a compared,
audited combined patch. Otherwise finish the prior baseline. The full fusion
candidate is still generated, so this is not a local processing speedup.

[The NCLT protected-density packet](../../benchmarks/vector-map/nclt-protected-density/README.md)
uses the same June box/resolutions as the rejected preceding trial. It retains old
nearby query evidence and passes all four retained-HD checks. The inspected missing
8–14 m source interval still has no fresh candidate to add, so the final pair stays
unchanged: **118 m** source extent and **66 m** longest route. Native-fixture tests
separately verify a protected point update followed by a successfully adopted gap patch.

## Continue from a delivered map

`inspect_mapping_run` resumes an unfinished decision loop. To repair another region
**after finishing**, call the new MCP entry with a distinct output directory, an
explicit **3..8** HD attempt budget and a reason:

```python
continue_mapping_run(
    finished_job_dir="/absolute/accepted-run",
    out_dir="/absolute/next-region",
    max_attempts=6,
    reason="Repair the next inspected gap while retaining the accepted map pair",
)
```

Startup reuses the exact delivered point cloud and audited HD map as candidate **1**,
without odometry, fusion, proposal extraction or HD generation. The seed spends
zero new attempts. Source, original retained frames/poses, native version, fixed
layout, audit protocols and full-input extent goal remain unchanged. A finished
run with no HD output, added frames, incomplete audits, structural errors or
changed artifact hashes cannot seed this first version. Existing source holds
are allowed and remain explicit; continuation is not evidence of accuracy.

Inspect candidate 1's gaps afresh, preview an explicit new box with
`inspect_local_density`, and call `retry_local_density` with that preview and
reduced thinning. Continue the returned child, inspect its fresh proposals,
draft **only missing ranges**, inspect/execute a combined gap patch, compare
against candidate 1, and explicitly finish/adopt. Existing lane IDs, geometry,
metadata and directed edges remain the baseline; outside point records retain
bytes, attributes and order. Explicit source-checked connections can also extend
the seed. The old point-update box is archived in `continuation.json`, so it does
not constrain the next repair's box.

Full-replacement drafts and frame-adoption trials are unavailable in the new
root. If a trial fails or is worse, `finish` candidate **1** returns the exact prior
map pair and the rejected trial evidence. The new family uses one transferred
budget; continuation neither reopens a finished run nor restores a spent budget.
`session_spent_attempts` and `cumulative_spent_attempts` count actual HD attempts,
including failures, across roots/children, excluding inherited seeds and unused
allocations. Point-fusion trials remain separately recorded in their stages.

Thinning settings describe the **last full fusion**, rather than uniform resolution
of a hybrid map. A new density trial must strictly reduce them within the existing
scan **0.1 m** / map **0.05 m** floors. The runner does not reset these settings to
allow repeated same-resolution updates; this limits repeated density repairs.

Inputs are immutable **references**, not copies: preserve earlier directories.
`continuation.json` hashes the previous finished run/job, exact map pair, point
composition evidence, native/layout/source, and all saved decision/audit artifacts.
These hashes are verified before subsequent actions and propagated into retry
children. A change to a prior decision invalidates continuation rather than silently
changing its baseline.

[The NCLT continuation packet](../../benchmarks/vector-map/nclt-repair-continuation/README.md)
records a live MCP session from the previously accepted June map. The second,
disjoint point-density trial is **rejected** because retained source support regresses;
the final output preserves the earlier local lane/connector and map pair exactly.
The native fixture separately verifies two successive accepted gap patches.
The real-log session adds no new HD extent or route connectivity.

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

## Add missing HD intervals without rebuilding the point map

Use `repair_hd` when the retained point map already contains an observed corridor
that was omitted from the accepted HD draft. Inspect the root candidate's gaps
first. Each returned gap now includes `retained_hd_occupancy`: a source-extent gap
can already contain a retained connector. `unoccupied_intervals` identifies the
parts available for an addition; it does not establish source support.

```json
{"type":"repair_hd","candidate_id":1,"gap_ids":[9]}
```

This creates an `hd-repair` child using the **exact retained point-cloud files and
source proposal**. It performs no odometry, fusion or corridor re-extraction and
spends no HD attempt during preparation. At least three remaining shared attempts
are required for geometry, lane generation and a combined gap patch. HD-only,
density and frame trials share one root repair allocation. Transferred attempts
cannot also be spent by the root; failed attempts count and unused allocations
are not restored. A continuation can start this child with its explicit new budget.

Inspect the frozen proposal in the child, then `draft` only observed missing
station ranges using the unchanged layout. Proposal refinement is unavailable in
this child. `inspect_patch` and `patch_gaps` preserve the root's geometry, IDs,
metadata and directed edges. Added lanes must lie entirely within the selected
root gaps and must not overlap any retained lane or connector. All new traces
require full support in the four complete audits, with no newly failing retained
sample locations. An isolated draft cannot be adopted as a replacement map.

Use `compare_retry`, finish the child with the combined candidate, then explicitly
`finish_retry` in the root. The comparison records exact point-artifact identity,
gained and lost source extent and route spans. Final `hd_repair_decision` records
whether the HD patch was adopted and that `pointcloud_regenerated` is false.
Finishing the root baseline retains the preceding pair after a rejected trial.

[The NCLT HD-only packet](../../benchmarks/vector-map/nclt-hd-only-repair/README.md)
records a live MCP session that adds 118–120 m without changing the 646,309-point
map. Source extent increases from 118 to 120 m with no lost intervals. An earlier
184–188 m addition passes its isolated source audits but is rejected before the
combined patch because retained connector 45 already occupies it. The runner
then uses only the nonoverlapping interval. No supported connector is offered
for the new fragment; the longest route remains 66 m. This is a recorded agent
session with prior diagnostic probes, rather than an independent autonomy or
accuracy benchmark. Existing source disagreements and unmet full-drive extent
remain visible.

## Inspect source-supported HD repair intervals before drafting

After inspecting the root's gap pages, request a bounded preflight:

```json
{"type":"inspect_hd_plan","candidate_id":1,"gap_ids":[14,19,29],"offset":0}
```

The response pages up to eight adjacent observed-station intervals. It excludes
intervals that cannot satisfy the fixed minimum lane width or overlap retained
lanes/connectors. Ordering uses the number of coincident retained endpoint links,
then original station and source candidate ID; it is a geometric ordering, not a
quality score. Source ambiguity and exact endpoint-link distances remain visible.
Currently this inspection supports one forward, one-way lane occupying the entire
observed source span.

For each interval, the native quantile and lowest-supported-layer estimators check
its two observed side curves and the derived centre trace on the exact retained
point map. `reference_traces_fully_supported` requires complete inspections and
every sample, including endpoints, supported by **both** estimators. Failure totals,
protocols and full bounded source-problem locations are saved as hashed reports.

These checks use a transient reference-trace carrier. They call neither the lane
builder nor the OSM exporter, change no point/HD artifacts, and spend no shared HD
generation attempt. There are at most 256 intervals in an index and a conservative
100,000-sample limit per page per estimator. A completed page is immutable and
reused; an interrupted page resumes completed interval inspections. This separates
source inspection from generation while keeping its work bounded and visible.

Use the results to choose root gaps for `repair_hd`. Inspect the frozen proposal in
the returned child and explicitly draft the chosen ranges. Separate adjacent ranges
can be drafted together; inspect and adopt all their coincident patch endpoint links.
The actual lane export, OSM reload, four complete audits and retained-map checks
remain required. Preflight support cannot establish a final patch pass, full road
width, independent accuracy or legal routing. No automatic adoption is introduced.

[The NCLT preflight packet](../../benchmarks/vector-map/nclt-hd-repair-preflight/README.md)
records paging all 31 source gaps, selecting nine with observed candidates and free
HD extent, and inspecting 11 intervals after width/occupancy filtering. Nine intervals
have source holds; two pass both estimators. One actual geometry/lane draft and one
combined patch adopt 222–226 m, increasing source extent from 120 to 124 m and the
local route from 220–222 to 220–226 m. All point files and prior geometry/edges are
retained; the global longest route remains 66 m. The calling agent selects ranges
from MCP evidence without separate offline lane-generation probes in this session.
This is a recorded workflow, not an independent model evaluation or accuracy test.

## Apply an explicit supported plan

After reading `inspect_hd_plan` pages, a calling MCP agent can execute its chosen
intervals with one tool call:

```json
{
  "job_dir": "/maps/new-repair-session",
  "plan_files": ["/maps/new-repair-session/inspected-plan-page.json"],
  "interval_ids": [3, 7],
  "connect_endpoints": true,
  "reason": "Add only the inspected intervals supported by both estimators",
  "expected_revision": 5
}
```

Pass this object to `apply_supported_hd_plan`. The paths must be the exact saved
artifacts returned by inspections on this root, not copies. Supply one to eight
distinct pages from the same frozen index and one to eight distinct global
interval IDs. Every selected interval must have complete reference support from
both estimators, unambiguous source geometry and endpoints visible in the bounded
candidate preview. Unsupported, unseen or overlapping choices are rejected before
a child or generation attempt is created.

The tool prepares an unchanged-point-map HD child, inspects its source proposal,
drafts only the chosen ranges, inspects all offered exact geometric endpoint
pairs, creates a combined patch, finishes the audited child and compares it with
the retained baseline. This uses at most three shared HD generation attempts.
`connect_endpoints: true` explicitly permits all inspected coincident pairs;
`false` requires an isolated addition and returns a hold if a join would be needed.
It establishes neither traffic permission nor lane semantics.

`ready_for_agent_decision` returns the comparison, retention checks, child output
and hashed application policy. The root remains unfinished. Read the results and
use `advance_mapping_run` with `finish_retry` to adopt explicitly, or finish the
root baseline to retain the prior map. Holds and failed attempts remain recorded.
The helper neither ranks candidates nor calls a model or adopts the parent.

After interruption, inspect the root revision and repeat the same choices and
reason. Completed stages are verified and reused; interrupted ordinary actions
use their existing resume path. Failed native attempts are never silently rerun.
Changed policies, stale revisions, source/artifact changes or unrelated manual
child decisions require inspection and manual continuation. When the root is
finished, its output retains the application policy for portable review and later
continuation. Keep the original source and prior run directories accessible.

[The two-drive NCLT application packet](../../benchmarks/vector-map/nclt-hd-plan-application/README.md)
records live MCP application with explicit choices and a separate root adoption.
June replays a known 4 m extension; April adds a freshly inspected contiguous
14 m gap. Each uses three actual HD attempts with unchanged point/proposal files,
zero lost extent and four final audits. A repeated June call spends no additional
attempt; an unsupported April choice is rejected before child allocation.
