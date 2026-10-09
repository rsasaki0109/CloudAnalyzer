# Agent-controlled point-cloud and HD-map generation

The agent's task is to turn a raw recording into both a point-cloud map and a
Lanelet2 road draft, then use evidence to decide the next trial. Register
CloudAnalyzer with `claude mcp add cloudanalyzer -- ca mcp`. The calling agent
supplies the decisions; these tools execute them and persist their reasons.
They do not embed an LLM, require a model API key or choose legal road semantics.

## Tool sequence

1. `start_mapping_job(source, out_dir)` reads an MCAP, ROS1 bag or rosbag2 SQLite
   file. LiDAR odometry, loop closure, IMU gravity when present, and optional dynamic
   removal produce a point-cloud map, corrected KITTI trajectory and pose graph.
   The new directory contains `job.json` and separate odometry/correction reports.
2. `inspect_mapping_job(job_dir)` reads the small persisted job without loading
   point clouds. Inspect processing reports, artifact hashes and remaining attempts.
3. `propose_mapping_corridors(job_dir)` searches the frozen source for surface
   corridor candidates **without lane count, width, direction or speed priors**.
   Read `inspect_mapping_corridors(job_dir)` for the paged index, then use
   `candidate_id` to inspect a candidate's cross sections and edge evidence.
   It saves candidate geometry, all profiles and unresolved intervals as a hashed
   `corridor-proposals.json`. This one-time stage is cached and does not spend an
   HD attempt. Search options are frozen; changing them needs a new job.
4. `generate_mapping_candidate(job_dir, road_options, reason)` generates one HD
   hypothesis from the same frozen map and trajectory. It audits both editable IR
   and reopened OSM, retaining validation/export issues and source-review lanes.
   Read the result before trying another fit. Failed attempts retain their errors
   and do not replace earlier maps or the selected draft.
   Use `diagnose_mapping_candidate(job_dir, candidate_id)` to read the saved
   per-lane/trace evidence: height mismatches, insufficient returns, endpoint
   holds, audit completeness and retained extent. It verifies source/artifact
   hashes, does not run native processing and does not spend an attempt.
   New jobs retain bounded `problems` with lane/curve/reason, oriented stations,
   original-frame XYZ and `source_heights_m` parallel to those points. A null
   source height means insufficient support for that report's estimator. Older jobs explicitly
   report `problems_available=false`; limited previews set `problems_limited=true`.
   Neither absent nor truncated locations mean the unshown source is supported.
   New jobs save both the original quantile evidence and a separate
   `ground_consensus` report for each artifact. Read their `ground_estimator`
   metadata and compare outcomes; diagnosis keeps both and flags disagreement.
5. `select_mapping_candidate(job_dir, candidate_id, reason)` records the chosen
   draft and justification. It requires nonempty lanes, complete source checks,
   zero structural errors, unchanged inputs/output hashes and the job's minimum
   retained fraction of the corrected trajectory (default 90%). Low source support
   remains a visible hold. Both saved estimators must have complete audits;
   `source_quality_passed` requires both to pass. A successful selection always
   has `deployment_ready: false`.

Road options require explicit `forward_lanes`, `backward_lanes`,
`left_hand_traffic`, `lane_width` and `speed_limit`. Other fitting options from
[`build_vector_map`](vectormap-build.md) are available. Existing/reference maps and
projection options are excluded in this initial raw-recording workflow: its
coordinates are local SLAM metres, not a known georeferenced frame. Both map
outputs share that frame. Each build's `report.json` records effective options.

When lane semantics are unresolved, use `generate_mapping_geometry` after step 3
to adopt inspected source curves as an editable **geometry** draft. This lane-free
branch shares the attempt budget with lane hypotheses and preserves existing
selection. It does not qualify for `select_mapping_candidate` or a source pass.

Compare `extraction.generated_length` and `trajectory_length` alongside source
support. A shorter map, fewer lanes or narrower road can improve a support score
without improving the requested map. Lane count, width, traffic direction, speed
and the trajectory's lane identity are hypotheses. Source checks do not establish
legal lanes, missing equipment, clearance or independent survey accuracy. The
point-cloud map itself is marked `generated_unverified`; processing convergence
and loop counts are evidence of execution, not a survey-accuracy certificate.
Set `minimum_retained_fraction` when starting the job (`--minimum-retained-fraction`
in the CLI). It is an explicit extent goal, not the source-support threshold.
Candidate summaries expose the generated/trajectory lengths, fraction and extent
hold; a highly supported short fragment cannot satisfy a whole-drive goal.

Diagnosis separates observed failure types from possible causes. A height mismatch
can come from wrong XY, another surface level or generated Z; it is not a command
to move Z. Insufficient returns can reflect sparse/occluded coverage or a wrong
lane/trace hypothesis; they do not prove that a road is absent. Inspect source
footprints and assumed-width anchors before selecting another fitting option.
Totals count samples per oriented lane/trace, including shared boundaries checked
from both lanes; they are not unique source points or fractions of road length.
Editable and reopened OSM evidence are kept separately, with discrepancies visible.
Location previews cap at 4,096 failed samples and 256 intervals independently of
the 100,000-sample audit budget. Full summary counts remain authoritative when a
preview is limited. Ground-height observations use the explicitly recorded
estimator, not certified road heights; another level and wrong XY can still match.

For an explicitly reasoned estimator experiment, `road_options.local_ground_height`
uses the lowest spatially supported local height layer instead of the median of
longitudinal cross-section bins for seed-road Z. It defaults to false. The layer
requires three occupied 0.2 m XY cells spanning at least 0.01 m² in a 0.15 m height
window within 0.75 m. Each cell supplies its lowest return and the median of those
votes supplies Z. Missing coherent support defers sections; structural/extent
gates still apply. Its earlier quantile implementation did not improve NCLT.
Compare both audits and the [recorded experiments](../../benchmarks/vector-map/nclt-agentic-mapping/README.md)
before choosing an estimator; a better support score is not independent accuracy.

## Surface corridors before road semantics

Corridor search uses the same low-layer spatial estimator as the experimental
height stage, applied to 0.5 m lateral bins in longitudinal windows of ±2 m.
Profiles are spaced at 2 m **on the original corrected input trajectory's XY arc
length**, including its exact endpoint. This station length can differ from the
HD builder's resampled/smoothed trajectory length; do not mix their denominators.
The symmetric search reach defaults to 8 m and rounds outward to complete 0.5 m
bins. At least 1 m of adjacent supported bin-centre span is required, with no
adjacent height step exceeding 0.08 m. Narrower/sparse features can be missed.

A band must match the source low layer under the path within 0.3 m plus 0.12 times
its lateral distance. This rejects distinctly elevated/lower bands; it is a
geometric guard, not a legal grade or verified ground-level rule. All observed
bands, their `path_level_supported` flags and the per-profile source anchor remain
in the file. Missing anchors defer candidates rather than borrowing sensor height.

Adjacent profiles connect only when bands overlap uniquely with bounded motion
and vertical change. Branching bands are not silently chosen by width/proximity.
Every connected interval checks its centre and both edges against the low-layer
source at ≤0.5 m spacing with both endpoints included. No width prior fills absent
profiles or unsupported curves. These three curves do not certify the full-width
interior or obstacle clearance. Returning passes remain separate proposals.

Each edge is `curb_profile`, `support_gap`, `height_discontinuity` or `search_limit`.
The curb check requires bounded raised returns in two outside bins and nearby
inside support. Only candidates with two curb-like edges in **every** cross section
receive `curb_width_range_m`; even that estimate has 0.5 m bin-centre quantization
and requires review. Other `support_span_m` values measure observed source extent
and do not establish complete road/path width or pavement identity. No lane count,
traffic direction, speed, legal use or equipment is inferred; every candidate has
`review_required=true` and the report has `deployment_ready=false`.

Coverage fields measure unions of **input-station intervals**, not the sum of
overlapping candidates or unique roads. `with_candidate_station_length_m` and
`without_candidate_station_length_m` partition the full requested path; deferred
intervals preserve missing surface, missing anchors, unmatched bands, source gaps,
branches, unusable headings and processing limits. Ambiguous station length may
overlap candidate coverage when a separate uniquely matched band exists. Neither
coverage figure certifies which band is the semantic road used by the recording.
`trajectory_covered_station_length_m` counts connected intervals whose band
contains the recorded path in both endpoint cross sections; intermediate path
positions and full-width interiors are not separately certified by that figure.

Search is bounded to 2,048 profiles, two million profile-return examinations and
100,000 longitudinal support samples. An oversized trajectory fails before
resampling. Reached processing budgets produce `limited=true` and explicit
deferred remainder, preserving partial results. The full hashed report retains
all bounded geometry. Inspection pages 16 candidates at a time via `next_offset`;
individual geometry previews cap at 128 sections and interval previews at 128
entries, with explicit truncation flags. Inspection verifies source and report
hashes without needing the original native binary. Failed stages retain errors
and leave point maps, HD attempts and selected drafts intact.

The index also exposes up to 16 `curb_width_profile_hints` from individual
paired-curb profiles, with a total count and truncation flag. These local position
and span observations can exist even when no continuous candidate has two curb
profiles throughout. Inspect competing hints at the same station; a single
paired-curb profile does not establish complete corridor width or road identity.

Surface proposals help the agent inspect position/width hypotheses before making
an HD candidate. They are not automatically adopted as Lanelet2 geometry or traffic
semantics. If only point-coverage edges are available, retain the unresolved width;
do not narrow a road solely to increase source support.

The [two bundled NCLT runs](../../benchmarks/vector-map/nclt-corridors/README.md)
produced 52 and 77 candidates, with 5 and 7 local paired-curb hints. Neither has
a candidate with paired-curb evidence throughout; complete widths remain unresolved.
The recorded figures, hashes and deferred intervals make this limit inspectable.

Attempts default to four and are bounded to eight. A job pins its recording,
generated inputs and native extension by SHA-256. Changing them requires a new
job. Candidates also retain output hashes, decision reasons and audit reports.
Calls refuse existing output directories. Job writes use atomic replacement and
an exclusive writer lock. Inspection works while processing; another mutation
reports busy. Ordinary errors and PyO3 panics are recorded; user interrupts are
recorded and re-raised. This version does not resume an interrupted SLAM stage.
After a hard process exit, inspect the job and confirm the process is gone before
removing a leftover `.mapping-lock`; retain the directory and start a new job.

## Adopt source geometry while lane semantics remain unresolved

`generate_mapping_geometry(job_dir, decisions, reason)` records the calling
agent's corridor choices and writes `geometry-NN/vector_map.json` plus
`geometry-NN/report.json`. Each decision has `candidate_id`, `action` (`include`
or `defer`) and a nonempty `reason`. Optional `from_m`/`to_m` must be actual
observed stations in that candidate. The defaults use its complete interval.
No extrapolation is allowed. Included ranges cannot overlap input stations;
resolve competing bands explicitly. Disjoint ranges of the same candidate are
allowed. Adjacent pieces remain separate, without joining gaps or smoothing.

Each included piece saves the exact source centre, left and right curves as
`vectormap-ir` boundaries of kind `other`, with attributes identifying the source
role, proposal candidate and review hold. All section-level edge evidence and
original stations remain in the report. These are reference curves, not lane
markings, complete road borders or a drivable interior. Open the editable IR in
the map editor and enable virtual lines to view/edit the reference geometry.

Lane count, direction, speed and complete width remain unresolved; no lanes,
roads, connectivity or traffic rules are invented. This stage intentionally does
not publish Lanelet2 OSM: the current reader drops standalone unknown ways on
reload, so their geometry would not survive. The IR is the editing artifact.
Resolve lane hypotheses separately before lane-map export.

The report partitions **every input station interval** into `included_geometry`,
`source_deferred` (no connected proposal), `agent_deferred` (all offered candidates
explicitly deferred) or `not_reviewed`. It records available/deferred candidates,
source failure reasons and ambiguity, including ambiguity beside an included
band. An omitted decision does not become acceptance or a reviewed rejection.
No physical absence is inferred from a deferred band. Included/unresolved unions
sum to the original input trajectory extent. The fraction and the job's extent
goal are exposed; a partial draft does not satisfy the full-drive goal simply
because its local curves have source support.

One call consumes one shared HD attempt; native failures also retain their reason
and consume that attempt. Invalid decisions consume none. Up to 256 decisions
are accepted per call. Frozen inputs, native binary, proposal and output hashes
are checked; publishing is atomic. Existing lane drafts and selection remain
intact. Native IR loading normalizes the curves and checks structural errors;
intentionally unused reference boundaries can produce informational
`unused_boundary` issues. Keep these reference curves rather than applying their
suggested removal. A zero-error, zero-lane draft has `source_quality_passed=false`
and `deployment_ready=false`.

`inspect_mapping_geometry(job_dir, candidate_id, offset=0)` verifies saved input,
proposal and output hashes without native processing. It pages 16 segments, caps
each section preview at 128 and the station-disposition preview at 128, with
explicit totals/truncation flags. The hashed report retains all geometry and
decisions. Failed/running geometry attempts remain inspectable.

For example, after inspecting April NCLT candidate 17, save `decisions.json`:

```json
[
  {
    "candidate_id": 17,
    "action": "include",
    "reason": "Retain the observed source curves as review geometry; width and lane semantics remain unresolved"
  }
]
```

```sh
ca mapping-geometry runs/nclt-agent --decisions decisions.json --reason "Build a geometry draft before lane hypotheses"
ca mapping-geometry-inspect runs/nclt-agent --candidate 1
```

The `--candidate` is the job's shared attempt ID, not a proposal ID. Full-extent
decisions and actual native-reloaded outputs are recorded in the
[NCLT geometry runs](../../benchmarks/vector-map/nclt-geometry/README.md).

## CLI equivalent

```sh
ca mapping-start web/public/samples/nclt-2012-04-29.mcap --out runs/nclt-agent
ca mapping-status runs/nclt-agent
ca mapping-corridors runs/nclt-agent
ca mapping-corridors-inspect runs/nclt-agent
ca mapping-corridors-inspect runs/nclt-agent --candidate 1
```

Write an explicit hypothesis to `roads.json`:

```json
{
  "forward_lanes": 1,
  "backward_lanes": 1,
  "left_hand_traffic": false,
  "lane_width": 3.5,
  "speed_limit": 40.0
}
```

```sh
ca mapping-candidate runs/nclt-agent --options roads.json --reason "Baseline road hypothesis; inspect source support and retained extent"
ca mapping-status runs/nclt-agent
ca mapping-diagnose runs/nclt-agent --candidate 1
ca mapping-select runs/nclt-agent --candidate 1 --reason "Retain this draft with its unresolved source and traffic-rule holds"
```

These numeric priors demonstrate the interface; they are not verified NCLT road
attributes. Failed generation returns JSON evidence and a nonzero CLI exit.
Selection paths identify the Lanelet2 OSM, projector YAML, editable map and report;
`pointcloud.files` identifies the point map, trajectory and graph. Pass the point
map and candidate directory to [`ca web-view`](web-view.md) or `view_link` to inspect
them together. Review the [NCLT run](../../benchmarks/vector-map/nclt-agentic-mapping/README.md)
for real source/results, retained extent and unresolved quality.
