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
3. `generate_mapping_candidate(job_dir, road_options, reason)` generates one HD
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
   source height means fewer than three nearby returns. Older jobs explicitly
   report `problems_available=false`; limited previews set `problems_limited=true`.
   Neither absent nor truncated locations mean the unshown source is supported.
4. `select_mapping_candidate(job_dir, candidate_id, reason)` records the chosen
   draft and justification. It requires nonempty lanes, complete source checks,
   zero structural errors, unchanged inputs/output hashes and the job's minimum
   retained fraction of the corrected trajectory (default 90%). Low source support
   remains a visible hold. A successful selection always has `deployment_ready: false`.

Road options require explicit `forward_lanes`, `backward_lanes`,
`left_hand_traffic`, `lane_width` and `speed_limit`. Other fitting options from
[`build_vector_map`](vectormap-build.md) are available. Existing/reference maps and
projection options are excluded in this initial raw-recording workflow: its
coordinates are local SLAM metres, not a known georeferenced frame. Both map
outputs share that frame. Each build's `report.json` records effective options.

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
preview is limited. Ground-height observations are local low-return quantiles,
not certified road heights; another level and wrong XY can still match them.

For an explicitly reasoned estimator experiment, `road_options.local_ground_height`
uses local low returns under the trajectory instead of the median of longitudinal
cross-section bins for seed-road Z. It defaults to false. Missing local returns
defer sections, and structural/extent gates still apply. This did **not** improve
the bundled NCLT result; see the recorded experiment before choosing it.

Attempts default to four and are bounded to eight. A job pins its recording,
generated inputs and native extension by SHA-256. Changing them requires a new
job. Candidates also retain output hashes, decision reasons and audit reports.
Calls refuse existing output directories. Job writes use atomic replacement and
an exclusive writer lock. Inspection works while processing; another mutation
reports busy. Ordinary errors and PyO3 panics are recorded; user interrupts are
recorded and re-raised. This version does not resume an interrupted SLAM stage.
After a hard process exit, inspect the job and confirm the process is gone before
removing a leftover `.mapping-lock`; retain the directory and start a new job.

## CLI equivalent

```sh
ca mapping-start web/public/samples/nclt-2012-04-29.mcap --out runs/nclt-agent
ca mapping-status runs/nclt-agent
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
