# Compare mapping motion against a supplied reference

`ca mapping-trajectory-evaluate` and the MCP `evaluate_mapping_trajectory` tool
compare original odometry and corrected mapping poses on the same timestamped
reference samples. They report improvements **and regressions**, without
generating maps, spending attempts, selecting a candidate or changing quality
status. This supplies a diagnostic for the agent's next decision.

```bash
ca mapping-trajectory-evaluate /data/run \
  --reference /data/reference-sensor.tum \
  --provenance /data/reference-provenance.json \
  --alignment-prefix-fraction 0.3 \
  --out /data/comparisons/run-trajectory.json
```

`reference-provenance.json` must contain exactly these fields:

```json
{
  "source": "Dataset URL, sequence and reference preparation method",
  "license": "Reference data license",
  "frame": "Reference converted to the scan sensor origin, xyz metres",
  "time_basis": "Same scan clock in seconds; state any explicit conversion",
  "used_for_generation": false
}
```

For MCP, supply that object as `reference_provenance` and the new output file
as `report_path`. Parent directories must already exist. Existing reports are
never replaced, including a competing report created during evaluation. Keep
the report outside the mapping job. The file has schema
`cloudanalyzer.mapping_trajectory_comparison.v1`, input hashes, both full metric
results and matched samples. The tool returns metrics and the report hash,
omitting the bulky matched trajectories/error series from the response. No
thresholds are calibrated by this command, so it emits no quality pass/fail gate.

## Binding and comparison protocol

- Use the job that **owns the point map being evaluated**. A finished root's
  `output.candidate_job_dir` may identify a repair child; a root inspection alone
  does not imply that its point-map files are the delivered pair.
- Jobs need saved, hashed `pointcloud.source_motion.trajectory`, a timestamped
  TUM/CSV original odometry, and a corrected KITTI trajectory agreeing with the
  frozen g2o graph. Graph node IDs must be ordered original odometry row IDs.
  Legacy jobs without saved source motion are refused rather than inventing
  timestamps. No missing source motion is reconstructed automatically.
- References are TUM/CSV in metres with strictly increasing, finite timestamps
  in seconds. Convert the **sensor origin**, clock, units and axis convention
  explicitly before evaluation. Each trajectory has at most 4,096 poses;
  trajectory/graph/reference files are limited to 16 MiB each. Crop a full
  reference to the recording interval before calling the tool.
- Reference positions are linearly interpolated at retained original scan
  timestamps; optional quaternions use SLERP. Both reference brackets must be
  within `max_time_delta` (default 0.05 s, allowed `(0, 1]`). Exact timestamps
  match directly. No extrapolation outside reference coverage occurs.
- Both estimates use exactly the same supported frame IDs. At least three
  matches and a noncollinear reference are required. Coverage is the matched
  count divided by **retained corrected poses**, not the size of a denser
  reference. Retained/matched duration and original frame IDs are also saved.
- Each estimate fits its **own** rigid SE(3) alignment to the same reference
  positions, with no scale fitting. Alignment matrices are saved. Default
  `alignment_prefix_fraction=1` evaluates the fitted samples. A fraction below
  1 fits `floor(fraction * matched_poses)` from the chronological prefix and
  evaluates only the disjoint suffix; both subsets need at least three poses,
  and each estimate/reference prefix must constrain a rigid fit. Fitted and
  evaluated frame IDs/counts are saved separately. This avoids fitting the
  evaluated suffix, but does not establish unseen-session generalization.
  RPE translation uses consecutive matched poses with variable time intervals,
  not a fixed one-second interval. `change.ate_rmse_m_corrected_minus_original`
  is positive for a regression. Read coverage alongside that number.
- Recorded job/source/map/graph/motion hashes are verified before reading and
  again before publishing the report. The original job and map files are left
  unchanged. This detects changes during evaluation, not future edits; compare
  report hashes to current artifacts before reusing the result.

`used_for_generation=false` records a **caller declaration**, not verified
reference independence. A byte-identical reference matching a recorded input
is marked `not_independent` even if the declaration is false. A differently
formatted copy can still reuse generation data: review the source and method.
References supplied as odometry/priors, or derived from the generated map, cannot
establish independent accuracy. Survey uncertainty and sensor correlations also
remain outside this diagnostic. Trajectory error does not certify point-map
surface accuracy, HD boundaries, traffic semantics or georeferencing.

## Locate regions for review

Use the exact `report` artifact returned by evaluation, including its expected
`path`, `sha256` and `bytes`, with MCP `inspect_mapping_trajectory_comparison`:

```json
{
  "report_file": {"path": "/data/comparison.json", "sha256": "<evaluation digest>", "bytes": 244297},
  "window_poses": 12,
  "ranking": "regression",
  "offset": 0
}
```

The equivalent CLI requires the expected digest and size rather than accepting
whatever file currently occupies that path:

```bash
ca mapping-trajectory-inspect /data/comparison.json \
  --sha256 <evaluation-digest> --bytes <evaluation-byte-count> \
  --window-poses 12 --ranking regression
```

The tool verifies the report and **all** recorded input artifacts before and
after reading. Original jobs, large maps and reference files must remain
accessible and unchanged. It reads frozen pose files without loading clouds,
calling the native core, rerunning alignment or spending attempts. Changed
job metadata also invalidates the comparison, even if the point map is unchanged.

Only evaluated retained poses are divided into chronological, nonoverlapping
windows of `window_poses` (integer 2..64, default 12). The last window may be
shorter; a single-pose tail has no RPE. `ranking=regression` orders by local
corrected-minus-original ATE RMSE, descending; `corrected_ate` orders by corrected
ATE RMSE. Ties use chronological window IDs. Rankings include improvements too;
they are not calibrated failure labels. Read up to eight windows at once and
follow `next_offset` for all results. Window IDs remain stable across rankings
for the same report and window size. Windows of unequal duration/length are
not weighted equally in the full-report RMSE.

Each window contains exact original frame IDs, the time range, both ATE/RPE
translation values, and its **unaligned corrected point-map** sensor-origin XY
bounds. The bounds do not use reference-frame coordinates and do not include
scan returns. `corrected_graph_distance_range_m` follows the complete frozen
graph's 3D path from its first pose. Unsupported retained poses inside a frame
span are counted, not assigned errors. Excluded raw frames and alignment-prefix
poses are not silently evaluated.

Use the returned point-map artifact and exact frames to inspect the observed
region and its source evidence. The bounds are a viewing envelope, not an
authorized repair box or an HD gap ID. A large trajectory error does not establish
its cause. Existing density/HD-only repairs freeze motion and cannot fix that
error. No repair, threshold, quality status or adoption is inferred. The current
reference uncertainty remains uncalibrated.

The [NCLT localization evidence](../../benchmarks/vector-map/nclt-trajectory-error-regions/README.md)
contains both complete ranked views, portable recomputation and unchanged-job
receipts. Maturity remains unchanged: localization improves diagnosis, without
demonstrating a corrected map's accuracy.

## Validation and NCLT external-reference results

Synthetic contract tests cover known improvement/regression, nonconsecutive frame
IDs, partial coverage, clock mismatch, provenance declarations, known input reuse,
changed inputs, interleaved changes, report collisions and the CLI/MCP interfaces.
The real NCLT graph/motion wiring was also exercised through MCP using the **same
saved odometry as a deliberately non-independent control**. It is not an accuracy
benchmark and provides no maturity increase.

The bundled recordings contain Velodyne scans and MS25 IMU observations, not the
separately distributed NCLT ground-truth CSV. The authoritative reference source
used by `scripts/prepare_nclt.py` is:

```
https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2012-04-29.csv
https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2012-06-15.csv
```

After the user published the additive network setting, both source CSVs were
downloaded over verified TLS, length-checked and hashed. The bounded reference
preparation script transforms the dataset body poses to the synchronized scan
sensor origin and retains only original scan-time interpolation brackets:

```bash
python scripts/prepare_nclt_reference.py \
  --job /data/finished-point-map-owner \
  --ground-truth /data/groundtruth_2012-04-29.csv \
  --source-url https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2012-04-29.csv \
  --out /data/new-reference
```

Both estimates were compared over live MCP calls on two finished NCLT jobs.
All retained poses matched (199 April / 195 June). The first-30%-fit suffix ATE
was 0.289283 → 0.279199 m for April and 0.246734 → 0.252191 m for June; full-data
fits slightly regressed for both. No reference-uncertainty calibration, pass/fail
gate or map adoption was inferred from these small differences. All existing
job files stayed unchanged. The portable
[external-reference packet](../../benchmarks/vector-map/nclt-reference-accuracy/README.md)
retains raw reference brackets, source hashes/calibration, original motion,
corrected graph/trajectory, full reports and a standard-library verifier.
This is external-reference trajectory shape evidence, not independent HD-map
survey accuracy. Follow NCLT's ODbL/DBCL attribution when sharing the derived data.
