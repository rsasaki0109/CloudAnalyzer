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
  positions, with no scale fitting. Alignment matrices are saved. ATE and RPE
  are evaluated on those fitted samples; this is not a held-out alignment test.
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

## Validation and the pending NCLT reference check

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

At implementation time that host was denied by the cloud network policy.
Its domain addition was saved as an environment configuration draft; it has not
been applied or published. Actual reference-based NCLT accuracy remains pending.
Do not substitute the self-comparison control for that measurement. Follow NCLT's
ODbL/DBCL attribution and preserve reference/preparation hashes when it is run.
