# `ca mapping-motion-trial`

Generate one explicit alternative motion/point-map candidate from an existing
point-map owner's job. Keep the baseline accessible and unchanged, compare the
candidate on the same reference protocol, and inspect local regressions before
deciding what to do next.

```json
{"find_loops": true, "use_gravity": false}
```

```bash
ca mapping-motion-trial /data/point-map-owner \
  --out /data/new-loops-only-trial --policy /data/policy.json \
  --reason "Test the correction policy behind the observed trajectory regression" \
  --max-attempts 4
```

For the calling agent, MCP `trial_mapping_motion(job_dir, out_dir, find_loops,
use_gravity, reason, max_attempts=4)` exposes the same operation. Both booleans
and a nonempty reason are required. One call performs one motion trial; no
automatic policy search, adoption or existing budget transfer occurs. The new
1..8 attempt budget is for subsequent **HD generation**, separate from the
single point-map trial. Child jobs initially have no HD candidates or selection.

## What changes and what stays fixed

The tool verifies the baseline job, raw recording, original motion, map/graph
files and native extension. It decodes the raw recording afresh into a new
directory, verifies its timestamps against original odometry, and keeps the
baseline reference graph's exact original frame IDs. It rebuilds the original
odometry edge chain at those nodes, rather than inheriting old loop edges or
optimizing already-corrected poses a second time. The chosen loop search and
IMU gravity correction use existing native algorithms and defaults. No reference
or ground-truth argument is passed to correction.

Scan/map thinning and the dynamic-removal policy stay fixed. Gravity, if chosen,
is extracted from the recording at original timestamps with the same default
IMU mount and drive calibration policy; every retained node must be supported.
Missing gravity fails the trial rather than silently disabling it. No survey
uncertainty or physical mount calibration is inferred.

Processing is bounded to 4,096 raw frames and five million raw returns.
Trajectory/graph inputs are limited to 16 MiB. Source and generated inputs are
hashed and rechecked after processing; candidate graph and trajectory must agree
and preserve the same node IDs, with a nonempty point map. `job.json`, `trial.json`,
fresh scan/IMU evidence and processing reports remain in the new directory.
Existing, interrupted and failed directories are refused on another generation
call. Inspect them with `inspect_mapping_job`; partial outputs are not resumed,
replaced or implicitly trusted.

**Expanded fusion frames are not inherited.** A baseline may have fused extra
interpolated scans into its point map while retaining a smaller reference graph.
The trial uses only reference-graph nodes, and records the excluded raw frame
IDs. It does not prove equivalent point-map coverage, retain points outside a
local region, or patch motion locally. It is an exploratory whole-motion trial.

## Evaluate and compare

Evaluate the returned child `job_dir` with `evaluate_mapping_trajectory`, using
the same reference, provenance, timestamp tolerance and alignment-prefix fraction
as the baseline. Save the evaluation **outside** all mapping jobs. Then call:

```json
{
  "baseline_report_file": {"path": "/data/baseline-comparison.json", "sha256": "<saved digest>", "bytes": 244297},
  "candidate_report_file": {"path": "/data/candidate-comparison.json", "sha256": "<saved digest>", "bytes": 243523},
  "window_poses": 12,
  "offset": 0
}
```

MCP `compare_mapping_motion_trials` verifies both report artifacts and all their
inputs. It requires identical recording, original motion, reference content and
provenance, retained IDs, timestamp matching, alignment protocol and actual
fit/evaluation partitions. It refuses comparisons where apparent gains could
come from dropping evaluated poses or changing the protocol. Reports may be
stored at different paths if their source/reference content is identical.

The response contains both global ATE/RPE values and up to eight nonoverlapping
pose windows per page, ranked by **candidate-minus-baseline** ATE RMSE. Positive
values indicate numerical regressions. Each map's sensor-origin XY bounds stay
in that map's own unaligned frame. Follow `next_offset` for all windows. A smaller
global average can coexist with locally worse regions; a single-pose tail has
no translation RPE. Rankings are descriptive, not calibrated acceptance gates.

Neither evaluation nor comparison changes maps, attempts or selection. Reusing
one reference to choose correction policies is exploratory: excluding alignment
poses does not make the evaluated suffix independent of policy selection or
establish unseen-data accuracy. Review reference uncertainty and sensor
correlation; no automatic `used_for_generation` declaration is supplied.

Old HD geometry and audits cannot be transferred to moved poses. If explicitly
continuing toward both maps, use the new job's corridor/geometry/lane tools with
fresh source inspection, explicit lane assumptions and all existing audits.
Keep the delivered baseline pair until a separate adoption decision. A lower
trajectory RMSE alone does not justify replacing it.

## NCLT evidence

Two explicit ablations were generated and evaluated over live MCP for each
saved NCLT session. Loops-only suffix ATE decreased from 0.279199 to 0.272974 m
(April) and 0.252191 to 0.231641 m (June), while four of twelve local windows in
each session worsened. Gravity-only increased average error in both sessions.
No candidate was adopted or HD map inherited. The original 240/760 job files
stayed byte-identical. See the [portable trial/comparison evidence](../../benchmarks/vector-map/nclt-motion-trials/README.md).
