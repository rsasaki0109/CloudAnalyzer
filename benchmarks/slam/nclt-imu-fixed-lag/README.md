# NCLT chronological inertial-window replay

This new cohort evaluates all scans, including IMU-gap transitions, while
maintaining a bounded window with oldest-state marginalization. The
[previous pilot](../nclt-imu-pilot/README.md) used independent offline windows
and future-dependent velocity initialization; its subset scores are not
directly comparable to these full-session latest-state outputs.

| Session | Full-session translation RPE RMSE, LiDAR → replay | Rotation RPE RMSE, LiDAR → replay | Global ATE RMSE, LiDAR → replay |
| --- | --- | --- | --- |
| 2012-04-29 | 0.16794 → 0.14308 m | 2.03809 → 1.35364° | 0.16545 → 0.16028 m |
| 2012-06-15 | 0.14110 → 0.13842 m | 1.73089 → 1.48639° | 0.15414 → 0.15358 m |

All 474 poses / 472 consecutive edges are scored. Translation RPE RMSE drops
14.8% in April and 1.9% in June. These are aggregate changes, not improvements
on every individual edge. June's largest rotation RPE increases from 3.50° to
3.55°, and largest globally aligned position error from 0.339 m to 0.345 m.
The raw and subgroup metrics are retained in [results.json](results.json).

## Window updates and missing input

The replay uses existing actual LiDAR-only pose estimates, original NCLT ms25
IMU channels and the same fixed diagnostic weights as the first pilot. No
reference trajectory, reference velocity or optimized pilot state initializes
the replay. Each update uses the current and previous LiDAR-derived IMU-origin
positions for velocity initialization; later LiDAR observations do not alter
already emitted outputs. Gravity comes from the same causal initial sensor
roll/pitch. Bias starts at zero at each segment start; a new IMU edge is
integrated at the last estimated bias thereafter.

There are at most **four active states**, about 2.4 seconds at the bundled scan
cadence. Removing the oldest state transfers its incident factors into a dense
15D frozen Gaussian prior. Retained observations and edges remain exactly
once. The experiment performs 112 marginalizations in April and 11 in June;
all 193 + 102 fusion updates converge.

The original sample-gap guard remains 50 ms. Across 42 April and 135 June
edges, the replay clears all inertial-window information and emits the current
LiDAR pose. The next valid edge begins a new segment. No measurement, bias
uncertainty or marginal prior is propagated through a rejected gap. Each
session's first scan also uses LiDAR alone. Thus all scans have outputs, while
IMU fusion itself covers 193/235 April edges and 102/237 June edges. This does
not claim continuous inertial coverage through missing data.

The primary RPE includes those fallback transitions. `valid_imu_edges` and
`gap_edges` in results.json expose their errors separately. ATE uses one rigid
position fit for the whole session, without scale or per-window alignment.
Estimates for both sessions were frozen before a separate evaluator opened
ground truth; [protocol.json](protocol.json) records the preselected model.

## Rechecking and running another experiment

```powershell
python benchmarks/slam/nclt-imu-fixed-lag/verify.py
```

The verifier reads the full [April](april-poses.csv) and [June](june-poses.csv)
pose arrays, status and active-state counts. It reuses frozen independent
homogeneous-SE3/reference math from the first pilot's public verifier, without
calling the estimator or production evaluation helper. It checks every
full-session and subgroup metric, all fallback poses, and the chronological
marginalization events. Named public text hashes accept only Git's LF/CRLF
conversion. Original binary/recording hashes remain exact captured hashes.

See [the API and replay instructions](../../../docs/slam-imu-marginalization.md)
to create a new experiment in fresh output directories. Original raw IMU bags,
full optimizer arrays and native binaries are not included in this compact
packet. Their hashes and source/build provenance are retained. Successful
cohorts, source controllers and historical packets are not overwritten.

The estimator is an experimental pose/IMU replay, with no feedback into LiDAR
registration, loop closure, automatic reintegration or calibrated observation
covariance. Shared-map pose priors and clipped adjoining IMU measurements have
unmodeled correlations; gravity and old marginal linearizations are fixed.
These results do not enable default fusion or establish a GPU speedup,
real-time throughput, long-duration robustness or a GLIM comparison.

## Attribution

Source: [University of Michigan North Campus Long-Term dataset](https://robots.engin.umich.edu/nclt/),
Carlevaris-Bianco, Ushani and Eustice, *University of Michigan North Campus
Long-Term Vision and LIDAR Dataset*, IJRR 2016. NCLT data and derived trajectory
contents retain [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) /
[DBCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) terms.
Reference poses use SLAM/RTK and odometry interpolation, rather than
statistically independent sensor truth. Only the documented body-z-up frame
conversion is applied to original reference rows.
