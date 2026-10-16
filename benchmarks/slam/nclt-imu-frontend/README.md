# NCLT inertial predictions for the LiDAR frontend

The optional IMU prediction now initializes ICP before each supported scan is
registered. Both arms use the same four-state window and previous diagnostic
weights. ICP's pose builds the map and becomes the published trajectory. The
baseline uses constant-motion initialization. The assisted arm derives its
relative body motion from the previous optimized IMU state, world velocity and
bias, including the mounting lever arm.

These are the existing synchronized April and June samples, with 474 poses and
472 consecutive SE3 edges. There are 203 usable prediction edges, 177 gap
fallbacks and 92 bootstrap intervals. The two initial poses are also included.
Every gap clears the window and its marginal prior. The first valid interval
after each reset initializes velocity after LiDAR registration; IMU prediction
starts with the next valid interval. All 590 window updates across both arms
converged. At most four states were active.

| Session | ICP steps, baseline → assisted | Reduction | Translation RPE RMS | Rotation RPE RMS | Global ATE RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2012-04-29 | 12,998 → 10,009 | 23.0% | 16.794 → 16.640 cm | 2.0381 → 2.0346° | 16.545 → 16.361 cm |
| 2012-06-15 | 14,788 → 14,009 | 5.3% | 14.10996 → 14.11043 cm | 1.7309 → 1.7259° | 15.414 → 15.398 cm |

Translation accuracy is nearly unchanged. June's translation RPE increases
slightly; maximum translation RPE increases in both sessions (61.56 → 63.83 cm
and 38.27 → 38.68 cm). Maximum ATE also increases in both sessions. April's
endpoint translation error increases from 34.33 to 34.49 cm. The experiment
remains opt-in. The baseline poses exactly match the previously frozen LiDAR
outputs. These frontend poses are a different output from the optimized poses
scored in the earlier fixed-lag replay.

`factor-performance.json` records a separate warm-filesystem measurement of
IMU factor construction. The original path decodes the topic on each call;
`ImuSeries` loads once per measured block and indexes each interval. With the
original 9,043 and 8,839 samples and all 193 and 102 valid intervals, median
times over three alternating-order blocks were:

| Session | Repeated decoding | Snapshot load + all factors | Ratio |
| --- | ---: | ---: | ---: |
| 2012-04-29 | 1,818.28 ms | 16.33 ms | 111.3× |
| 2012-06-15 | 838.85 ms | 11.28 ms | 74.4× |

Every measurement field, covariance and bias Jacobian was bit-identical in
all 885 compared factor pairs. The whole topic stays in memory. These local
factor timings and ICP step counts do not establish end-to-end SLAM runtime.

Estimation read no reference poses. The controller, settings, binary and all
outputs were frozen before scoring against original body-pose reference rows.
The evaluator scores every frontend pose, including gap and bootstrap
transitions. ATE uses one global rigid position fit per session without scale;
RPE uses full consecutive SE3 transforms. Each arm's map evolves independently,
so a fallback pose can differ after earlier predictions changed registrations.

```bash
python benchmarks/slam/nclt-imu-frontend/verify.py
```

The verifier uses the frozen independent pose math from the earlier public
packet, reconstructs all 474 poses from the CSVs, checks every reported score,
and validates chronological window sizes, marginalizations and coverage.
It does not rerun the estimator. Exact input and artifact hashes, raw timing
blocks, convergence outcomes and full-precision scores accompany the poses.

See [the API and replay instructions](../../../docs/slam-imu-frontend.md).
The input cohort was already examined during development. Fixed gravity,
shared-map pose-prior correlations, marginal linearization and sensor noise
remain diagnostic approximations. Longer unseen sequences and calibration are
needed before default adoption or a comparison with a complete LIO system.

Source: [University of Michigan North Campus Long-Term dataset](https://robots.engin.umich.edu/nclt/),
Carlevaris-Bianco, Ushani and Eustice, *University of Michigan North Campus
Long-Term Vision and LIDAR Dataset*, IJRR 2016. NCLT data and derived trajectory
contents retain [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) /
[DBCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) terms. Software remains
under the repository's software license. Reference poses combine SLAM/RTK and
odometry interpolation and are not statistically independent sensor truth.
The documented body-z-up conversion is applied to original reference rows.
