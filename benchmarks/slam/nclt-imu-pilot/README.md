# NCLT bounded LiDAR-pose/raw-IMU pilot

This is the first real-data accuracy pilot of the bounded native IMU window.
The same actual scans produce LiDAR-only estimates and pose observations for
fusion with original Microstrain channels. Ground truth is opened by a separate
evaluator after both sessions' estimated outputs are frozen. Settings were
fixed before scoring in [protocol.json](protocol.json).

| NCLT session | Consecutive SE3 translation RPE RMSE, LiDAR → fused | Rotation RPE RMSE, LiDAR → fused | Evaluated windows / edges |
| --- | --- | --- | --- |
| 2012-04-29 | 0.16620 → 0.08242 m | 2.06678 → 0.73268° | 20 / 166 |
| 2012-06-15 | 0.12767 → 0.09207 m | 1.49278 → 0.68150° | 12 / 47 |

Translation RMSE decreased by 50.4% and 27.9% respectively on these selected
valid intervals. All 32 windows converged, and all selected windows remain in
the comparison. Secondary per-window rigid-position-fit ATE was
0.11271 → 0.08391 m and 0.07638 → 0.06022 m; these short independently aligned
windows can hide drift. This is not a full-sequence or GLIM comparison.

## Coverage and limits

All 474 scans / 2,711,872 points contribute to the LiDAR baseline. The fixed
50 ms IMU sample-gap guard excludes 42 of 235 April edges and 135 of 237 June
edges. Disjoint 4–16-state windows retain 186 / 236 April scan nodes and
59 / 238 June scan nodes. They cover 166 / 235 (70.6%) and 47 / 237 (19.8%) of
all consecutive edges, or 133.207 / 188.410 s and 37.599 / 189.596 s. Other
valid edges are lost at chunk boundaries or in short excluded runs/tails.
Excluded frames are not silently counted as fused coverage.

Noise and pose-prior weights are diagnostic, not calibrated. Shared-map LiDAR
estimates and adjoining clipped IMU samples have correlations that this model
omits. Velocity initialization uses neighboring LiDAR-derived IMU-origin
positions, including future estimates and the mounting lever arm. Gravity
comes from one causal raw Euler attitude; bias starts at zero. No reference
poses, velocities or bias values initialize or tune the estimator. Each window
restarts its priors; the outputs are not spliced into continuous LIO.

The largest per-axis optimized bias corrections are recorded in results.json.
A separate [mean linearization audit](bias-linearization-audit.json) reintegrates
the edges with largest gyro and accel bias norms in each session. Across those
four edges, the largest first-order/reintegrated delta-position difference is
0.249 mm, delta-velocity difference 0.000927 m/s and rotation difference
7.24e-7 rad. This checks selected means only; covariance, trajectory refitting
and a general validity bound are outside that audit.

These results motivate further validation. Default odometry remains unchanged;
no GPU benefit, continuous LIO accuracy or comparison with another SLAM system
is established. See the [tool protocol](../../../docs/slam-nclt-imu-pilot.md)
for input preparation, commands and estimator limitations.

## Rechecking the frozen scores

```powershell
python benchmarks/slam/nclt-imu-pilot/verify.py
```

The verifier reads [april-poses.csv](april-poses.csv) and
[june-poses.csv](june-poses.csv), uses the original body CSV rows already retained
in `benchmarks/vector-map/nclt-reference-accuracy`, and reconstructs body-z-up
reference poses with SciPy. It evaluates explicit homogeneous SE3 relative
errors independently of the estimator and production evaluation helper.
It checks every window and pooled metric in [results.json](results.json),
including input/output hashes. It does not rerun estimation.

The frozen hashes record the original bytes. For the named public text sources
and CSV files only, verification also accepts Git's LF/CRLF checkout conversion.
Any other content change fails verification. Original recording and native
binary hashes remain exact byte hashes in the estimation receipts.

CSV columns retain exact integer nanosecond scan times, original scan indices,
window membership and the two body-origin pose estimates. Results include
input recording, source script and native extension hashes, coverage, all
per-window scores and convergence status. Original raw IMU recordings and the
full estimation arrays are not included in this small public packet. To run a
new experiment, use new output directories and the documented original inputs;
do not replace this cohort or historical benchmark packets.

## Data source and attribution

Source: [University of Michigan North Campus Long-Term dataset](https://robots.engin.umich.edu/nclt/),
Carlevaris-Bianco, Ushani and Eustice, *University of Michigan North Campus
Long-Term Vision and LIDAR Dataset*, IJRR 2016. NCLT source data and derived
trajectory contents remain subject to [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
and [DBCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
NCLT reference poses use SLAM/RTK and odometry interpolation; they are not
statistically independent sensor truth. This cohort uses corrected body-frame
references, without the additional mounting transform in historical packets.
