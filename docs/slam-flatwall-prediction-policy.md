# Flatwall prediction-policy trial

The last-pose warm start lowers the conditional rigid position RMSE of the raw
and gyro legacy controls from 3.206/2.289 m to 0.784/1.076 m. Fine raw sampling
changes little in this metric, while fine gyro sampling increases from 1.484 m
to 1.778 m even though adjacent rotation error falls from 23.657 to 2.690 degrees.
All eight cases, alignment diagnostics, relative errors and paths are in the
[evidence packet](../benchmarks/slam/flatwall-prediction-policy/README.md).

The change is one existing API argument. The first scan uses ordinary
`LidarOdometry.register(points)`. Every later candidate scan passes
`relative_prediction=np.eye(4)`, which starts ICP at the last returned pose.
The original policy repeats the preceding displacement on the right. No
external prediction is sent to the first scan, and no new native implementation
or estimator default is introduced.

All four compensation/sampling controls retain the original scans 1–185,
maximum integer point timestamps, range/map/source settings and stopping rules.
The first two pose bits/reports, all selected source point bits/counts and the
entire gyro-compensated point digest match the frozen originals. All candidate
trajectories and 740 raw diagnostics are closed before scoring.

This is a closed-loop trial: prediction changes later maps, correspondence
searches, adaptive thresholds and rotation centers. Candidate fixed-match
equations are independently reconstructed, but their map matching is not
independently replayed. The inherited gyro/extrinsic and reference mapping
assumptions remain unresolved. Fewer adjacent rotation spikes or a lower fitted
position error does not establish calibrated physical accuracy or a default
policy suitable for other recordings.

## Reproducing the trial

The [recorded protocol](../benchmarks/slam/flatwall-prediction-policy/protocol.json)
and controller freeze specify exact private inputs/native/dependency identities.
Use a fresh process with `RAYON_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=1` and new
output directories outside all frozen inputs. Preserve failed attempts, raw
snapshots, original estimates and every receipt.

```console
python scripts/audit_flatwall_prediction_policy.py estimate --protocol path/to/protocol.json --controller-freeze path/to/controller-freeze.json --out path/to/new-estimates
python scripts/audit_flatwall_prediction_policy.py score --protocol path/to/protocol.json --controller-freeze path/to/controller-freeze.json --estimates path/to/new-estimates --out path/to/new-results
python benchmarks/slam/flatwall-prediction-policy/verify.py
```

Scoring requires all four complete candidate cohorts and their raw diagnostic
closure. It reuses the unchanged zero-offset/unit-scale/body/layout assumptions,
100 ms bracketing, proper rigid fit and first-pose/adjacent-edge policies. The
original default scores reproduce exactly, as does their paired reference.
Frozen evo arithmetic separately checks all eight cases.

The public packet includes the derived 183-pose paired reference and all new
candidate trajectories, so NumPy can recompute the error sequences without the
raw recording. It depends on 26 hashed files at their existing repository paths;
the manifest lists them explicitly. This arithmetic check does not establish
the unresolved raw-reference provenance or rerun private native map searches.

The result is mixed across settings and metrics. No default prediction,
compensation policy, initialization gate or accepted physical state changes;
this recording supplies no GPU or runtime result.
