# Prospective full-rate NCLT evaluation, protocol v1

Status: **prepared, not executed**. No accuracy or performance result exists for
this protocol yet. The current environment permits neither the NCLT S3 host
`s3.us-east-2.amazonaws.com` nor the university host. Do not substitute the
previous sparse demonstration scans or claim a held-out result from them.

## Fixed selection before estimation

Use sessions **2012-02-04** and **2012-03-17**, in that order. Neither date occurs
in the repository's existing benchmark reports at protocol preparation. This
establishes a new evaluation cohort for this work, not a claim about training
exposure of upstream algorithms. For each, retain every synchronized scan in
`[first timestamp, first timestamp + 600 seconds)`. Require the archive to cover
the entire window; do not choose a later start using reference coverage, motion,
or observed errors. No spatial thinning, keyframe selection, or warm-up removal.
These are the full available rate of NCLT **motion-compensated synchronized**
scans, not raw packets or an evaluation of online deskew/LIO.

Official archive URL pattern (from the repository's NCLT preparation script):
`https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/DATE_vel.tar.gz`.
Reference: same prefix, `ground_truth/groundtruth_DATE.csv`.
Check archive sizes and free space before downloading; preserve source URLs,
archive SHA256, byte sizes and acquisition date. Extract only regular
`velodyne_sync/*.bin` entries into a dedicated directory, reject duplicate
basenames, and verify the archive completes before running. Archive acquisition
and validation are still pending; the runner accepts the resulting directory.

## Fixed methods and execution

Compare native CloudAnalyzer with KISS-ICP **1.2.3**, preserving the operating
points from `../nclt-odometry-baselines`: range 1.5–80 m, map voxel 1 m,
20 points/voxel, deskew off, existing convergence policies, no IMU/loop closure.
Do not tune against this cohort. small_gicp tuning remains development work and
is excluded from this confirmation experiment prospectively.

Run three repetitions in fresh processes per method/session (12 runs total).
Order per session: CA/KISS, KISS/CA, CA/KISS. Set all four thread variables to 1.
Set an external timeout of 3600 seconds per run. Retain failed/timeout attempts
and partial outputs; do not silently replace them. Record Python/package/native
binary versions, CPU/quota, process wall time and return status in the experiment
receipt before scoring. Compare input manifests across all repetitions. Pin the
runner and native binary identities before the first run.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  python scripts/nclt_full_rate.py --scans /data/DATE/velodyne_sync \
  --driver cloudanalyzer --out /data/results/DATE-cloudanalyzer-0
```

The runner holds one decoded scan at a time, retains all poses and per-frame
registration durations, and hashes each input as consumed. It never opens ground
truth. Registration includes estimator preprocessing/map update, but excludes
file reading, shared range cropping and input hashing. Pipeline includes those
operations, indexing and estimator setup, but excludes final artifact writes.
Peak RSS includes imports, maps and retained trajectory/manifest metadata; it is
not incremental estimator memory. Report process wall time separately. A caller
must record timeout/kill status because an uncatchable kill cannot finalize the
runner receipt. A receipt still marked `running` is not a successful run.

## Scoring and reporting gate

Only after all 12 attempts are recorded, load original reference CSV rows and
convert body poses using `scripts/nclt_body_frames.py`, without a LiDAR extrinsic.
Use the existing independent SE3 scoring procedure: rigid position-fit ATE
without scale, consecutive-pose translational/rotational RPE, endpoint error,
and reference path length. Require bracketing within 50 ms, no extrapolation.
If invalid reference rows or missing coverage prevent scoring the fixed window,
report it as unscorable with the reason; do not trim the window to improve it.
Report timestamp intervals, total/covered frames and failures alongside metrics.

Publish every run, medians and observed ranges for registration time, total wall
time, p95 frame latency and RSS. Include all startup frames. Do not compare the
new full-rate times directly to the old sparse packet. On this shared VM these
are descriptive measurements, not confidence intervals or a SOTA ranking.
Scoring/orchestration and independent audit of the new packet remain pending.

Source: University of Michigan NCLT, Carlevaris-Bianco, Ushani and Eustice,
IJRR 2016. Data: ODbL 1.0 / DBCL 1.0. No dataset files are committed here.
