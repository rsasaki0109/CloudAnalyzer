# Full-cohort ICP constraints and motion prediction

This audit captures all 185 original scans for each of the four frozen
raw/gyro and legacy/fine-source controls. It checks that diagnostic capture
preserves every prior pose bit, timestamp and registration report before
interpreting any local matrix. The [evidence packet](../benchmarks/slam/flatwall-frontend-constraints/README.md)
contains all diagnostic rows and bindings to the original baselines.

## What is measured

The existing internal constant-motion prediction repeats the last body-frame
displacement on the right:

```text
T_pred = I                                     (first scan)
T_pred = T_last                                (one preceding pose)
T_pred = T_last * inverse(T_previous) * T_last  (two or more preceding poses)
```

The estimator orthonormalizes the returned rotation. With an empty map or fewer
than ten selected source points, it returns the prediction without ICP
alignment. These rows retain their stopping labels and absent support. The audit
reports the correction from prediction to returned pose separately from its
final-pose residual and matrix. Neither a small correction nor a convergence
label proves physical accuracy.

`LidarOdometry(capture_linearization=True)` already provides the owned diagnostic
snapshot. This audit independently replays range filtering, first-point voxel
selection, exhaustive nearest search over 27 map cells, strict distance ties,
per-cell capacity and map pruning for every control. It uses canonical native
insertion bytes, separately checked against the transformed insertion cloud,
to preserve cell-boundary behavior. It also reconstructs fixed-match robust
weights, `J.T W J`, `-J.T W residual`, total support and weighted RMS radius.

The parameters are translation xyz and world rotation xyz about the fixed
prediction center. With radius `r`, the spectrum is computed from
`sym(D H D)/sum_w`, `D=diag(1,1,1,1/r,1/r,1/r)`. Eigenvalues retain their sign;
the existing weak-direction ratio is `1e-3`. Unsupported or zero-radius systems
have no spectrum. No matrix is inverted, regularized or fed back into estimation.

This conditional point-to-point surrogate can be full rank on a plane. Its
conditioning does not certify physical observability, calibrated covariance,
GICP geometry or a correct trajectory. The reference was already inspected
before this experiment was designed; this run parses no reference poses. The
unresolved body/clock/layout and IMU/LiDAR calibration assumptions from the
[prior comparison](slam-flatwall-frontend-controls.md) still apply.

## Reproducing the audit

The [protocol](../benchmarks/slam/flatwall-frontend-constraints/protocol.json) and
controller freeze pin the four original estimates, recording, native package,
archived build inputs, audit sources and selected dependency identities before
reading measurements. A full run requires those exact private artifacts. Use a
fresh process and a new output directory outside every frozen input; failures
retain partial snapshots and completed rows at that new path.

```console
python scripts/audit_flatwall_frontend_constraints.py --protocol path/to/protocol.json --controller-freeze path/to/controller-freeze.json --out path/to/new-audit
python benchmarks/slam/flatwall-frontend-constraints/verify.py
```

Set `RAYON_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=1` before starting Python.
The independent replay is diagnostic work, with no runtime or GPU comparison.
The public NumPy verifier checks file integrity, baseline pose/report bindings,
prediction arithmetic, normalized spectra and summaries. It does not rerun
the private point snapshots or independent map search. Raw snapshots and their
native scalar metadata remain local and are listed by hash in the public catalog.

This adds an audit controller and evidence. Estimator defaults, motion prediction,
compensation policy, initialization gates and accepted states are unchanged.
