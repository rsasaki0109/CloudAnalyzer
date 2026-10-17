# Fixed-match target-normal objective on real flatwall maps

The first target-normal diagnostic does not explain the remaining fine-sampling translation drift. Across both recordings, none of the fine-sampling systems has a weak direction at the declared relative threshold of 1e-3, under either prediction policy. Sparse legacy cases often have weak plane systems, but some also have weak point-to-point systems on the identical accepted-normal subset. No estimator behavior or default changes here.

The packet evaluates **2,560 original frames across 16 cases**, including anchors and frames without usable normal support. It retains every original pose, match and robust weight. One heuristic normal policy was frozen before this geometry audit, after the previous trajectory results were known. These are exploratory diagnostics, not blind accuracy results.

## Results

Normal coverage is the median accepted fraction among frames with original matches. Weak counts use the supported systems only; denominators differ when normal support is absent. The point-to-point comparator uses exactly the same accepted rows, center and weights as the plane system. It is not the original full-match system.

| Recording | Prediction | Input / sampling | Normal coverage | Plane weak / supported | Same-subset P2P weak / supported |
|---|---|---|---:|---:|---:|
| 01 | repeat-motion | raw-legacy | 21.4% | 103/134 | 43/134 |
| 01 | repeat-motion | gyro-legacy | 28.2% | 106/135 | 24/135 |
| 01 | repeat-motion | raw-0.1875 | 44.6% | 0/184 | 0/184 |
| 01 | repeat-motion | gyro-0.1875 | 46.6% | 0/184 | 0/184 |
| 01 | last-pose | raw-legacy | 33.3% | 102/147 | 27/147 |
| 01 | last-pose | gyro-legacy | 34.4% | 97/147 | 23/147 |
| 01 | last-pose | raw-0.1875 | 50.2% | 0/184 | 0/184 |
| 01 | last-pose | gyro-0.1875 | 53.9% | 0/184 | 0/184 |
| 02 | repeat-motion | raw-legacy | 17.7% | 76/93 | 21/93 |
| 02 | repeat-motion | gyro-legacy | 15.9% | 76/91 | 25/91 |
| 02 | repeat-motion | raw-0.1875 | 34.8% | 0/134 | 0/134 |
| 02 | repeat-motion | gyro-0.1875 | 51.4% | 0/134 | 0/134 |
| 02 | last-pose | raw-legacy | 20.3% | 70/94 | 12/94 |
| 02 | last-pose | gyro-legacy | 20.6% | 66/91 | 11/91 |
| 02 | last-pose | raw-0.1875 | 45.9% | 0/134 | 0/134 |
| 02 | last-pose | gyro-0.1875 | 47.0% | 0/134 | 0/134 |

Fine sampling retains roughly 35–54% of original matches under this normal policy. Rejection changes the geometry, and the local map includes samples from earlier estimates that may already have drifted. A full-rank conditional system therefore does not establish physical observability, reliable covariance or accurate trajectory motion. This diagnostic supplies no evidence that switching objectives alone fixes the previously observed drift.

## Frozen geometry and arithmetic

For each case, replay original `inserted_world` arrays in scan order with 1 m map cells, capacity 20 per cell and the captured 80 m pruning rule. Check byte membership of every captured target in the reconstructed pre-insertion map. **This replays insertion and target membership, not the original nearest-match search.**

Sort map points lexicographically by x/y/z, preserving duplicates and their sampling density. Query a radius of 1 m plus a 1e-10 m broad-phase padding with four CPU workers; recompute exact squared distances, retain distances <= 1 m², and select at most 12 neighbors ordered by distance and canonical map index. Padding only broadens candidates; it never broadens final acceptance. No timing or parallel speedup claim is made.

Compute the population PCA covariance around the neighborhood centroid. Its smallest-eigenvalue eigenvector is the normal; orient its largest absolute component positively. Require at least six neighbors, middle variance > 1e-8 m², middle/largest >= 0.1 and smallest/middle <= 0.1. Preserve signed eigenvalues. Record `insufficient_neighbors`, `line_or_point` and `not_planar` explicitly, with too-few-neighbor rejection taking precedence. This is one experimental heuristic, not a calibrated noise model or a parameter search.

With frozen world source q, target p, normal n, robust weight w and prediction center c:

```
r = n dot (q - p)
J = [n, (q - c) cross n]       # translation, then centered left rotation
H = J^T W J
b = -J^T W r
```

The point-to-point comparator uses `[I, -[q-c]x]` and three residual components on the identical accepted rows. Normalize each system using `D H D / sum(w)`, with `D = diag(1,1,1,1/R,1/R,1/R)` and R the accepted-row weighted RMS source arm. Symmetrize before eigendecomposition and count eigenvalue/max < 1e-3, retaining negative roundoff. Missing support has no spectrum. No solve, inversion, damping, fallback, spectral gate or state feedback is performed.

The analytic plane test recovers three unconstrained directions (two in-plane translations and rotation about the normal) while its same-pair P2P comparator is full rank. Other tests check finite-difference Jacobians, gradient sign, coordinate/normal-sign invariance, deterministic serial/parallel neighborhoods and explicit rejection cases. These fixtures validate arithmetic; they are not performance data.

The formulas follow primary descriptions of [PCA normal estimation (PCL)](https://pointclouds.org/documentation/tutorials/normal_estimation.html) and the [point-to-plane objective (Open3D)](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html). CloudAnalyzer already has a separate native point-to-plane ICP implementation; this experiment does not change that solver or adopt its damping/fallback behavior into odometry.

## Public verification and scope

Run from the repository root:

```
python benchmarks/slam/flatwall-plane-objective/verify.py
python -m pytest cloudanalyzer/tests/test_flatwall_plane_objective.py cloudanalyzer/tests/test_flatwall_plane_objective_packet.py -q
```

The independent verifier imports no candidate implementation. It checks hashes, frozen source/protocol/receipt bindings, all 2,560 frame spectra and case summaries, and original frame metadata/snapshot identities from three existing packets. It recomputes covariance, normal acceptance, robust weights and both objective systems on **48 full-map samples**. Samples were fixed as cohort ordinals 1, floor(N/2), N-1, without replacing empty cases. On match rows 0, floor(M/2), M-1 it independently searches every map point: **132 exhaustive queries**; four sampled frames have no matches and remain explicit. Full private neighborhood search is not independently replayed by this public verifier, nor is all-frame H/b recomputed from public raw points.

The private closed audit reconstructs original weights/H/b and independently accumulates plane H/b on every frame, preserves all 2,560 derived snapshots and records every file hash. Public `snapshot-catalog.json` retains that complete inventory; only the predetermined 48 NPZ files are distributed. `receipt.json` is the verbatim full-output receipt, so it also names private NPZ files absent from this packet. Public tests resign altered hashes and still reject changes to neighborhoods, normal geometry, matrices, spectra, selection, summaries, source identity and unsupported claims.

Both full recordings and their previous scores were already known. No reference poses are read in this diagnostic, and no reference-driven clock, body frame, scale, extrinsic, scan window or normal threshold fitting occurs. Author-configured IMU scale/offset/identity transform and reference axes/clock remain previously documented assumptions, not physical calibration certificates. No trajectory improvement, GPU speedup, runtime speedup or GLIM equivalence is claimed. The existing native binary and its source/build archives remain unchanged.

## Provenance

Derived from `flatwall_01.bag` and `flatwall_02.bag`, Kenji Koide (AIST), [Zenodo record 7641866, version 2](https://zenodo.org/records/7641866), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes are derived local maps, neighborhoods and geometry diagnostics from historical CloudAnalyzer estimates. Raw recordings and ground-truth poses are not redistributed in this packet. See `attribution.json`, the original snapshot catalog dependencies, `protocol.json` and `controller-freeze.json` for hashes and exact conditions.

The next useful experiment is a frozen, explicit correspondence/trajectory update policy evaluated against both complete recordings, with the present objective diagnostic retained as a baseline. A low-dimensional geometry-aware update must first pass analytic motion and failure controls; the full-rank fine cases require investigating correspondence/map consistency as well as local curvature. No universal prediction-policy or objective winner is selected here.
