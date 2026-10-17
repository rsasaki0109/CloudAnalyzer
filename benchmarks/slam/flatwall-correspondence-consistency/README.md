# Captured correspondence consistency on two real recordings

Fine source sampling produces many repeated target coordinates: deduplicating reverse queries saves a median 57.9–60.8% of query points per matched frame across the eight fine cases. Strict reciprocal filtering retains only a median 27.6–30.4% of original pairs and 28.2–31.4% of robust weight. This identifies a density/reuse effect to investigate, not certified outliers or an established cause of trajectory drift.

The diagnostic processes all **2,560 preserved frames, 16 cases and 1,062,382 captured forward pairs**. Every reverse index is independently checked by exhaustive search. No pose, map, original correspondence, robust weight or default is changed. The reciprocal subset is evaluated without solving or feeding back a motion update.

## All cases

Fractions are per-frame medians over frames with original matches. Query savings count duplicated target coordinates, not runtime. Weak counts use the supported reciprocal-subset systems and the frozen relative eigenvalue threshold 1e-3. Anchors and empty cases remain explicit.

| Recording | Prediction | Input / sampling | Reverse queries saved | Reciprocal pairs retained | Robust weight retained | Weak / supported subset systems |
|---|---|---|---:|---:|---:|---:|
| 01 | repeat-motion | raw-legacy | 7.7% | 88.5% | 88.4% | 0/148 |
| 01 | repeat-motion | gyro-legacy | 7.7% | 89.3% | 89.6% | 1/147 |
| 01 | repeat-motion | raw-0.1875 | 60.6% | 28.6% | 29.3% | 0/184 |
| 01 | repeat-motion | gyro-0.1875 | 60.8% | 27.6% | 28.2% | 0/184 |
| 01 | last-pose | raw-legacy | 7.5% | 89.7% | 89.7% | 0/148 |
| 01 | last-pose | gyro-legacy | 7.1% | 90.0% | 89.8% | 1/147 |
| 01 | last-pose | raw-0.1875 | 60.5% | 28.1% | 29.1% | 0/184 |
| 01 | last-pose | gyro-0.1875 | 59.7% | 28.9% | 29.9% | 0/184 |
| 02 | repeat-motion | raw-legacy | 9.2% | 84.5% | 84.6% | 0/94 |
| 02 | repeat-motion | gyro-legacy | 9.7% | 85.0% | 85.3% | 0/93 |
| 02 | repeat-motion | raw-0.1875 | 59.4% | 29.8% | 30.7% | 0/134 |
| 02 | repeat-motion | gyro-0.1875 | 58.7% | 29.6% | 30.5% | 0/134 |
| 02 | last-pose | raw-legacy | 8.3% | 87.5% | 87.8% | 0/94 |
| 02 | last-pose | gyro-legacy | 7.7% | 88.0% | 88.2% | 0/93 |
| 02 | last-pose | raw-0.1875 | 58.8% | 29.4% | 30.5% | 0/134 |
| 02 | last-pose | gyro-0.1875 | 57.9% | 30.4% | 31.4% | 0/134 |

The eight fine cases retain full rank under this diagnostic after filtering, but reduced support can still hurt motion estimation. The sparse gyro cases each contain one weak subset system. A lower subset cost follows from dropping rows and does not demonstrate better alignment.

Fine registration points use a 0.1875 m source voxel, while insertion retains its previous 0.5 m thinning and 1 m/20-point map cells. Different sample densities naturally allow multiple source points to select one map coordinate. Reuse or failure of a reciprocal test does not by itself make a pair wrong, nor does a full-rank conditional system establish physical observability. Map consistency and a new closed-loop trajectory must be evaluated before adoption.

## Exact check and candidate

The forward pairs are captured native 27-cell matches at the original final pose. This study does not reconstruct that forward search or alter its domain. For each captured target p, search **all original final-pose source points**, including those without captured forward matches. Choose the nearest Euclidean source, breaking exact squared-distance ties by the lowest original source index. Retain a pair only if that index equals its captured matched-source index. This is a reverse-consistency check on the captured pairs; it does not prove global mutual nearest neighbors for a different forward search.

Group exactly equal target coordinates before querying, treating signed zero coordinates as equal. Four-worker `cKDTree` queries obtain a distance bound, followed by a radius broad phase with 1e-10 m padding. Recompute squared distances and choose by distance/index; padding does not relax the final ordering. Both serial and parallel queries match the exhaustive analytic tie controls. All 1,062,382 real reverse indices are independently checked using blocked exhaustive distances, not a second tree implementation.

On the reciprocal subset, preserve original robust weights and prediction center c. The original point-to-point residual is q-p with centered left-perturbation Jacobian `[I, -[q-c]x]` in translation/rotation order. Independently reconstruct both original and subset H/b, support and weighted radius. Spectra use signed eigenvalues of sym(D H D)/sum(w), with D=diag(1,1,1,1/R,1/R,1/R) and R the subset weighted RMS arm. No solve, inversion, damping, fallback or spectral gate is performed. Missing support has no spectrum.

The reciprocal definition is described in the primary [PCL correspondence-estimation API](https://pointclouds.org/documentation/classpcl_1_1registration_1_1_correspondence_estimation.html). Parallel exact-tree query controls follow the [SciPy API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html). No PCL implementation or performance result is imported here.

## Reproduce and verify

From the repository root:

```
python benchmarks/slam/flatwall-correspondence-consistency/verify.py
python -m pytest cloudanalyzer/tests/test_flatwall_correspondence_consistency.py cloudanalyzer/tests/test_flatwall_correspondence_consistency_packet.py -q
```

The public verifier imports no candidate implementation. It checks all frame counts, source/target snapshot identities, original systems against the three existing packets, subset support/PSD/spectra and all case summaries. It exhaustively recomputes **20,126 reverse queries** and both H/b systems on **48 predetermined source/match samples**. Samples are cohort ordinals 1, floor(N/2), N-1 with no replacement; four have no matches and remain explicit. Every sample includes all original source points so unmatched points are available to the reverse oracle. None happens to win a reverse query to an unmatched source in this corpus; the analytic control exercises that case explicitly.

`snapshot-catalog.json` inventories all 2,560 private derived NPZ files. Only the predetermined 48 are distributed. `receipt.json` is the verbatim full private output closure and consequently names private NPZ files absent from the public packet. All private raw reverse indices and original/subset H/b were checked by the frozen controller. The public verifier recomputes raw arithmetic only on the 48 samples, not every private raw system. Original source extraction and map construction are evidenced by prior receipts and snapshot identities, not rerun from the bags by this public verifier.

Tests cover unmatched-source competition, target reuse, duplicate source coordinates, exact ties, signed zeros, serial/parallel parity, empty support, invalid inputs, centered systems and origin translation. Packet controls resign modified hashes and still reject changed reverse indices, multiplicities, subset matrices, spectra, original identities, summaries, sample selection and unsupported claims.

## Interpretation and next candidate

Both full recordings and prior trajectory scores were already known, so this is exploratory rather than blind evaluation. No reference poses are read in this stage, and no clock, frame, extrinsic, scale, scan window or threshold is fitted. Existing IMU/reference provenance limits remain unchanged. There is no calibrated-outlier, physical-observability, trajectory-improvement, GLIM-equivalence or GPU/runtime speedup claim. Saving repeated reverse query points is a count reduction; end-to-end timing has not been measured.

Do not adopt the strict reciprocal filter from this result alone. A next candidate can retain all pairs and balance each target coordinate's aggregate influence with inverse multiplicity weights. Freeze that distinct rule and analytic duplicate/support controls before running a new closed-loop estimator on both complete recordings and all arms/policies. Preserve this strict-filter diagnostic, the unfiltered baselines and every failure attempt. A new closed-loop trial and the inherited reference-scoring policy are required to establish accuracy effects.

## Attribution

Derived from `flatwall_01.bag` and `flatwall_02.bag`, Kenji Koide (AIST), [Zenodo record 7641866, version 2](https://zenodo.org/records/7641866), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes are derived source/match samples and correspondence-consistency diagnostics from historical CloudAnalyzer estimates. Raw recordings and ground-truth poses are not redistributed here. See `attribution.json`, `protocol.json`, `controller-freeze.json` and original catalog dependencies for exact identities.
