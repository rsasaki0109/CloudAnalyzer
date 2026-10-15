# NCLT alternative motion trials

Four explicit point-map/motion candidates generated over live MCP, then evaluated
and compared against the same saved NCLT references. This is an exploratory
correction-policy comparison, not independent accuracy validation or adoption.

| Session | Delivered reference-graph motion | Loops only | Gravity only |
|---|---:|---:|---:|
| 2012-04-29 | 0.279199 m | 0.272974 m | 0.324391 m |
| 2012-06-15 | 0.252191 m | 0.231641 m | 0.353994 m |

Values are 3D ATE RMSE over the identical evaluated suffix, with each trajectory's
separate rigid fit on the first 30% of matched reference-graph poses: April
59 fit / 140 evaluate, June 58 / 137. Original odometry, raw recording, reference
content, timestamp matching and node IDs remain the same. Reference uncertainty
and sensor correlation are uncalibrated. No statistical confidence is claimed.

Average improvement hides local regressions. Loops-only worsened four of twelve
matching windows in **each** session. Its worst local candidate-minus-baseline
ATE increase was 0.031505 m at April frames 176–191 and 0.052344 m at June
frames 140–154. Gravity-only's worst local increases were 0.460754 m at April
frames 226–235 and 0.699283 m at June frames 231–237. Printed spans refer to exact
retained-frame lists, not every raw frame between endpoints. Both policy results,
including worsening regions, are retained; no candidate was adopted.

## Generation and scope

`trial_mapping_motion` freshly decoded each hashed original recording, matched
original odometry timestamps, and rebuilt its odometry edge chain on exact
baseline reference-graph IDs (199 April / 195 June). Policy booleans explicitly
enabled loops or IMU gravity. Scan/map voxel sizes and dynamic-removal policy
stayed fixed; no reference or truth file was passed to correction. Generated
scans, IMU directions, initial graph and processing outputs were hashed and
verified during the live call. No original job or binary was edited.

Choosing hypotheses after reading previous reference results and comparing them
on the same suffix is **reference-informed exploration**, not a blind test. The
suffix is excluded from rigid fitting, but is not held out from policy selection.
No unseen drive, sensor, survey accuracy or adoption threshold is validated.

The baseline point map may contain expanded-fusion frames. April's delivered map
fused 212 frames while its reference graph has 199; the new candidates use the
199 reference nodes only. They do not inherit expanded fusion, prove equal point
coverage, preserve outside-region records or patch motion locally. Old HD geometry,
traffic semantics and source audits are not validly transferred to changed poses.
Each child has zero existing HD candidates/selection and a separate four-attempt
budget for explicit future HD generation. No HD trial or baseline replacement was
performed in this evidence run.

Baseline jobs remained byte-identical (240 April / 760 June files), as did the
candidate jobs during subsequent evaluation/comparison. Snapshot SHA-256 digests
are in `receipt.json`. Existing/failed trial directories are refused for generation,
so the tool cannot silently overwrite or rerun a trial. Failed paths, changed inputs,
missing IMU coverage, clock mismatch, output disagreement and other refusal cases
are tested with synthetic evidence.

## Portable evidence and verification

Each session/policy folder retains the MCP trial/evaluation responses, child job,
trial/processing reports, raw-scan hash manifest, initial/final graph, frozen KITTI
poses, optional generated IMU directions, full suffix evaluation, original-relative
local windows and both pages of the **baseline-relative** comparison. `receipt.json`
links artifacts and metrics; `comparison-receipt.json` retains local gain/loss counts
and worst regions. All files are covered by `SHA256SUMS`.

Original recordings, raw scan bytes, native binaries and full point maps remain
external. Portable checks cannot reproduce ICP/IMU processing, rehash those large
files, certify point/HD accuracy or prove that fitted alignment is optimal. They
verify copied artifacts, original IDs/clocks, reference/pose transformations,
ATE/RPE arithmetic, matching partitions, local ordering, separate map bounds,
weighted global RMSE and all receipt links using the earlier reference packet.

```bash
python benchmarks/vector-map/nclt-motion-trials/verify.py
```

The verifier uses only the standard library and needs no native core, raw logs,
original jobs or network. The earlier
[reference](../nclt-reference-accuracy/README.md) and
[localization](../nclt-trajectory-error-regions/README.md) packets are dependencies.

Derived NCLT data retains its ODbL/DBCL attribution: University of Michigan North
Campus Long-Term Vision and LiDAR Dataset. Cite Carlevaris-Bianco, Ushani and
Eustice (2016), *The University of Michigan North Campus Long-Term Vision and
LiDAR Dataset*, IJRR, DOI 10.1177/0278364915614631. See the earlier reference packet
for source URLs and license details. Diagnostic implementation is MIT-licensed.

Maturity stays CloudCompare 75% / Agentic mapping 52%. This demonstrates retained
alternative generation and comparison, not a validated repair or better HD map.
