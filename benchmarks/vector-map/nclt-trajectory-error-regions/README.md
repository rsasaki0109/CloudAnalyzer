# NCLT trajectory error regions

Live MCP inspection of the two saved first-30%-fit comparisons in the
[external-reference packet](../nclt-reference-accuracy/README.md). This packet
locates trajectory review regions; it does not regenerate maps or establish
physical map accuracy. Reference uncertainty and sensor correlation remain
uncalibrated.

| Session | Largest local ATE regression: original frame span | Original RMSE | Corrected RMSE | Change |
|---|---|---:|---:|---:|
| 2012-04-29 | 108–121, 12 retained poses | 0.144672 m | 0.206962 m | +0.062290 m |
| 2012-06-15 | 181–196, 12 retained poses | 0.263464 m | 0.358336 m | +0.094872 m |

April's full evaluated suffix improved from 0.289283 to 0.279199 m, while the
window above worsened. Its largest corrected ATE window is instead frames
151–163, 0.410837 → 0.427495 m. June's largest corrected ATE and largest
regression occur in the same window. These differences guide investigation,
not adoption: they are small against an uncalibrated reference.

## Protocol and retained evidence

- Use the exact hashed report artifact from the earlier evaluation. Original
  job, source, point map, graph, motion, reference and report hashes are verified
  before and after every live inspection. Report positions, timestamps, alignment
  and errors are checked against frozen motion; no new alignment is fitted.
- Partition the **evaluated suffix only** into nonoverlapping chronological
  windows of 12 retained poses. April has 140 evaluated poses (12 windows, tail
  8); June has 137 (12 windows, tail 5). Both complete rankings (`regression` and
  `corrected_ate`) are saved in two eight-window pages per session. Frame lists
  are exact: the printed span is not a claim that every raw frame was retained.
- ATE is 3D position RMSE in each estimate's saved rigid reference alignment.
  RPE translation uses adjacent evaluated poses **within** the window at variable
  time intervals, excluding cross-window steps. Unequal-length windows are not
  equally weighted when reconstructing the full-report RMSE.
- XY bounds use frozen **unaligned corrected-map sensor origins**, not aligned
  reference positions. They omit scan-return extent and are not repair permissions.
  Path-distance ranges use the complete frozen corrected graph's 3D path.
- Original jobs stayed byte-identical: 240 April / 760 June files. The complete
  pre/post SHA-256 snapshot digest is retained in `receipt.json`. No job, attempt,
  map, native binary, reference report or selection was edited. Existing
  density/HD-only repairs freeze motion and cannot correct a trajectory error.

`receipt.json` links the saved report artifact to page files and top windows.
`SHA256SUMS` covers this entire packet. The original point maps and raw recordings
remain outside both packets; portable checks verify retained evidence rather
than rehashing those unavailable large inputs. Live MCP checks did rehash them.

```bash
python benchmarks/vector-map/nclt-trajectory-error-regions/verify.py
```

The standard-library verifier first checks the earlier external-reference packet,
then independently recomputes window ATE/RPE, IDs, frame/time coverage, corrected
map bounds and graph distance from its saved reports and frozen KITTI poses.
It checks both complete rankings/pages, weighted global RMSE and receipt links.
It requires no native core, original mapping jobs, raw logs or network.

The derived data retains the NCLT ODbL/DBCL attribution and restrictions described
in the earlier packet. Source: University of Michigan North Campus Long-Term
Vision and LiDAR Dataset; cite Carlevaris-Bianco, Ushani and Eustice (2016),
*The University of Michigan North Campus Long-Term Vision and LiDAR Dataset*,
IJRR, DOI: 10.1177/0278364915614631. The diagnostic implementation is MIT-licensed.

CloudCompare maturity stays 75%; Agentic mapping maturity stays 52%. This
localization does not demonstrate repair success, independent HD-map accuracy
or calibrated acceptance thresholds.
