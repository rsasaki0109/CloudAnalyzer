# NCLT: agent-controlled generation of both maps

On 2026-10-08, the calling agent ran the bundled NCLT recording through
`mapping-start`, read each result, chose four explicit road hypotheses through
`mapping-candidate`, and selected a review draft. The tools contain no LLM or
automatic semantic decision maker. This verifies that the agent can generate
both maps without operating the editor; it does not verify survey accuracy or
complete HD-map semantics.

Input: [`nclt-2012-04-29.mcap`](../../../web/public/samples/nclt-2012-04-29.mcap),
a thinned 190-second excerpt (236 LiDAR frames), SHA-256
`c74ed00c774c8c19fcc2a0e71116f3479c6fa48ea7ade8a46045a9e6d4b6f5e6`.
The point-cloud stage retained 199 keyframes at 1 m spacing, added six loop
constraints, applied IMU gravity and removed 36 dynamic points. It generated a
683,690-point PLY, corrected KITTI trajectory and pose graph. Optimizer
convergence is processing evidence, not an accuracy measurement. The corrected
trajectory used for road generation is 248.92 m; the odometry report's 252.2 m
measures the earlier trajectory.

All trials kept one forward lane, one backward lane, right-hand traffic,
3.5 m nominal width and a 40 km/h speed limit as **unverified hypotheses**.
Source support is the aggregate sampled center/boundary support; the lane review
gate checks individual traces. Neither determines actual lane identity or rules.

| Candidate | Changed fitting options | Generated length | Retained extent | Sample support | Lane sections needing review |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Defaults | 243.14 m | 97.68% | 1,728 / 3,147 | 9 / 10 |
| 2 | `fit_source_surface=true` | 20.28 m | 8.15% | 336 / 336 | 0 / 4 |
| 3 | `track_boundaries=false`, `fit_boundaries=false` | 243.14 m | 97.68% | 1,848 / 3,257 | 8 / 10 |
| 4 | `track_boundaries=true`, `fit_boundaries=false` | 243.14 m | 97.68% | 1,748 / 3,169 | 8 / 10 |

Every candidate had complete source audits, zero structural/export errors and
matching summaries for editable IR and saved/reopened Lanelet2 OSM. Candidate 2
deferred 222.86 m of the evaluated road; attempting to select it returned exit 1
because it missed the job's 90% retained-extent goal. Its perfect local support
score cannot satisfy the full-drive request.

Candidate 4 was selected as a review draft: it preserves the extent and traffic
hypotheses, retains tracking and has less angular jitter than candidate 3 in the
visual comparison. This is a recorded tradeoff, not proof that it is more accurate
than candidate 1 or 3. Eight lane sections still need source review.
`source_quality_passed=false` and `deployment_ready=false` remain explicit.
Road equipment, legal semantics and georeferencing remain unresolved. All outputs
share local SLAM metre coordinates; exported OSM reports `lanelet2.no_georeference`.

![Four hypotheses over the same generated point cloud; orange boundaries belong to lanes needing source review](candidate-comparison.png)

[`verification.json`](verification.json) records decisions, effective settings,
processing results, quality summaries and artifact hashes. Paths are relative to
the repository (source), Python package (native extension) or generated job.
Large point-cloud and candidate artifacts are generated outputs, not committed
fixtures. Binary hashes identify this run; rebuilds/platforms may differ.

## Follow-up: diagnose and compare anchor hypotheses

`mapping-diagnose` now reads verified saved IR/OSM evidence without rerunning
generation or spending attempts. The original selected draft has 832 height
mismatches and 589 insufficient-return samples. Counts are per oriented
lane/trace, so shared boundaries may be checked twice. These failure types do not
establish whether the root cause is generated Z, wrong XY, another surface level,
missing returns or the assumed road layout. Endpoint holds remain visible even
when aggregate support reaches the threshold.

On 2026-10-09 JST, a new three-attempt job reproduced the same point-map SHA-256
and baseline audit. The agent inspected diagnoses between trials, keeping the
same traffic priors, tracking and 243.14 m extent:

| Trial in new job | Changed option | Supported / sampled | Height mismatches | Insufficient returns | Lane sections needing review |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Reproduce earlier candidate 4 | 1,748 / 3,169 | 832 | 589 | 8 / 10 |
| 2 | `physical_anchors_only=true` | 1,742 / 3,160 | 825 | 593 | 8 / 10 |
| 3 | `anchor_width_prior=false` | 1,677 / 3,140 | 909 | 554 | 9 / 10 |

Trial 2 excluded ten point-coverage-edge candidates as anchors for inferred
boundaries but resolved no additional lane section. Trial 3 reduced missing-return
samples while increasing height mismatches and reviewed lanes. Neither establishes
an improvement. The agent retained trial 1, explicitly recording the unsuccessful
hypotheses. All three had complete audits, matching editable/reopened diagnoses
and zero structural/export errors. Source-review and deployment holds remain.
[`diagnosis-verification.json`](diagnosis-verification.json) records the baseline
per-trace evidence, trials, hashes and decision.

To inspect a generated candidate:

```sh
ca mapping-diagnose runs/nclt-agent --candidate 4
```

For another comparison, start a new job with its budget declared up front; do not
extend an exhausted job or overwrite its artifacts. Test fitting choices based
on the returned evidence. Further investigation should locate the affected traces
in the point footprint and compare nearby surface levels before changing height
alone or changing lane semantics.

## Reproduce the tool sequence

Build/install the current Rust Python core and install `cloudanalyzer`. From the
repository root, create a new output directory:

```sh
ca mapping-start web/public/samples/nclt-2012-04-29.mcap --out runs/nclt-agent --max-attempts 4 --minimum-retained-fraction 0.9
```

Use the `road_options` objects in `verification.json` as four separate JSON files.
Run each trial, reading its reports before deciding the next action:

```sh
ca mapping-candidate runs/nclt-agent --options roads-01.json --reason "Baseline; inspect source support and retained extent"
ca mapping-status runs/nclt-agent
ca mapping-candidate runs/nclt-agent --options roads-02.json --reason "Test source-surface fitting without changing traffic hypotheses"
ca mapping-status runs/nclt-agent
ca mapping-candidate runs/nclt-agent --options roads-03.json --reason "Compare unfitted boundaries at the original extent"
ca mapping-status runs/nclt-agent
ca mapping-candidate runs/nclt-agent --options roads-04.json --reason "Retain tracking while disabling XY fitting"
ca mapping-status runs/nclt-agent
ca mapping-select runs/nclt-agent --candidate 2 --reason "Verify that a short supported fragment cannot meet the full-drive goal"
# Expected exit 1; the job remains unselected.
ca mapping-select runs/nclt-agent --candidate 4 --reason "Retain extent and tracking; eight lane sections and road semantics remain unresolved"
```

These commands reproduce this experiment's explicit choices. For another input,
the calling agent must read evidence and choose hypotheses appropriate to that
input. See the [mapping-job interface](../../../docs/commands/mapping-job.md).

## Attribution and data license

NCLT, University of Michigan: N. Carlevaris-Bianco, A. K. Ushani and R. M. Eustice,
"University of Michigan North Campus long-term vision and lidar dataset", IJRR
2016. See [sample attribution](../../../web/public/samples/ATTRIBUTION.md) and
[upstream NCLT](http://robots.engin.umich.edu/nclt/).
The NCLT-derived evidence and image in this directory are offered under
[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) with contents under the
[Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
