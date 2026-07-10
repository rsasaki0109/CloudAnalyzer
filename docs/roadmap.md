# Development Roadmap

Status snapshot: 2026-07-02. Phase numbering continues the sequence used in
[CHANGELOG.md](../CHANGELOG.md) (Phase 30 = `ca image-evaluate`).

## Strategic Direction

CloudAnalyzer's moat is the **QA / CI / regression-gate layer**, not the metrics
themselves. Adjacent tools (nerfstudio `ns-eval` / gsplat for photometric, evo
for trajectory, MapEval for map geometry) are all single-shot evaluators with
no regression gates. The goal is to put photometric **and** geometric quality
on the same config-driven regression-gate footing — see [VISION.md](../VISION.md).

## Current State

Completed and merged to `main`:

| Phase | Deliverable |
| ----- | ----------- |
| 30 | `ca image-evaluate` — PSNR / SSIM scoring of rendered vs. reference image sets |
| 31 | `kind: image` config check — photometric quality rides the CI gate (`min_psnr` / `min_ssim`) |
| 32 | LPIPS via the optional `gs` extra (`torch` + `gsplat` + `lpips`), `max_lpips` gate key |
| 33 | Camera-pose I/O — nerfstudio / Instant-NGP `transforms.json` + COLMAP (`ca/core/cameras.py`) |
| 34 | `ca rendered-evaluate` — 3DGS PLY + camera poses + references → gsplat render → photometric gate (`kind: rendered`) |

Also landed (v0.5.0-alpha scope): benchmark report bundles
(`ca benchmark eval --out`), static leaderboard (`ca leaderboard build`),
unified gate severity policy, SLAM driver conformance helper.

**v0.5.0-alpha.1 is not yet tagged.** The CHANGELOG entry (2026-06-17) and the
[release preflight](release-v0.5.0-alpha.md) exist, and all four must-pass
workflows (Test, Pages, SLAM Benchmark Smoke, Public Benchmark Pack) were green
on the release-candidate commit, but git tags stop at `v0.4.0`.

## Plan

### 1. Fix the scheduled `SLAM Leaderboard` workflow failure — first, small

The 2026-07-01 cron run failed at the "Build public SLAM leaderboard snapshot"
step (run 28502082013; logs no longer retrievable). The 2026-06-17 manual run
succeeded, so an environment drift (e.g. a new release of an unpinned
dependency) is the prime suspect.

- Reproduce the snapshot build locally; identify and fix the cause.
- If dependency-driven, consider upper-bound pins.
- Scheduled workflows should be green before cutting the release tag.

### 2. Tag and publish v0.5.0-alpha.1 — small

Follow the remaining steps in [release-v0.5.0-alpha.md](release-v0.5.0-alpha.md):

- Run the Golden Path smoke (`ca benchmark eval` → `ca leaderboard build`).
- `python -m build` + `twine check dist/*`.
- Cut the `v0.5.0-alpha.1` tag and GitHub Release.

### 3. Phase 35 — MapEval AWD / SCS in `ca map-evaluate` — headline, medium–large

Adopt AWD (Average Wasserstein Distance) and SCS (Spatial Consistency Score) —
voxel-level Wasserstein map metrics — plus the 100–500× evaluation speedup from
[JokerJohn/Cloud_Map_Evaluation](https://github.com/JokerJohn/Cloud_Map_Evaluation)
(RA-L '25) into `ca map-evaluate` (`ca/core/map_evaluate.py`).

Follow the established integration pattern end to end:

1. Core metrics in `ca/core/map_evaluate.py`.
2. Gate keys (`max_awd` / `max_scs`) on the `map` check kind (both metrics are
   lower-is-better in MapEval; Equation 10 defines SCS as a local coefficient
   of variation).
3. Triage dimensions + PR-comment metric surfacing.
4. Docs (`docs/commands/`) and tests.

This completes the strategy: photometric (Phases 30–34) and geometric
(Phase 35) quality both gated by the same config-driven CI layer.

### 4. Follow-ups (in priority order)

- **MS-SSIM** — deferred from Phase 32; pure numpy/scipy, no extra required. Small.
- **v0.5.0 stable** — after alpha feedback.
- **CI actions refresh** — resolve the Node.js 20 deprecation warnings
  (`actions/checkout@v4`, `actions/setup-python@v5`). Mechanical.
