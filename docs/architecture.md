# Architecture

## Project Structure

```
cloudanalyzer/
├── ca/                     # Core library
│   ├── core/               # Stable minimal interfaces extracted after comparison
│   │   ├── __init__.py
│   │   ├── web_progressive_loading.py # Stable contract for browser progressive loading
│   │   ├── web_sampling.py # Stable contract for browser point-cloud reduction
│   │   └── web_trajectory_sampling.py # Stable contract for browser trajectory reduction
│   ├── experiments/        # Discardable concrete implementations
│   │   ├── process_docs.py # Consolidated docs writer for active experiment slices
│   │   ├── web_progressive_loading/ # Alternative progressive-loading planners + evaluator
│   │   ├── web_sampling/   # Alternative point-cloud reducers + evaluator
│   │   └── web_trajectory_sampling/ # Alternative trajectory reducers + evaluator
│   ├── __init__.py         # __version__
│   ├── io.py               # Point cloud I/O (pcd/ply/las)
│   ├── registration.py     # ICP / GICP registration
│   ├── scan_match_debug.py # Scan-to-map matching diagnostics
│   ├── metrics.py          # NN distance, summarize, threshold
│   ├── evaluate.py         # F1, Chamfer, Hausdorff, AUC, plot
│   ├── visualization.py    # Colorize, snapshot
│   ├── report.py           # JSON / Markdown report generation
│   ├── compare.py          # Full compare pipeline
│   ├── diff.py             # Quick diff (no registration)
│   ├── info.py             # Point cloud metadata
│   ├── stats.py            # Density, spacing statistics
│   ├── batch.py            # Batch processing
│   ├── downsample.py       # Voxel downsampling
│   ├── sample.py           # Random sampling
│   ├── filter.py           # Statistical outlier removal
│   ├── merge.py            # Merge point clouds
│   ├── align.py            # Sequential registration + merge
│   ├── split.py            # Grid tile splitting
│   ├── convert.py          # Format conversion
│   ├── normals.py          # Normal estimation
│   ├── crop.py             # Bounding box crop
│   ├── density_map.py      # 2D density heatmap
│   ├── pipeline.py         # filter → downsample → evaluate
│   ├── plot.py             # Multi-F1 comparison, 3D heatmap
│   └── log.py              # Logging configuration
├── cli/
│   ├── __init__.py
│   └── main.py             # Typer CLI
├── tests/                  # Core, CLI, process, and integration tests
├── pyproject.toml          # Package config, mypy, pytest
└── setup.py                # Editable install shim
```

## Design Principles

1. **Compare concrete implementations first** — New problem areas start in `ca.experiments.*` with multiple interchangeable implementations
2. **Stabilize only the minimum** — `ca.core.*` keeps the shared request/result contract and the currently adopted implementation
3. **CLI is thin** — CLI commands only parse args, call the core function, and format output
4. **Logging on stderr** — Progress info goes to stderr via `logging`, keeping stdout clean for `--format-json` piping
5. **No global state** — Each function is stateless and takes explicit arguments unless an experiment explicitly tests a stateful design

## Core Promotion Checklist

An experiment slice graduates from `ca.experiments.<slice>/` to `ca.core.<slice>.py` only when **all** of the following are true. The checklist is enforced socially (PR review), not yet mechanically.

- [ ] Stable request / result contract is documented in `docs/interfaces.md` (regenerated via `ca.experiments.process_docs`).
- [ ] At least **three fixtures**: one synthetic, one small public sample, one regression fixture covering a past failure mode.
- [ ] JSON output is deterministic across re-runs on the same input.
- [ ] At least one path through `ca check` can consume the result (or the slice declares it is diagnostic-only).
- [ ] Failure cases produce triage-able output (named error / metric, not just an exception).
- [ ] A documented performance budget (target runtime + peak memory on the largest fixture) is recorded in `docs/decisions.md`.
- [ ] `docs/decisions.md` records why this implementation was adopted and what the others traded off.
- [ ] Losing implementations are either archived under `experiments/<slice>/archive/` or deleted within 1–2 releases of promotion.
- [ ] Evaluation harness and fixtures stay in `experiments/<slice>/` even after promotion — they double as regression tests.

A slice that fails any item stays in `ca.experiments` and keeps being re-benchmarked by `process_docs`.

## Evaluation Command Roles

CloudAnalyzer exposes several evaluation entry points that answer different questions. They are intentionally separate so that downstream consumers (CI gates, batch reports, library callers) can pick the right one without overloading a single command.

| Command | Question | Input shape | Notes |
|---|---|---|---|
| `ca evaluate` | Preservation QA — did processing degrade the artifact relative to its source? | Two artifacts of the same kind | Used by `--evaluate` on processing commands; uses F1 / Chamfer / AUC curves |
| `ca map-evaluate` | Map-quality QA — how close is a reconstructed map to a reference map? | Estimated map + reference map | MapEval-inspired accuracy/completeness@t; experimental, not yet in `ca.core` |
| `ca run-evaluate` | SLAM-run QA — is one run acceptable end-to-end (map + trajectory)? | Map pair + trajectory pair | Combines map evaluation and trajectory evaluation; emits a combined HTML report |
| `ca check` | Gate orchestration — run all configured gates and report pass/fail with triage | `cloudanalyzer.yaml` | Chains `evaluate`, `map-evaluate`, `traj-evaluate`, `loop-closure-report`, `ground-evaluate`, perception evals; produces config-driven exit codes |
| `ca benchmark eval` | Frozen-suite SLAM QA — does this run pass a published reference + gate? | Benchmark suite YAML + user map + user trajectory | Wraps `run-evaluate` against a suite's fixed reference + gate so swapping SLAM pipelines is one command |
| `ca loop-closure-report` | Manual loop-closure QA — did closing the loop actually improve the map / trajectory? | Before / after / reference artifacts (+ optional posegraph session) | Wires `ca.evaluate` + trajectory evaluation + `posegraph-validate` |

Naming heuristic: `evaluate` ≈ preservation, `map-evaluate` ≈ map quality, `run-evaluate` ≈ run quality, `check` ≈ gate. When unsure, start from `ca check` with a config snippet — it picks the right primitive per check.

## Dependencies

| Package | Purpose |
|---|---|
| `open3d` | Point cloud I/O, KDTree, registration, visualization |
| `numpy` | Array operations |
| `typer` | CLI framework |
| `matplotlib` | Density maps, F1 curve plots |

## Data Flow

```
Input PCD/PLY/LAS
    │
    ▼
  ca.io.load_point_cloud()
    │
    ├─── ca.info / ca.stats        → metadata dict
    ├─── ca.filter                 → cleaned PCD
    ├─── ca.downsample / ca.sample → reduced PCD
    ├─── ca.registration.register  → aligned PCD
    ├─── ca.metrics                → distance array
    │       ├─── ca.evaluate       → F1/Chamfer/AUC dict
    │       └─── ca.visualization  → colorized PCD → snapshot
    ├─── ca.core.web_progressive_loading → stable browser progressive-loading interface
    ├─── ca.core.web_sampling      → stable browser reduction interface
    ├─── ca.core.web_trajectory_sampling → stable browser trajectory reduction interface
    ├─── ca.experiments.web_progressive_loading → alternative progressive planners + evaluator
    ├─── ca.experiments.web_sampling → alternative point-cloud reducers + evaluator
    ├─── ca.experiments.web_trajectory_sampling → alternative trajectory reducers + evaluator
    ├─── ca.experiments.process_docs → consolidated experiment docs
    ├─── ca.split                  → tile PCDs
    └─── ca.pipeline              → filter → downsample → evaluate
```
