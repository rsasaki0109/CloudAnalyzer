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
│   └── main.py             # Typer CLI (22 commands)
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
