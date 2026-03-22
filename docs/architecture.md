# Architecture

## Project Structure

```
cloudanalyzer/
├── ca/                     # Core library
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
├── tests/                  # 153 tests
├── pyproject.toml          # Package config, mypy, pytest
└── setup.py                # Editable install shim
```

## Design Principles

1. **Each module returns a dict** — Every function returns a JSON-serializable dict, making it easy to compose, test, and output
2. **CLI is thin** — CLI commands only parse args, call the core function, and format output
3. **Logging on stderr** — Progress info goes to stderr via `logging`, keeping stdout clean for `--format-json` piping
4. **No global state** — Each function is stateless and takes explicit arguments

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
    ├─── ca.split                  → tile PCDs
    └─── ca.pipeline              → filter → downsample → evaluate
```
