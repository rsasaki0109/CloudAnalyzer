# Interfaces

Stable interfaces keep only the request/result shapes that production callers already need.

## web_point_cloud_reduction

### Current Minimal Interface

The stable interface is intentionally small. It keeps only the contract required to compare and adopt reducers.

```python
class WebSamplingStrategy(Protocol):
    name: str
    design: str
    def reduce(self, request: WebSampleRequest) -> WebSampleResult: ...

@dataclass(slots=True)
class WebSampleRequest:
    point_cloud: o3d.geometry.PointCloud
    max_points: int
    label: str = "point cloud"

@dataclass(slots=True)
class WebSampleResult:
    point_cloud: o3d.geometry.PointCloud
    strategy: str
    design: str
    original_points: int
    reduced_points: int
    metadata: dict[str, Any]
```

### Stable Boundary

- Stable core: `cloudanalyzer/ca/core/web_sampling.py`
- Experimental space: `cloudanalyzer/ca/experiments/web_sampling/`
- Current stabilized lineage: `random_budget` adopted directly in core


## web_trajectory_sampling

### Current Minimal Interface

The stable interface keeps only the contract needed to compare browser trajectory reducers.

```python
class WebTrajectorySamplingStrategy(Protocol):
    name: str
    design: str
    def reduce(self, request: WebTrajectorySamplingRequest) -> WebTrajectorySamplingResult: ...

@dataclass(slots=True)
class WebTrajectorySamplingRequest:
    positions: np.ndarray
    max_points: int
    timestamps: np.ndarray | None = None
    label: str = "trajectory"
    preserve_indices: tuple[int, ...] = ()

@dataclass(slots=True)
class WebTrajectorySamplingResult:
    positions: np.ndarray
    kept_indices: np.ndarray
    strategy: str
    design: str
    original_points: int
    reduced_points: int
    timestamps: np.ndarray | None = None
    metadata: dict[str, Any]
```

### Stable Boundary

- Stable core: `cloudanalyzer/ca/core/web_trajectory_sampling.py`
- Experimental space: `cloudanalyzer/ca/experiments/web_trajectory_sampling/`
- Current stabilized lineage: `turn_aware` adopted directly in core


## web_progressive_loading

### Current Minimal Interface

The stable interface keeps only the data needed to serve an initial point payload plus deferred chunks.

```python
class WebProgressiveLoadingStrategy(Protocol):
    name: str
    design: str
    def plan(self, request: WebProgressiveLoadingRequest) -> WebProgressiveLoadingResult: ...

@dataclass(slots=True)
class WebProgressiveLoadingRequest:
    positions: np.ndarray
    initial_points: int
    chunk_points: int
    distances: np.ndarray | None = None
    label: str = "point cloud"

@dataclass(slots=True)
class WebProgressiveLoadingResult:
    initial_positions: np.ndarray
    initial_distances: np.ndarray | None
    chunks: tuple[WebProgressiveLoadingChunk, ...]
    strategy: str
    design: str
    original_points: int
    initial_points: int
    chunk_points: int
    metadata: dict[str, Any]
```

### Stable Boundary

- Stable core: `cloudanalyzer/ca/core/web_progressive_loading.py`
- Experimental space: `cloudanalyzer/ca/experiments/web_progressive_loading/`
- Current stabilized lineage: `distance_shells` adopted directly in core

