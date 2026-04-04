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


## check_scaffolding

### Current Minimal Interface

The stable interface keeps only the contract needed by `ca init-check`: profile in, YAML text out.

```python
class CheckScaffoldingStrategy(Protocol):
    name: str
    design: str
    def render(self, request: CheckScaffoldRequest) -> CheckScaffoldResult: ...

@dataclass(slots=True)
class CheckScaffoldRequest:
    profile: str = "integrated"

@dataclass(slots=True)
class CheckScaffoldResult:
    profile: str
    yaml_text: str
    strategy: str
    design: str
    metadata: dict[str, Any]
```

### Stable Boundary

- Stable core: `cloudanalyzer/ca/core/check_scaffolding.py`
- Experimental space: `cloudanalyzer/ca/experiments/check_scaffolding/`
- Current stabilized lineage: `static_profiles` adopted directly in core


## check_regression_triage

### Current Minimal Interface

The stable interface keeps only the failed-check contract needed to rank regressions for `ca check`.

```python
class CheckTriageStrategy(Protocol):
    name: str
    design: str
    def rank(self, request: CheckTriageRequest) -> CheckTriageResult: ...

@dataclass(slots=True)
class CheckTriageRequest:
    failed_items: tuple[CheckTriageItem, ...]
    project: str | None = None

@dataclass(slots=True)
class CheckTriageItem:
    check_id: str
    kind: str
    metrics: dict[str, float]
    gate: dict[str, float]
    reasons: tuple[str, ...] = ()
```

### Stable Boundary

- Stable core: `cloudanalyzer/ca/core/check_triage.py`
- Experimental space: `cloudanalyzer/ca/experiments/check_triage/`
- Current stabilized lineage: `severity_weighted` adopted directly in core


## check_baseline_evolution

### Current Minimal Interface

The stable interface keeps only the candidate/history contract needed to decide baseline promotion.

```python
class BaselineEvolutionStrategy(Protocol):
    name: str
    design: str
    def decide(self, request: BaselineEvolutionRequest) -> BaselineEvolutionResult: ...

@dataclass(slots=True)
class BaselineEvolutionRequest:
    candidate: BaselineEvolutionSnapshot
    history: tuple[BaselineEvolutionSnapshot, ...] = ()

@dataclass(slots=True)
class BaselineEvolutionSnapshot:
    label: str
    checks: tuple[BaselineCheckSnapshot, ...]
    passed: bool
```

### Stable Boundary

- Stable core: `cloudanalyzer/ca/core/check_baseline_evolution.py`
- Experimental space: `cloudanalyzer/ca/experiments/check_baseline_evolution/`
- Current stabilized lineage: `stability_window` adopted directly in core


## ground_segmentation_evaluate

### Current Minimal Interface

The stable interface keeps only the ground/non-ground contract needed for `ca ground-evaluate`.

```python
class GroundEvaluateStrategy(Protocol):
    name: str
    design: str
    def evaluate(self, request: GroundEvaluateRequest) -> GroundEvaluateResult: ...

@dataclass(slots=True)
class GroundEvaluateRequest:
    estimated_ground: np.ndarray
    estimated_nonground: np.ndarray
    reference_ground: np.ndarray
    reference_nonground: np.ndarray
    voxel_size: float = 0.2
```

### Stable Boundary

- Stable core: `cloudanalyzer/ca/core/ground_evaluate.py`
- Experimental space: `cloudanalyzer/ca/experiments/ground_evaluate/`
- Current stabilized lineage: `nearest_neighbor` -> `voxel_confusion`

