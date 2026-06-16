# SLAM Driver Plugin Contract

Third-party packages can add `ca slam-run --driver <name>` support without
changing CloudAnalyzer. Publish an entry point under
`cloudanalyzer.slam_run_drivers`:

```toml
[project.entry-points."cloudanalyzer.slam_run_drivers"]
my-slam = "my_slam_pkg.driver:MySlamDriver"
```

The driver class must satisfy `ca.core.slam_run.SlamRunDriver`: a non-empty
`name` and `run(request: SlamRunRequest) -> SlamRunResult`.

Add this pytest to the plugin repository:

```python
from ca.testing.conformance import run_slam_driver_conformance
from my_slam_pkg.driver import MySlamDriver


def test_my_driver_contract(tmp_path):
    run_slam_driver_conformance(MySlamDriver, tmp_path=tmp_path)
```

The conformance helper checks protocol compatibility, result shapes, finite
values, valid homogeneous poses, non-empty map output, deterministic output on a
tiny fixture, TUM / PLY writer compatibility, required metadata, and stdout /
stderr discipline.

See [`plugins/cloudanalyzer-driver-example`](../plugins/cloudanalyzer-driver-example/)
for a complete working package.
