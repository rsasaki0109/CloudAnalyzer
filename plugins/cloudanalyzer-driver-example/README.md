# cloudanalyzer-driver-example

Canonical third-party SLAM driver for [`ca slam-run`][cli].

After `pip install cloudanalyzer-driver-example` (or `pip install -e .` from
this directory), this package registers an additional driver under the
`cloudanalyzer.slam_run_drivers` entry-point group, surfacing as:

```bash
ca slam-run scans/ runs/ --driver example
```

It depends only on what CloudAnalyzer already pulls in (`numpy`, `open3d`) —
no `kiss-icp` / `kiss-slam` / `small_gicp` extras required. The
implementation is a deliberately plain **scan-to-scan point-to-point ICP**
via `open3d.pipelines.registration.registration_icp`, with global pose
accumulated by composing per-frame relative transforms and the world-frame
map built from scan-stitched + voxel-downsampled inputs.

This is *not* tuned to beat the built-in drivers on quality. It exists as a
**worked, pip-installable example** of the entry-point contract — a
template anyone shipping their own SLAM driver can copy.

## Layout

```
plugins/cloudanalyzer-driver-example/
├── pyproject.toml                       # registers the entry-point
├── README.md                            # this file
└── src/
    └── cloudanalyzer_driver_example/
        ├── __init__.py                  # re-exports the driver class
        └── driver.py                    # implements SlamRunDriver
```

## How the registration works

`pyproject.toml` carries:

```toml
[project.entry-points."cloudanalyzer.slam_run_drivers"]
example = "cloudanalyzer_driver_example.driver:Open3DICPSlamDriver"
```

When `ca slam-run --driver example` is invoked, CloudAnalyzer's registry
(`ca.core.slam_run`) sees that `example` isn't a built-in, scans installed
entry-points under `cloudanalyzer.slam_run_drivers`, imports
`Open3DICPSlamDriver`, and instantiates it. The class itself satisfies the
`ca.core.slam_run.SlamRunDriver` Protocol (`name` attribute +
`run(request) -> result`).

See [`docs/commands/slam-run.md`][cli] in the main repo for the full
plugin authoring guide.

## Conformance test

Plugin packages should add a small pytest that uses CloudAnalyzer's
conformance helper:

```python
from ca.testing.conformance import run_slam_driver_conformance
from cloudanalyzer_driver_example.driver import Open3DICPSlamDriver


def test_driver_contract(tmp_path):
    run_slam_driver_conformance(Open3DICPSlamDriver, tmp_path=tmp_path)
```

The helper runs a tiny synthetic scan sequence, validates the
`SlamRunResult` contract, checks TUM / PLY artifact writing, rejects plugin
`print()` output, and runs the driver twice to catch nondeterministic output.

## License

MIT. Same as the rest of the CloudAnalyzer repository.

[cli]: ../../docs/commands/slam-run.md
