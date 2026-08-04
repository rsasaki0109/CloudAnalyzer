# MapEval parity and scale validation

CloudAnalyzer's AWD/SCS lane follows the [MapEval paper](https://arxiv.org/abs/2411.17928)
and records the adopted parameters as `cloudanalyzer.mapeval_awd_scs.v1`. The
external comparison target is the MIT-licensed
[`JokerJohn/Cloud_Map_Evaluation`](https://github.com/JokerJohn/Cloud_Map_Evaluation)
repository at fixed commit
[`5955f495df5fbf39f0a184c5d275823b3b5db31b`](https://github.com/JokerJohn/Cloud_Map_Evaluation/tree/5955f495df5fbf39f0a184c5d275823b3b5db31b).

## CI-safe parity smoke

The normal Python CI does not need the upstream C++/Open3D SDK. It generates a
small, deterministic pair of ASCII PCD files and records the local result:

```bash
python3 scripts/mapeval_parity.py \
  --output qa/mapeval-parity
```

The report is `qa/mapeval-parity/mapeval_parity.json`. A run without an
upstream executable has `status: smoke_pass` and
`comparison.status: not_run`; this is intentional and is not presented as
external parity. The report includes the fixed upstream commit, fixture SHA-256
digests, Python/NumPy/platform versions, elapsed time, and peak RSS.

The fixture has two adjacent 3m voxels with 128 points per voxel. Points occupy
distinct 1cm downsample cells, so the upstream `min_voxel_points: 100` filter
remains active after its default 1cm Open3D downsample.

## External executable comparison

Build the upstream project at the fixed commit on a machine with the dependency
stack named by its CMake project: Open3D, Eigen3, PCL, yaml-cpp, TBB, and
OpenMP. The upstream README uses Ubuntu 20.04 and Open3D 0.15.1 as its baseline.
Then run:

```bash
python3 scripts/mapeval_parity.py \
  --output qa/mapeval-parity \
  --official-executable /absolute/path/to/map_eval \
  --upstream-dir /absolute/path/to/Cloud_Map_Evaluation
```

`--upstream-dir` is optional metadata verification; when supplied, the harness
records the checkout commit and dirty state and checks the fixed commit. The
harness creates the upstream executable's hard-coded `../config/config.yaml`
working layout, enables `evaluate_using_initial: true`, disables visualization,
and parses `map_results/map_results.txt`.

The default acceptance tolerance is:

| Metric | Absolute | Relative |
|---|---:|---:|
| AWD (`awd_m`) | `1e-5` m | `1e-5` |
| SCS (`scs`) | `1e-5` | `1e-5` |

With an executable, a mismatch exits non-zero and is recorded as
`comparison.status: fail`. A missing or non-executable path is recorded as
`unavailable` and exits non-zero when explicitly requested, so a release job
cannot silently skip a requested external comparison.

## Why the report has two metric lanes

The fixed upstream source and the CloudAnalyzer core are not numerically
identical implementations:

- the upstream Open3D voxel path normalizes its covariance accumulator in
  `buildVoxelMap`, `computeVoxelEntropy`, and the Gaussian distance path; and
- the upstream Gaussian distance uses Cholesky factors for the covariance term,
  while CloudAnalyzer core uses the symmetric positive-semidefinite Bures matrix
  square root.

The report therefore exposes `cloudanalyzer_corrected` and
`official_compatibility`. The latter is a source-level Python port and is the
expected reference for the external executable comparison. The public core
lane remains mathematically corrected; it is not weakened to reproduce an
upstream numerical quirk.

## 1M / 10M point streaming benchmark

Run the labeled benchmark, outside the normal PR gate, with the same runner
configuration for every comparison:

```bash
python3 scripts/benchmark_mapeval_scale.py \
  --sizes 1_000_000,10_000_000 \
  --chunk-size 100000 \
  --output qa/mapeval-scale.json
```

The benchmark uses `evaluate_map_streaming`, generates both maps lazily, and
records AWD/SCS, elapsed seconds, and the Python process high-water RSS. It does
not claim that timing or memory from different CPU/NumPy/Open3D environments
are directly comparable.

The CI smoke uses the same path at 100k points:

```bash
python3 scripts/benchmark_mapeval_scale.py \
  --sizes 100000 \
  --chunk-size 100000 \
  --output /tmp/mapeval-scale.json
```

Policy:

- 100k points: every CI run, contract and streaming regression only.
- 1M points: labeled CPU benchmark runner and release notes.
- 10M points: labeled high-memory runner; publish the JSON report before
  changing chunk sizes or memory budgets.
- Do not gate cross-hardware wall-clock values; use a fixed runner and compare
  against its previous baseline.
