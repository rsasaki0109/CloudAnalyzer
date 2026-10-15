# Streaming moment-merge candidates: not adopted

Three bounded CPU candidates for merging chunk moments into existing voxels
were evaluated after [reusing finite verification](cpu-map-stream-finite.md).
All outputs matched, but every candidate had a negative paired median on at
least one real source. None is adopted; the runtime retains the preceding
implementation.

## Candidates and measured outcome

The candidates reuse the existing moment windows, prepare independent
existing-cell merges together, and commit cells in their original order.
Initialization remains scalar. Each eligible window needs at least 128
existing selected cells and contains at most 1,024 groups. Incoming point
gathering retains its preceding 32,768-point bound.

They preserve the scalar expression's parentheses, separate multiplication
and addition operations, and mean/m2 array identity during commit. Each state
owns its arrays after the window is consumed. Counts, totals and count
products must be exactly representable within `2**53`; unsafe states retain
the scalar path. No worker pool or GPU dispatch is involved.

| Candidate | Preparation |
| --- | --- |
| `checked` | Check state arrays, bounds and moment magnitudes; assemble with `np.stack` |
| `trusted` | Use the builder's private accumulator invariants; check existing mean/m2 magnitudes; assemble with `np.stack` |
| `compact` | Same guards as `trusted`; assemble with `np.asarray` |

The first candidate bounds coordinates/means/bounds by `1e140` and moments
by `1e300`. The latter two rely on the preceding finite incoming-moment guard
and bound existing means by `1e150` and m2 by `1e304`. These are exploratory
parameters, not new runtime defaults.

| Candidate | PandaSet 019 | Autoware planning | Tokyo Takanawadai | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| `checked` | −11.04% (1/3) | +2.90% (2/3) | +7.54% (3/3) | 6/9 |
| `trusted` | +1.83% (2/3) | +11.64% (2/3) | −9.15% (1/3) | 5/9 |
| `compact` | +14.20% (3/3) | −4.22% (0/3) | +2.13% (2/3) | 5/9 |

Percentages are the median of `100 * (1 - candidate_ms / control_ms)` over
the three paired trials; positive means shorter. Parentheses count faster
pairs. All 27 pairs are preserved in the
[raw report](../benchmarks/cpu/map-evaluate/real-source/windows-stream-merge-candidates.json).
The three variants ran sequentially, with separate controls and no randomized
cross-variant comparison; their percentages do not establish a ranking.

Observations vary substantially. For example, the checked Autoware third
pair took 2,312.6832 ms for control and 5,859.3637 ms for the candidate.
The trusted Tokyo third pair took 3,856.4196 ms and 1,706.8804 ms. Both remain
in the report, and their causes were not established. Three pairs are
exploratory evidence, not a broad performance guarantee. The regressions and
inconsistent outcomes do not justify adding this path to production.

## Inputs and timing scope

The control is the preceding finite-verification implementation. Original
coordinates, read-only f64 snapshots, 100,000-point chunk boundaries and
final short chunks remain unchanged. The protocol uses 3 m voxels, minimum
100 points and neighbor radius 5, without alignment or downsampling.

[Real-source preparation](c2c-real-source.md) records original source hashes,
attribution and licenses. Six distinct coordinate arrays are reused across
the three probes. No generated geometry, added noise or additional thinning
is used for timing; these source subsets are not independent surveys or
accuracy ground truth.

Clocks cover complete streaming AWD/SCS computation: chunk iteration,
classification, stable grouping, allocation, finite verification, moment
reduction/merge/finalization, Wasserstein and SCS setup/query/statistics.
Loading, snapshot mapping/hashing, parity tracing and report writes are
excluded. Pages are warmed; cold ingestion and peak RSS are not measured.
This lane does not perform resident point NN or PLY export work.

Each variant starts a fresh process and visits PandaSet → Autoware → Tokyo.
Each source has excluded traced parity, one excluded warm-up per policy,
then three pairs alternating control/candidate order. No edits, builds, tests
or scans run during timing. No speed gain is claimed for GPU computation,
resident map evaluation, ingestion or the entire CLI workflow.

## Verification and reproducibility

Each probe compares all final count/mean/covariance/minimum/maximum bytes,
voxel insertion order, logical group input bytes and complete reports.
The 18 candidate and 18 control map states also match the six preceding
finite-stage state chains. Parity and timed calls check 72 complete reports; warm-up results
were excluded from these checks.

A separate audit rebuilt 24 maps and checked 192,880 complete cell records
and 12 complete reports. Its independent scalar builder uses axis-based
classification and public accumulator updates. All 147,315 batched existing
state updates, covering 1,504,029 point visits in 300 windows, were also
checked against scalar moment/bounds merging. Counts are repeated visits
across candidates, not unique cells or coordinate arrays. The audit checked
mean/m2 identity, absence of retained window row views and 2,370 weak array
lifetimes at chunk boundaries.

The [public reproducer](../cloudanalyzer/scripts/benchmark_map_stream_merge_candidates.py)
generates all three byte-identical timed candidate payloads from the captured
baseline. It was created after the probes; separate execution hashes identify
the scripts that actually produced the published timings. Three successful
CLI smoke runs verify its outputs separately; their diagnostic timings are
excluded from the 27-pair comparison. Nine CLI rejection checks cover invalid
repeat counts/types, an unknown variant, duplicate arguments, existing
destinations and mismatched core/I/O sources, preserving sentinel contents.

The record pins 22 unchanged source fingerprints and the compiled cKDTree
binary. Core SHA-256 is
`6216e8fb8ea8faa441a6a1fd126bff0740e3dbe887c8232d22199cd90ee2f603`;
I/O SHA-256 is
`10c1c542a4ad4a70052d947b48c499f2dbf5b9d6d2a49e900dfe40494f61ebff`.
The runtime source has not changed in this experiment. Existing finite-stage
tests were not rerun; new checks cover the isolated candidates and reproducer.
Windows Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and Open3D 0.19.0 use
`OMP_NUM_THREADS=12` and `OPENBLAS_NUM_THREADS=1` for these measurements.

For a fresh reproduction, supply the captured baseline and a preceding
finite-stage coordinate report whose recorded source/snapshot files are
available. The script verifies hashes before loading the candidates and
refuses existing work/output destinations:

```powershell
python cloudanalyzer/scripts/benchmark_map_stream_merge_candidates.py `
  --source-root . --baseline <captured-core.py> `
  --coordinate-report <preceding-finite-report.json> `
  --variant compact --repeats 3 `
  --work-dir <new-work-directory> --output <new-report.json>
```

Select `checked`, `trusted` or `compact` and a new destination for each run.
The experiment does not change the installed runtime implementation.
