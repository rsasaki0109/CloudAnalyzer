# Parallel SOR: merge only the sorted candidate prefix

Parallel statistical outlier removal (SOR) now merges the first `k + 1`
distances from already sorted worker lists, retaining only that prefix after
each merge. It avoids concatenating and sorting all candidates at partition
boundaries. The guarded prefix path is enabled by default in `finishSorMeans`.
Existing partitioning, neighbour kernels, pool policy, preparations and
numerical reductions retain their behavior.

The subsequent [distance-only query experiment](cpu-sor-slots.md) skips unused
point-index conversion but remains off after inconsistent whole-call timings.
The prefix join documented here stays enabled.

Across two passes on three measured clouds, whole-SOR paired median reductions
are **3.55–10.37%**, with **48/60 timed pairs** faster. Isolated candidate
joining improves by **55.46–61.00%**, with **60/60 pairs** faster. Whole-RPC
gains are smaller because kernels, transfers and output indexing still run.

## Implementation and exactness

Each part returns `k + 1` ascending nonnegative squared distances, optionally
padded with positive infinity. The analysis worker validates each row, then
performs a two-way merge with the current prefix, writing only `k + 1` values.
Discarded values cannot precede the retained prefix in a subsequent merge.
Equal admitted values have identical f64 bits. The join still skips sorted
index zero and sums the same `Math.sqrt` terms from indices 1 through k in
the same order before dividing by k.

NaNs, negative distances, signed negative zero, unsorted rows and short rows
use the preceding reusable full typed-array sort. A failure after earlier
merges rebuilds the full candidate list from the original unchanged rows.
Invalid or nonpositive k also retains the preceding path. Candidate and answer
arrays remain untouched, including views with nonzero offsets. Transferred
query packets are not read during joining; retained row mappings still identify
the returned answers.

Two `Float64Array(k + 1)` buffers belong to one calculation and are reused
across open points. Their payload totals `16 * (k + 1)` bytes, or 400 bytes
at k=24. Unusual rows still allocate the preceding per-list-count scratch
buffers lazily. These figures exclude inputs, answers, other scratch,
allocator overhead and concurrent work; they are not a process-memory cap.
No source cache or additional GPU work is introduced by this join.

The change applies to existing prepared and fresh parallel SOR jobs. Small
jobs keep their existing serial WASM path. At most eight lanes and at least
100,000 source points per requested part remain the pool policy. The
[preparation cache](cpu-knn-preparation.md), [compact query packing](cpu-knn-join.md)
and [copy insertion](cpu-knn-insert.md) remain enabled. Combined neighbour
bounds and earlier GPU hint defaults are unchanged.

## Measured-source protocol

The full unchanged `compared.ply` clouds from
[measured-source preparation](c2c-real-source.md) are used: PandaSet 019
(531,908 points), Autoware planning (878,920) and Tokyo Takanawadai (941,933).
No generated performance data, displacement, noise or further input thinning
is added. Source attribution, licence, original preparation and hashes are
embedded in each report. These subsets are not independent surveys or
filter-quality ground truth.

Measurements use Windows/i7-9750H, 12 logical CPUs, 32 GiB RAM and Chromium
153.0.8010.12. Two independent production analysis workers compare the
preceding full-sort join with the prefix join. **Lane-worker and WASM hashes
must be identical**; only the main analysis worker changes. Both explicitly
request k=24, SOR ratio=1, `knnCache:true` and `knnBounds:false`. Cache job IDs
stay stable, reuse counters increment exactly and retained capacities/lane
counts match under the existing 128 MiB idle payload budget.

One excluded first-use pair and five excluded alternating warm-up pairs
precede ten alternating timed pairs per source/pass. Whole RPC includes
local/cross-part queries, packing, messages, joining, threshold/filter work
and filtered-cloud indexing. Pool startup, loading, cache diagnostics,
verification, export and result removal are excluded. First use is not a
cold median. Initial source order is PandaSet → Autoware → Tokyo; the second
pass reverses it. No build, test or repository scan runs alongside timing.

After both production workers terminate, independent old/current helper
modules join the same real WASM partition/query outputs. Only `finishSorMeans`
is timed; preparation, sequential kernels and packet packing are excluded.
Every f64 mean bit must match. All actual kernel rows are verified sorted and
nonnegative. These isolated joins omit concurrent pool work and messages and
must not be added to or substituted for whole-RPC timings.

## Results

Times are separate medians in milliseconds. Paired shortening is
`median(100 * (1 - prefix_ms / full_sort_ms))` for corresponding pairs, not
the ratio of marginal medians. Positive means faster; all spikes and losses
are retained.

| Source/pass | Whole medians, sort / prefix | Whole paired shortening; wins | Join medians, sort / prefix | Join paired shortening |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 629.55 / 573.60 | +5.23%; 7/10 | 31.35 / 12.50 | +61.00% |
| PandaSet final | 454.50 / 423.25 | +4.00%; 8/10 | 24.85 / 10.90 | +57.07% |
| Autoware initial | 674.20 / 640.15 | +5.62%; 9/10 | 47.85 / 20.60 | +58.69% |
| Autoware final | 641.35 / 610.00 | +3.55%; 6/10 | 48.90 / 21.80 | +55.90% |
| Tokyo initial | 629.95 / 587.80 | +6.59%; 9/10 | 62.95 / 28.45 | +55.54% |
| Tokyo final | 647.55 / 584.90 | +10.37%; 9/10 | 59.05 / 25.70 | +55.46% |

The six positive whole-call paired medians and consistent isolated join
improvements support the default prefix path. Autoware's second pass wins
only 6/10 pairs. CPU clocks, power, scheduling, GC and other applications are
uncontrolled; different timing ranges and occasional large spikes limit
precision and provide no universal speed prediction or result at other k.

PandaSet uses five parts and 19,275 open points; Autoware eight parts and
30,948; Tokyo eight parts and 43,142. Reference preparation payloads stay
35,138,912 / 58,041,728 / 62,200,604 bytes. Kept counts remain
498,305 / 822,878 / 874,565. Every one of the 96 whole-operation pairs
preserves the count and every PLY byte, including excluded first-use/warm-up
pairs: 192 outputs in total. All 90 isolated mean comparisons match every
f64 bit. Both passes use the same source/script/build fingerprints.

## Validation and reproduction

The TypeScript/Vite production build and 14 related Playwright tests pass.
The new prefix/full-sort regression covers 160 part-count/k/row combinations,
three repetitions each, ties, infinity padding, tiny/large distances, nonzero
view offsets, detached packets and unusual-row fallback after earlier merges.
The preceding allocating-join oracle additionally checks 64 combinations.
Browser regressions compare normal attributes and complete SOR exports with
global WASM controls and cover prepared/fresh jobs, budgets, source edits,
removal, worker loss and preparation recovery. Generated fixtures test
correctness only.

The benchmark checks full input, manifest, original measured source, six
implementation sources, archived helper, glue, script and both production
builds before/after timing. All six reports are independently hash-audited;
the adopted source, tests, script and build are saved for later comparisons.
Script syntax and nine invalid-option/overwrite/manifest controls pass.
No GPU adapter/model metadata is collected.

- PandaSet: [initial](../benchmarks/cpu/sor/real-source/windows-pandaset019-prefix-initial.json), [final](../benchmarks/cpu/sor/real-source/windows-pandaset019-prefix-final.json)
- Autoware: [initial](../benchmarks/cpu/sor/real-source/windows-autoware-planning-prefix-initial.json), [final](../benchmarks/cpu/sor/real-source/windows-autoware-planning-prefix-final.json)
- Tokyo: [initial](../benchmarks/cpu/sor/real-source/windows-tokyo-takanawadai-prefix-initial.json), [final](../benchmarks/cpu/sor/real-source/windows-tokyo-takanawadai-prefix-final.json)
- [Benchmark script](../web/scripts/benchmark-sor-prefix.mjs)
- [Join tests](../web/e2e/knn-join.spec.ts)

Archive the full-sort production build and its `knn-join.ts` first. Build the
matching prefix app with `USE_PREFIX_SOR_MERGE = true`, then run from the
repository root:

```powershell
node web/scripts/benchmark-sor-prefix.mjs --baseline-dist notes/sor-prefix-baseline-20261013/dist --input notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-prefix.json
```

The archived helper defaults to `../web/src/knn-join.ts` relative to the
baseline `dist` directory; use `--baseline-helper FILE` for another location.
The script rejects identical main workers, changed lane/WASM kernels, a
prefix-enabled archived helper and a disabled current source flag. The flag
guard does not identify the binary mode, so the matching build is required.
Full measured input/manifest and a new output in an existing parent are
mandatory. Reports cannot be overwritten. `--repeats` accepts 1..100,
default ten. An equivalent archived build/helper and local measured inputs
are required in another checkout.
