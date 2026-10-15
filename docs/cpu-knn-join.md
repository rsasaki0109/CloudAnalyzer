# Parallel normals and SOR: compact result joins

Parallel normal estimation now scatters raw f32 lanes into cloud order without a
typed-array view per point. Parallel statistical outlier removal (SOR) packs
boundary queries directly and reuses its candidate merge buffers. Both changes
are enabled for existing pool-eligible jobs; neighbour selection, partitioning,
orientation, sorting and numerical reduction order stay the same.

On three measured point clouds, two measurement passes found normal-estimation
whole-RPC paired median reductions of **3.86–11.65%**, with **56/60 timed pairs**
faster. SOR's packing and merge phases improved, but its whole-RPC paired medians
ranged from **−1.44% to +9.88%**, with **46/60 pairs** faster. These measurements
support the normal scatter improvement; they do not establish a consistent
whole-SOR speedup for every source.

The subsequent [sorted-prefix SOR join](cpu-sor-prefix.md) merges only the
required k+1 distances from the already sorted kernel lists. It is now enabled
after two measured passes with positive whole-call paired medians and exact
outputs. The measurements below describe the preceding reusable full-sort join.

## Implementation and precision

- `web/src/knn-join.ts::scatterNormalParts` creates one output and one raw-lane
  view per part. Three integer assignments per point preserve all f32 bits,
  including signed zero and NaN payloads, while avoiding per-point `subarray`.
- `makeSorQueries` counts each lane's boundary queries, allocates exact-size
  f64 packets, and fills them from the existing open-point coordinates. Typed
  row maps replace per-query coordinate objects, `flatMap`, and candidate views.
  Query order, lane assignment and reach masks match the prior path.
- `finishSorMeans` keeps one f64 scratch array for each candidate-list count
  encountered during the calculation. It fills the entire selected buffer,
  applies the same `Float64Array.sort`, skips index zero, and sums the same
  square roots in the same order. Different list counts use different buffer
  lengths: additional padding could change infinity/NaN behavior.
- Row maps remain on the analysis worker when query packets are transferred.
  Finishing a calculation does not read the detached packets. The scratch and
  row maps belong to one calculation and do not retain a source-cloud cache.
- The row maps cost four bytes per open query per lane that receives queries,
  plus four bytes per open query for its original index. Query packets and
  existing kernel candidate/output arrays also consume memory. This is a
  reduction in JavaScript objects and short-lived allocations, not a measured
  peak-memory bound.

Pool policy is unchanged: at most eight lanes, at least 100,000 source points
per requested part, and at least two parts. Small jobs use the existing serial
WASM path. Parallel normals still use each part's own neighbours near partition
borders, as before; their comparison is against that same partitioned operation.
SOR still asks all reached parts and matches the global filter.

## Measured-source protocol

[`benchmark-knn-join.mjs`](../web/scripts/benchmark-knn-join.mjs) compares an
archived pre-change production worker against the current production worker.
It requires identical **lane-worker and WASM hashes** so that the comparison
isolates the analysis-worker joins. Each method gets two fresh analysis workers
with independent pools and full, unchanged input files. Pool startup and loading
are excluded. Settings are k=24, upward normals and SOR ratio=1.

One excluded first-use pair precedes five excluded alternating warm-up pairs
and ten alternating timed pairs. Normal trials check every f32 bit; SOR trials
check the kept count and byte-identical exported PLY, including coordinates and
attributes. Timing covers the complete RPC, including kernels, query packing,
messages, result copies, joins and SOR output-cloud indexing. Export, validation
and result removal are outside timing. First-use observations are single
samples, not cold-start medians.

The inputs are the measured-source `compared.ply` files documented in
[`c2c-real-source.md`](c2c-real-source.md):

| Source | Points | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| PandaSet scene 019 | 531,908 | 12,765,915 | `08475ea4cd8e77cf4660ecf91a86efbe3380e39c34b4c8ce2d6bc2b6c938db23` |
| Autoware planning | 878,920 | 21,094,203 | `642bb9ed7ab2ea79a443a527aac19692b8c6416c4d90895231adeed996f663cc` |
| Tokyo Takanawadai | 941,933 | 22,606,515 | `dadc1295597c033e25f50f97923922cbe950ccdafb8b1ffdee7a4dbbc38020fd` |

The full input manifests, source attribution, licenses and hashes are embedded
in every report. These are subsets of measured geometry, not generated points,
independent surveys or accuracy ground truth. No builds, tests or file scans ran
during timing. CPU clocks, power, garbage collection and other applications were
uncontrolled. Initial source order was PandaSet → Autoware → Tokyo; the final
pass reversed it. Per-source trials alternate the two build orders.

## Whole-RPC results

Times below are separate wall-time medians in milliseconds. Reduction is the
median of each pair's `(baseline − optimized) / baseline`, which can differ from
the ratio of the two separate medians. Positive values mean faster. In
particular, Autoware SOR's final paired median is negative even though its
separate optimized median is lower.

| Source/pass | Normal medians, baseline / optimized | Normal paired reduction; faster | SOR medians, baseline / optimized | SOR paired reduction; faster |
| --- | ---: | ---: | ---: | ---: |
| PandaSet initial | 483.10 / 444.55 | +7.63%; 9/10 | 622.45 / 600.05 | +5.60%; 9/10 |
| PandaSet final | 597.70 / 553.95 | +3.86%; 7/10 | 650.90 / 632.35 | +1.59%; 7/10 |
| Autoware initial | 708.55 / 639.25 | +9.94%; 10/10 | 674.85 / 651.15 | +2.81%; 6/10 |
| Autoware final | 705.60 / 618.80 | +11.65%; 10/10 | 747.85 / 727.15 | −1.44%; 4/10 |
| Tokyo initial | 662.95 / 612.20 | +6.46%; 10/10 | 717.70 / 672.05 | +5.97%; 10/10 |
| Tokyo final | 715.15 / 666.60 | +7.64%; 10/10 | 819.60 / 732.65 | +9.88%; 10/10 |

## Isolated join measurements

After terminating both production worker contexts, the script indexes the same
full measured input with `Cloud.buildIndex`, partitions it, and obtains real
WASM normal and SOR outputs. It runs the original JavaScript joins and the new
helpers on identical outputs, again with five warm-up pairs and ten alternating
timed pairs. Every pair checks all mean/normal/query bits.

These diagnostics isolate JavaScript work. They do **not** measure message
transfer, production worker-pool latency, or a cold calculation. The recorded
partition, point-copy, tree, local-query, normal-kernel and cross-query times are
single sequential observations; the normal kernel includes its own tree and
Morton ordering. Adding them does not estimate parallel latency.

| Source/pass | SOR query packing medians | SOR merge medians | Normal scatter medians |
| --- | ---: | ---: | ---: |
| PandaSet initial | 16.55 / 5.85 | 41.05 / 25.25 | 44.75 / 4.35 |
| PandaSet final | 11.15 / 5.70 | 38.30 / 25.25 | 43.00 / 4.65 |
| Autoware initial | 21.95 / 7.80 | 54.15 / 31.45 | 60.20 / 6.50 |
| Autoware final | 24.35 / 10.65 | 73.90 / 46.60 | 84.20 / 7.80 |
| Tokyo initial | 24.30 / 9.10 | 74.35 / 45.80 | 62.30 / 7.05 |
| Tokyo final | 30.50 / 10.85 | 108.70 / 60.25 | 82.15 / 7.35 |

Across these six measurements, paired median phase reductions were 52.45–64.96%
for packing, 36.20–42.64% for merging, and 88.35–91.07% for normal scattering.
PandaSet used five parts with 19,275 open points; Autoware used eight with 30,948;
Tokyo used eight with 43,142. Query counts and partition outputs matched between
passes. Kernel, indexing, transfer and scheduling time can dominate the remaining
whole-RPC cost, especially for SOR.

## Validation and reproduction

- Production build (`tsc --noEmit` and Vite) passed.
- Ten Playwright tests passed: two pure join tests and eight browser tests,
  including the existing normal/filter UI, 1.2M-point parallel SOR, and prior
  scratch/reuse regressions.
- Pure join controls cover 64 part-count/k/reach/nonfinite combinations,
  detached query buffers, arbitrary f32 bits and nonzero view offsets.
- A 200,319-point browser fixture compares four parallel normal settings
  (k=3/12, up/outward) with independently assembled WASM partition outputs,
  checking f32 bits and PLY bytes. Four SOR settings (k=1/12, ratio=0.5/2)
  compare PLY bytes with the global WASM filter, including retained normals,
  duplicates and an outlier. Synthetic fixtures test correctness only.
- Six reports validated 180 production warm-up/timed pairs, 12 excluded
  first-use pairs, and 90 isolated join pairs: **282 comparisons** in total.
- Nine CLI guards passed, including invalid/repeated options, invalid repeats,
  manifest mismatch, identical builds, changed lane/WASM artifacts and attempted
  output overwrite. The existing report was preserved.
- The script checks input, manifest, helper source, WASM glue, script and both
  builds' artifact hashes before and after timing. No GPU metadata is collected.

Run from the repository root, with a preserved production build from before
this join change and the original measured input/manifests available:

```powershell
node web/scripts/benchmark-knn-join.mjs --baseline-dist notes/knn-join-baseline-20261012/dist --input notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output NEWREPORT.json
```

The default is ten timed pairs. Output must not exist. The script uses Node's
`stripTypeScriptTypes` for the isolated helper module (tested on Node 22.22.0).
Local baseline/input files are ignored, so another checkout needs equivalent
preserved artifacts and the source inputs; a current build alone cannot
recreate the historical control.

The baseline is the adopted build from
[`cpu-compute-scratch.md`](cpu-compute-scratch.md). Earlier reports retain their
historical fingerprints; this comparison changes only its main worker joins.

| Artifact | SHA-256 |
| --- | --- |
| Baseline `worker-C8ygN4MU.js` | `76d0ea3062843b6f4680820b2ed1c62a2234b491724caa1ccbd788c130b99fd0` |
| Optimized `worker-C8RuKzXn.js` | `b50fe839cf6a9de7f7e1313d749da1c21849829886b3b67cc8acd700b2374827` |
| Both `c2c-worker-Bygl9p01.js` | `74232abf342402a8d2c221a669333055bad0245b2f75f04befdeb34337d4163b` |
| Both `ca_wasm_bg-B9WDv0qt.wasm` | `390ef24b4453c8179d8fbae994226eff475318de5360a1985fffc8a47542f345` |
| Benchmark script | `fe2f6291ae30e73a69eeec51dc2e2c63eb93bedfaafa8ae2cb284a9ae25cc891` |
| Join helper source | `80cc0d63942e78323bf103682264045c12e42a62dc5aeb5e78329189fa4aed0f` |
| WASM glue | `f59eddbc4d3c0aac56066e3997aac9e1c383c614eadcb368b83f42b5df1017c6` |

Raw reports and SHA-256 checksums follow.

| Report | SHA-256 |
| --- | --- |
| [windows-autoware-planning-packed-join-final.json](../benchmarks/cpu/knn-join/real-source/windows-autoware-planning-packed-join-final.json) | `14e72b8810e66a5839819f6ef352f4a18b1d5857e0e7a442480c4f787277f58c` |
| [windows-autoware-planning-packed-join-initial.json](../benchmarks/cpu/knn-join/real-source/windows-autoware-planning-packed-join-initial.json) | `c1fe469c5f2bcfe23b6b0a6573f4e87ee79dae4ccbebbe42facfa6cc0a96cc70` |
| [windows-pandaset019-packed-join-final.json](../benchmarks/cpu/knn-join/real-source/windows-pandaset019-packed-join-final.json) | `81f5c551b0a683b2ea8f0b26b10b62caa2c54299bdb7cd35579b3dfb2d7cc712` |
| [windows-pandaset019-packed-join-initial.json](../benchmarks/cpu/knn-join/real-source/windows-pandaset019-packed-join-initial.json) | `26b5b35a368c03d07e928fe044057f8d0b9f817c9a6767d6570399382ef64130` |
| [windows-tokyo-takanawadai-packed-join-final.json](../benchmarks/cpu/knn-join/real-source/windows-tokyo-takanawadai-packed-join-final.json) | `690594bb91df08c67119377dfdc3ea2f47da04a194bcc09ed1332e0a0d4fd5a9` |
| [windows-tokyo-takanawadai-packed-join-initial.json](../benchmarks/cpu/knn-join/real-source/windows-tokyo-takanawadai-packed-join-initial.json) | `b01b418330e9a369423a9e1b2cc4f42ca4a838216369410cceadbdd681a58b43` |
