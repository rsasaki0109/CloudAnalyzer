# Measured workloads for the next GPU experiments

The current production build was measured on three existing source geometries,
with no added coordinates or additional input thinning. Nine operation modes
cover C2C, normals, SOR, M3C2, ICP, Map quality and C2M. Only C2C has a GPU
implementation in this comparison; the other measurements identify expensive
CPU/WASM work to investigate, not demonstrated GPU speedups.

The strongest next GPU experiments are the shared k-neighbour work for normals
and SOR, and the remaining M3C2 neighbourhood/projection work. C2M is the largest
whole-operation cost, but its existing workers rebuild the full mesh BVH on
every call: retaining preparation should be evaluated before attributing its
entire cost to a GPU-search opportunity. A subsequent GPU C2M candidate-search
experiment could target the remaining query work with exact CPU verification.

Follow-up [block hints](gpu-knn-block-hints.md) and
[nearest-k candidates](gpu-knn-candidates.md) implemented GPU selection for
normals/SOR but did not produce whole-operation speedups on these workloads.
Both remain off by default. Further work should remove substantial exact CPU
verification cost; retaining C2M's repeated mesh/BVH preparation remains a
separate promising CPU target.

The subsequent [retained C2M mesh/BVH implementation](cpu-c2m-preparation.md)
is now adopted for eligible parallel jobs. Final measured reuse comparisons
improved paired whole-operation time by 29–70%, preserving exact outputs under
a 1 GiB aggregate reservation budget. Its first-use preparation remains
expensive; future GPU C2M experiments should use this retained CPU baseline.

The subsequent [GPU C2M triangle candidates](gpu-c2m-candidates.md) use that
retained CPU baseline. All measured outputs matched, but paired median
reductions were −7.10%, −10.56% and −4.34%, with 8/30 faster pairs. The
four-byte-per-query candidate path remains off by default.

The subsequent [cached C2M traversal bounds](cpu-c2m-stack-bounds.md) are
adopted for ordinary CPU batch queries. Whole retained-CPU C2M improves by
23.80–30.38% across two measured passes, with 60/60 faster pairs. Geometry
payload/reservations and tested outputs remain identical. Future GPU C2M
comparisons should include this improved CPU traversal and preparation baseline.

The subsequent [C2M query-order cache](cpu-c2m-query-order.md) remains off:
PandaSet regresses in both passes, with 38/60 faster pairs overall and an
additional query-coordinate/order payload. Adopted cached traversal bounds
remain the CPU baseline for future GPU comparisons.

The subsequent [borrowed C2M queries](cpu-c2m-query-borrow.md) remove another
WASM XYZ allocation in an experimental build, but remain off: Autoware
regresses in both passes, with only 36/60 faster pairs overall. Ordinary
queries retain their preceding owned-triples policy; traversal bounds and
geometry preparation remain adopted.

The subsequent [native parallel C2M stack reuse](cpu-c2m-native-stack.md)
is adopted for the Rust API used by Python. All six native measured-source
comparisons have positive median paired reductions of 1.48–9.88%, with
41/60 faster pairs and identical outputs. These native API observations
exclude Python conversions and are separate from browser/GPU measurements.

The subsequent [native cached C2M traversal bounds](cpu-c2m-native-bounds.md)
are also adopted. All six comparisons against native stack reuse have positive
median paired reductions of 0.58–13.43%, with 42/60 faster pairs and identical
outputs. Some differences are small and the passes vary; these are native
API observations, separate from browser/GPU results and not additive percentages.

The subsequent [native BVH preparation parallelism](cpu-c2m-native-preparation.md)
is adopted for indexed centroid calculation and ordered corner copying on
large native meshes. PandaSet/Autoware improve by 3.39–5.56% in both passes;
Tokyo is mixed at −2.04% / +6.98%. Full outputs and tested capacities match,
with 42/60 faster pairs overall. Recursive splitting stays sequential, and
these native results exclude Python conversion/browser/GPU speedup claims.

The subsequent [bottom-up C2M mesh bounds](cpu-c2m-bound-merge.md) are adopted
in the shared native/WASM builder. Finite nonzero meshes merge child extrema
at internal nodes instead of repeatedly scanning triangle vertices;
zero/nonfinite meshes retain the preceding scans. Fresh native API calls
improve by 26.07–45.36% in six comparisons, with 60/60 faster pairs and
identical outputs. Browser compatibility and exported GPU buffers also
match, without establishing browser/GPU speedup. Recursive splitting and
existing vector capacities stay unchanged.

The subsequent [native parallel BVH construction](cpu-c2m-native-builder.md)
is adopted for large meshes. Independent child subtrees write disjoint
preorder ranges without extra subtree vectors, preserving tested node bits,
triangle order and capacities. Fresh native API calls improve by
14.49–33.45% in six comparisons against the bottom-up-bounds baseline,
with 60/60 faster pairs and identical outputs. Small meshes and WASM keep
the existing sequential builder; these observations establish native CPU
performance and do not add browser/GPU speedup claims.

The subsequent [native seed-triangle deduplication experiment](cpu-c2m-native-seed.md)
remains off. Skipping a successful warm-start triangle's duplicate leaf test
preserves tested outputs, but PandaSet and Autoware change direction between
passes; only 33/60 timed pairs improve. Tokyo's small 1.90–2.52% paired gains
are insufficient to establish a general default improvement. Adopted parallel
construction and cached traversal stay enabled.

The subsequent [native Morton preparation experiment](cpu-c2m-native-morton.md)
also remains off. Parallel coordinate bounds and indexed key generation
preserve tested bound/order bits and downstream signed distances, but only
30/60 timed pairs improve. PandaSet changes by −2.71% / +0.71%, Autoware by
+5.78% / +0.49%, and Tokyo by −2.67% / −0.04%. The original serial unstable
key-only sort and 4,096-point query chunks are retained in both modes.

The subsequent [native compact query results](cpu-c2m-native-results.md) are
adopted. Chunk vectors retain only f64 distances and reuse the existing Morton
indices for original-position scatter, halving temporary result payload on
the measured 64-bit native build. All six whole-call comparisons improve by
1.68–4.86%, with 45/60 faster pairs and identical outputs. This establishes
native C2M behavior and does not add C2C/browser/GPU speedup claims.

The subsequent [native parallel result restoration experiment](cpu-c2m-native-parallel-results.md)
remains off. A temporary inverse Morton map and parallel original-order
gather preserve complete tested outputs, but all six whole-call comparisons
regress by 0.10–5.06%, with only 18/60 faster pairs. Adopted compact temporary
results and serial scatter stay enabled.

The subsequent [native push-time pruning experiment](cpu-c2m-native-eager-prune.md)
also remains off. Omitting already-pruned children from the cached traversal
stack preserves tested outputs. Tokyo improves by 1.41–2.20%, but PandaSet
and Autoware regress in both passes by 2.33–4.26%; only 23/60 timed pairs
improve. Existing compact results, cached bounds and parallel construction
remain enabled.

The subsequent [native previous-hit triangle seed experiment](cpu-c2m-native-triangle-seed.md)
remains off. Reusing one triangle's corners/edges within each original query
chunk preserves complete tested outputs, but only 24/60 timed pairs improve.
PandaSet changes by −0.78% / +0.90%, Tokyo by +1.01% / −1.67%, and Autoware
regresses by 1.60–2.48% in both passes. Existing native traversal, preparation
and compact-result accelerations remain enabled.

The subsequent [native parallel centroid bounds experiment](cpu-c2m-native-centroid-bounds.md)
remains off. Large-node parallel scans preserve tested node bits, split axes,
triangle order and capacities. Autoware improves by 2.58–3.86%, but PandaSet
changes by +2.42% / −1.75% and Tokyo regresses by 0.45–1.36% in both passes.
Only 35/60 timed pairs improve. Existing parallel BVH construction and other
adopted native C2M accelerations remain enabled.

The subsequent [native original-vertex leaf access experiment](cpu-c2m-native-indexed-corners.md)
remains off. Omitting the copied ordered-corner array saves a logical
60,791,328–121,595,112-byte buffer on the measured meshes, while preserving
tested bits/order. All six whole-call comparisons regress by 5.05–18.47%,
with only 4/60 faster pairs. The preceding copied-corner access and adopted
native C2M accelerations stay enabled; peak process memory was not measured.

The subsequent [native C2M stage profiling](cpu-c2m-native-stages.md) changes
no production policy. Two passes on all three measured sources place parallel
queries at 57.07–84.71% of staged whole-call time, fresh BVH construction at
10.95–31.47%, and allocation/scatter at 0.40–1.34%. All 192 outputs match an
independent archived production oracle. This supports prioritizing query work
for further experiments; timer/code-layout effects and full verification/
transfer costs prevent translating these shares into a GPU speedup claim.

The subsequent [native C2M sibling-bound SIMD](cpu-c2m-native-bound-pair.md)
adopts two independent f64 SSE2 lanes with NaN handling folded into clamping.
All six whole-call measured-source comparisons improve by 3.81–6.46%, with
60/60 faster pairs and full output parity. The preceding explicit-mask
variant remains off after mixed Autoware results. These native query changes
establish no browser/WASM/GPU speedup; other targets retain scalar sibling
calculations. Future native GPU comparisons should use this improved CPU
baseline and include complete verification/transfer costs.

The subsequent [native C2M paired projection dots](cpu-c2m-native-projection-pair.md)
remain off. Paired f64 SSE2 projection dots preserve tested outputs, but
PandaSet regresses by 6.78–12.24% and Tokyo by 6.14–7.29% in both passes.
Autoware changes direction, with only 9/60 faster pairs overall. The adopted
clamped sibling-bound SIMD and preceding native accelerations remain enabled.

The subsequent [native retained parallel C2M](cpu-c2m-native-retained-par.md)
adds an owned immutable BVH query API, using the same native parallel kernel
as fresh C2M calls. Repeated same-TIN queries improve by 14.09–35.21% across
all six comparisons, with 60/60 faster pairs and full output parity. Initial
construction and an additional 111–223 MB of logical retained BVH payload
per measured mesh are separate costs; first-query totals do not always improve.
Future native GPU comparisons for retained references should include this
opt-in CPU baseline and its lifetime/setup costs. Existing browser/WASM/
Python/GPU selection is unchanged.

The subsequent [Python prepared-mesh API](cpu-c2m-python-prepared-mesh.md)
exposes that native retained kernel through `cloudanalyzer_core.PreparedMesh`.
Complete Python calls including NumPy conversion improve by 11.77–48.96%
across six real-source comparisons, with 58/60 faster timed pairs and full
unsigned native-oracle parity. Construction copies geometry and retains
approximately 134–264 MB of native payload per measured mesh, in addition to
any caller arrays. Setup and first-use totals are recorded separately. The API
is opt-in; existing stateless calls and browser/WASM/GPU dispatch stay unchanged.

The subsequent [scan-match diagnostic distance reuse](cpu-scan-match-distance-reuse.md)
removes two repeated CPU nearest-neighbour searches from CLI artifact generation.
Diagnostic metrics, coloring and PLY writing improve by 9.75–17.27% across six
full-source comparisons, with 54/60 faster pairs and complete output parity.
The measurements fix an actual ICP result and exclude loading/registration;
they are not full CLI or GPU timings. The before-distance array adds 4.3–7.5 MB
of temporary payload in these cases and is released after its artifact is saved.

The subsequent [owned C2C reference API](cpu-c2c-python-prepared-reference.md)
retains a k-d tree while accepting changed query clouds. Whole Python nearest
calls improve by 54.68–61.28% across six measured-source comparisons, with 60/60
faster pairs and full preceding-wheel f64/statistics/scalar-PLY parity. Initial
construction and approximately 18–32 MB of persistent native tree payload are
recorded separately. The API is opt-in; existing CLI/browser/WASM/GPU selection
stays unchanged. Retained-reference GPU comparisons should include this CPU
baseline and account for setup, memory lifetime, transfers and verification.

The subsequent [scoped scan-match reference reuse](cpu-scan-match-reference.md)
uses that API automatically when available, sharing one private working-map
tree across registration's before/after diagnostic searches. Preparation,
both searches, statistics and cleanup improve by 20.55–24.45% across six
real-source comparisons, with 60/60 faster pairs and complete output parity.
Loading, ICP solving and PLY writing are outside those clocks. Approximately
18–32 MB of tree payload is held during registration and released before
artifact writing. Older/disabled cores keep stateless dispatch.

The later [multi-frame SLAM map reuse](cpu-slam-map-reuse.md) shares one lazy
map snapshot and an uncropped reference across selected diagnostic attempts.
Three repeated real-source attempts improve by 17.47–44.66%, with 29/30 faster
pairs and full f64/report parity. Loading is included; actual ICP solving and
artifacts are excluded. Cropped frames keep private references. Future GPU
comparisons should include this CPU baseline and its longer scoped lifetime.

The later [target-normal reuse](cpu-registration-target-normals.md) saves one
pre-solver normal snapshot per uncropped diagnostic run. Source normals still
compute per attempt. Full preparation/result-copy work is 21.05–36.37% shorter,
with 18/18 faster pairs; actual ICP iterations and loading are excluded.

The subsequent [no-artifact ICP path](cpu-scan-icp-normals.md) omits missing
normal estimation because point-to-point registration and the returned
diagnostics do not use it. Three repeated real-source attempts shorten the
diagnostic interval, including loading, by 72.16–81.47% across six comparisons
with 18/18 faster pairs. Actual ICP iterations are excluded. GICP, PLY artifacts
and ordinary registration retain normals; future GPU comparisons should use
this CPU baseline where eligible.

The later [no-op scan preprocessing copy experiment](cpu-scan-preprocess-borrow.md)
was rejected: six diagnostic comparisons range from 11.58% slower to 15.03%
faster, with only 17/30 faster pairs despite fewer coordinate copies. The
preceding CPU implementation remains the baseline for future GPU comparisons.

The subsequent [binary XYZ PLY bulk loader](cpu-ply-xyz-loading.md) improves
the eligible I/O interval by 78.01–83.25% and diagnostic assembly by
23.08–31.70%, each with 30/30 faster real-source pairs. Iterative ICP is
excluded. Complex layouts keep Open3D; future GPU comparisons should include
this CPU loading path when their input layout qualifies.

The later [requested-only comparison coloring](cpu-compare-requested-colors.md)
skips heatmap work when there is no image consumer. Six full unregistered,
numeric-only real-source comparisons are 78.45–82.21% shorter, with 60/60 faster
pairs, actual NN work and identical reports. Snapshot behavior is preserved;
registered or rendering performance is not established by these measurements.
Prepared native inputs match in actual ICP/GICP checks, while parallel solver
rerun differences remain recorded. Future GPU comparisons should include this
CPU reuse and account for its extra owned normal buffer and temporary copies.

The subsequent [scoped batch reference reuse](cpu-batch-reference.md) loads one
reference snapshot per batch and shares its native tree across candidates.
Complete three-candidate batch functions improve by 23.83–33.78% across six
real-source comparisons, with 59/60 faster pairs and full bidirectional f64/
report parity. Loading, preparation, summaries and cleanup are included;
CLI startup and report serialization are excluded. The tree retains approximately
18–32 MB of native payload through reverse searches and is released per batch.
Older/disabled cores still benefit from one reference load. Browser/GPU
selection is unchanged.

The subsequent [scoped before/after map reference reuse](cpu-loop-reference.md)
shares one loaded reference/tree in `loop-closure-report`. Complete map-only
report functions improve by 10.77–23.88% across six actual-input/ICP-output
comparisons, with 57/60 faster pairs and full directional f64/evaluation/report
parity. Loading, preparation, statistics, deltas and cleanup are included;
solving, optional trajectory/posegraph work and CLI writes are excluded. The
approximately 18–32 MB tree is released before optional sections. This measures
report cost without establishing loop-closure accuracy or GPU speed.

The subsequent [parallel resident MapEval queries](cpu-map-nn-parallel.md)
use the existing exact SciPy query with all available CPU workers for large
directions. Whole resident evaluations, including both tree builds and default
AWD/SCS, improve by 16.57–26.35% across six real-source comparisons with 29/30
faster pairs and full f64/metric/colored-PLY parity. Loading and CLI writes are
outside those clocks. Small/empty inputs keep preceding dispatch; large calls
use more CPU concurrency with unmeasured worker scratch/RSS. Streaming and
browser/GPU selection are unchanged.

The subsequent [sparse integer SCS lookup](cpu-map-scs-index.md)
indexes only supported error cells, preserving exact cube membership and both
neighbor/center reduction orders. Complete resident evaluations improve by
5.43–18.12% across six real-source comparisons against the parallel-NN baseline,
with 25/30 faster pairs and full metric/colored-PLY parity. Scoped index setup
and cleanup are included; loading and CLI writes are excluded. Recorded coordinate/
native-index arrays total 38,464–126,336 bytes, excluding nodes/containers/RSS.
Streaming uses the same exact lookup without a streaming-speed claim. Small,
radius-1 or unsafe-span cases keep grid lookup; GPU selection is unchanged.

The subsequent [resident covariance reduction](cpu-map-covariance.md)
computes six symmetric components with point-order grouped sums. Complete
resident evaluations improve by 5.13–17.64% against the parallel-NN/indexed-SCS
baseline across six comparisons, with 27/30 faster pairs and full Gaussian/
f64/metric/PLY parity. The point-product temporary decreases from 72 to 8 bytes
per point, plus per-voxel component arrays; peak RSS is unmeasured. Loading and
CLI writes are excluded. Non-finite/extreme centered values retain scatter;
the streaming accumulator and GPU selection are unchanged.

The subsequent [resident scalar voxel grouping](cpu-map-grouping.md) encodes
bounded integer XYZ keys with exact mixed-radix codes, retaining lexicographic
order and all point memberships/Gaussian bits. Complete resident evaluation
improves by 45.73–54.44% against the preceding covariance baseline across six
comparisons, with 30/30 faster pairs and full metric/PLY parity. Loading and
CLI writes are excluded. Small inputs or unsafe integer capacities keep axis
grouping; streaming and GPU selection are unchanged.

The subsequent [stable streaming voxel groups](cpu-map-stream-groups.md)
remove one full-chunk boolean scan per occupied voxel. Point batches and moment
updates retain their original order and all Gaussian bits. Complete streaming
AWD/SCS reduction improves by 69.87–82.36% across six real-source comparisons,
with 18/18 faster pairs and identical reports. Fixed 100,000-point memory-map
chunks include grouping/moment/Wasserstein/SCS work; original-file loading,
snapshot preparation and CLI writes are excluded. This is a separate streaming
lane; resident and GPU selection are unchanged.

The subsequent [optional centroid-preprocessing improvement](cpu-map-downsample-centroids.md)
removes repeated sorted-key classification, batches singleton NumPy means and
preallocates outputs. Complete core-API evaluation with requested 0.2 m
downsampling improves by 33.67–81.01% across six real-source comparisons, with
18/18 faster pairs and full centroid/f64/Gaussian/metric/PLY parity. Loading and
CLI writes are excluded. Disabled downsampling, streaming and GPU selection
are unchanged; current `ca map-evaluate` options do not expose downsampling.

The subsequent [conditional small-group centroid means](cpu-map-downsample-small-groups.md)
batch equal-length 2–8-point groups with at least 4,096 eligible groups and
bounded gathers. Complete optional API evaluation with 0.2 m downsampling
improves by 55.24–70.38% for PandaSet/Tokyo, with 12/12 eligible pairs faster
and full centroid/f64/Gaussian/metric/PLY parity. Ineligible Autoware retains
preceding means and has mixed timings (−16.50%, +2.62%); no acceleration is
claimed there. Loading/CLI writes are excluded. Error-policy/input guards,
large-group fallback and the separate resident/streaming/GPU scopes are recorded.

The subsequent [resident mean-sum reduction](cpu-map-mean-sums.md) replaces
interleaved XYZ scatter with three point-order coordinate reductions for safe
inputs of at least 64 points. Aggregation improves by 65.02–74.36% across twelve
real-source cases, with 60/60 faster component pairs and identical full outputs.
Complete resident evaluation is mixed (−2.73% to +4.10%, 18/30 faster pairs), so
the component percentage is not a whole/GPU acceleration claim. Loading/CLI
writes are excluded, and small/unsafe inputs retain scatter.

The subsequent [bounded Gaussian Wasserstein batches](cpu-map-wasserstein-batch.md)
stack eligible covariance decompositions and matrix products while retaining
the scalar formula, mean dot arithmetic and all per-voxel distance bits.
Complete resident MapEval improves by 9.05–27.16% across six real-source
comparisons, with 28/30 faster pairs and identical full outputs. The isolated
distance component improves by 87.71–91.41%; that percentage does not describe
whole evaluation or GPU acceleration. Small/unsafe batches and explicit NumPy
error settings retain scalar calls; streaming shares the helper without a
streaming-speed claim. Loading/CLI writes are excluded.

The subsequent [bounded SCS statistics](cpu-map-scs-batch.md) batch equal-length
rows while preserving center/neighbor order and scalar reduction bits. Complete
SCS work, including fresh index/query setup, improves by 10.38–31.92% with
28/30 faster pairs. Whole resident evaluation is mixed (−2.30% to +5.99%,
16/30 faster pairs); no consistent whole/GPU gain is claimed. Unsafe inputs
retain ordered scalar calls; loading/CLI writes are excluded. The shared
streaming helper has correctness coverage without a streaming-speed claim.

The subsequent [compact midpoint point trees](cpu-map-nn-midpoint.md) use
midpoint partitioning for exact point searches when both maps have at least
65,536 points, retaining compact bounds, leaf size and existing query workers.
Complete resident evaluation improves by 7.92–17.04% across six real-source
comparisons, with 27/30 faster pairs and identical full outputs. Two-direction
point-search work improves by 5.13–20.33% with 29/30 faster pairs; those scopes
are separate. Smaller/asymmetric pairs retain preceding construction, and
loading/CLI writes are excluded. No GPU or streaming speed is claimed.

The subsequent [bounded SCS vector queries](cpu-map-scs-query-batch.md) group
eligible radius queries under a worst-case returned-index budget with one
worker, retaining ordered values and all outputs. Complete SCS work improves
by 14.47–26.73%, with 30/30 faster pairs. Whole resident evaluation is mixed
(−5.08% to +9.51%, 18/30 faster pairs), with both PandaSet passes regressing;
the component rate is not a universal whole/GPU improvement. Small or
insufficient-capacity indexes retain single-center calls, and unsupported
indexes retain preceding lookup. Loading/CLI writes are excluded; streaming
has correctness coverage without a streaming-speed claim.

The subsequent [streaming accumulator reuse](cpu-map-stream-accumulators.md)
removes 52.62–68.59% of accumulator constructions while preserving every
Gaussian bit, ordered update stream and full report. Whole streaming reduction
is mixed (−11.27% to +9.93%, 20/30 faster pairs), without a consistent
whole-evaluation speedup. This allocation change adds no GPU work or worker
pool; loading/CLI writes are excluded.

The subsequent [bounded streaming moment batches](cpu-map-stream-moments.md)
reduce enough equal-size small groups together and merge them in the original
order. Complete CPU streaming AWD/SCS work is 1.66–28.90% shorter across six
comparisons, with 28/30 faster pairs and identical Gaussian/bounds bits, ordered
groups and full reports. Unsafe or insufficient groups use scalar updates.
Loading/CLI writes are excluded; future GPU comparisons should include this
improved streaming CPU baseline.

The subsequent [expanded streaming groups](cpu-map-stream-expanded.md) allow
1–128-point groups with eight groups per size and a 32,768-point gather limit.
Complete CPU streaming work is a further 7.98–18.80% shorter, with 27/30 faster
pairs and identical Gaussian/bounds bits, ordered groups and reports. This
comparison already contains the preceding small-group change. Loading/CLI
writes are excluded; future GPU comparisons should include this CPU baseline.

The subsequent [reuse of chunk verification](cpu-map-stream-finite.md) borrows
verified scalar-group coordinates instead of repeating finite checks/copies.
Complete CPU streaming work is a further 6.88–16.46% shorter, with 25/30 faster
pairs and identical Gaussian/bounds bytes, ordered groups and full reports.
Unsafe/small chunks retain filtering; loading/CLI writes are excluded. Future
GPU comparisons should include this improved CPU baseline.

[Streaming moment-merge candidates](cpu-map-stream-merge-candidates.md) were
also measured after this change. Every candidate has a negative paired median
on at least one source. Their outputs match, but they remain unadopted and do
not change the CPU baseline for future GPU comparisons.

The subsequent [CPU M3C2 projection fusion](cpu-m3c2-projection.md) adopts
ordered projection during radius traversal when exact normals are reused.
Final reused-normal paired reductions were 13.03%, 2.13% and 8.83%; fresh
fitting keeps its earlier default. Future GPU M3C2 comparisons should include
this improved retained CPU baseline and report complete verification costs.

The subsequent [fixed voxel-reference reuse](cpu-voxel-reference.md) retains
reference Gaussian statistics and lazy covariance roots when compared Cloud
identities change. Complete Map quality improved by 9–17% in the tested
measured-source resubmission workload; future GPU voxel comparisons should
account for this CPU preparation policy and its reference-reuse conditions.

Current prepared CPU C2C was faster than the four-verifier hybrid GPU path in
every one of the 30 alternating timed pairs. This supports keeping CPU as the
default for these repeated, unchanged-source workloads on this machine. It
does not establish a universally faster backend or a result for new GPU
kernels, different hardware, changing inputs or different acquisition pairs.

The subsequent [ICP full-matrix SIMD experiment](cpu-icp-plane-simd.md)
updates independent columns together on CPU/WASM while preserving the full
matrix and tested outputs. Paired whole-call reductions were −7.25%, +1.35%
and +6.62%, with 17/30 faster pairs. It remains off by default after the
PandaSet regression; this experiment introduces no GPU dispatch.

The subsequent [parallel SOR prefix join](cpu-sor-prefix.md) is adopted for
existing parallel CPU jobs. It merges only the required sorted candidates;
whole-SOR paired reductions were 3.55–10.37% across two measured passes, with
48/60 faster pairs and identical exported points. Future GPU SOR comparisons
should include this updated CPU join and its complete verification costs.

The subsequent [SOR distance-only query experiment](cpu-sor-slots.md) preserves
tested outputs but remains off: paired whole-call changes were +3.26%, −3.71%
and +0.21%, with 15/30 faster pairs. The adopted prefix join remains the CPU
baseline for future comparisons.

The subsequent [paired normal eigensolvers](cpu-normal-pair.md) are adopted
for ordinary WASM normal fits. Two points retain independent convergence
while using f64x2 arithmetic. Whole-normal paired reductions were 5.69–11.42%
across two measured passes, with 55/60 faster pairs and exact normal/PLY
outputs. Future GPU normal comparisons should include this updated CPU path
and report complete verification costs.

The subsequent [paired voxel cross-matrix eigensolver](cpu-voxel-pair.md)
preserves tested outputs but remains off. Two passes yield only 54/120 faster
pairs, with mixed standalone voxel scores and combined Map regressions.
Reference Gaussian/root reuse remains the adopted CPU voxel baseline.

The subsequent [retained voxel neighbours](cpu-voxel-neighbors.md) are adopted
for owned reference calculations. Whole voxel-score RPC improves 16.67–34.09%
across two passes, with 59/60 faster pairs. Combined Map gains concentrate in
Autoware (10.80–15.47%); PandaSet/Tokyo do not establish a whole-call gain.
Future GPU voxel comparisons should include this updated CPU lookup baseline,
its graph preparation cost and increased retained payload.

The subsequent [paired prepared M3C2 normals](cpu-m3c2-pair.md) are adopted
for fresh WASM fitting. Whole-call paired reductions are 1.42–10.84% across
two measured passes, with 46/60 faster pairs and exact tested outputs.
Matching normal-cache hits bypass fitting and retain their existing projection
path. Future GPU fresh-M3C2 comparisons should use this updated CPU baseline.

The subsequent [short M3C2 pair guard](cpu-m3c2-short-pair.md) remains off:
fresh-fitting gains reverse on two sources in the second pass, with 36/60
faster pairs overall. The adopted paired fits remain the CPU baseline.

The subsequent [retained M3C2 core selection](cpu-m3c2-core-selection.md) is
adopted for repeated eligible parallel jobs. Whole-call paired reductions are
2.29–12.00% across fresh/reused-normal cases and two measured passes, with
100/120 faster pairs and exact tested outputs. It retains at most 8 MiB of
original point indices and takes fresh coordinate/attribute snapshots for
each result. Future GPU M3C2 comparisons should include this updated CPU
preparation policy and its first-use cost.

## Inputs and controls

These are the unchanged even/odd source-index splits documented in
[measured-source C2C](c2c-real-source.md). Each source retains its attribution,
licence/provenance, file hashes and original preparation in the report.

| Source | Reference points | Compared points |
| --- | ---: | ---: |
| PandaSet scene 019 | 531,908 | 531,908 |
| Autoware planning sample | 878,921 | 878,920 |
| Tokyo Takanawadai prepared survey | 941,933 | 941,933 |

All supplied points are loaded with `keepEvery: 1`. Tokyo retains the earlier
0.1 m original-point preparation; this measurement performs no new voxel
selection or averaging. These are splits of measured geometry, not independent
surveys or registration/change ground truth.

Each operation receives a fresh analysis worker and its existing CPU pool.
Pool startup, file loading, validation/export/hashing and cleanup are excluded
from wall timings. A single first-use observation is recorded, then five
excluded warm-up rounds precede ten timed rounds. First-use observations follow
pool prewarming and loading; they are not cold-app medians.

- C2C alternates the current default CPU backend with WebGPU plus four exact
  verification workers in the same ordinary analysis context. CPU preparation,
  GPU reference/device and verifier snapshots must be reused. CPU uses five
  lanes for PandaSet and eight for the larger clouds. This compares available
  application paths, rather than isolating GPU hardware with otherwise
  identical trees, partitions and lane counts.
- Normals use k=24 and upward orientation. SOR uses k=24 and a 1.0 standard
  deviation threshold. Both retain the current prepared-neighbour policy.
- M3C2 uses 0.4 m normal radius, 0.25 m projection radius, 0.5 m half-depth and
  0.25 m original-point core spacing. `m3c2-fresh` disables normal caching;
  `m3c2-reused` exercises the automatic warm policy. Both use default prepared
  exact trees, shared search and resident-lane policy. Actual lanes are three
  for PandaSet and two for Autoware/Tokyo. The two modes run in separate
  contexts, not as an alternating cache-speedup comparison.
- ICP fits fresh unchanged moving input against a warm reference, with at most
  ten point-to-plane iterations, 0.8 overlap and no initial centroid matching.
  Reloading is excluded; fitting, applying the fitted transform, rebuilding
  the output index and preparing display buffers are included.
- Map evaluation recomputes bidirectional distances, summaries and inlier
  metrics while reusing scalar voxel scores and distance preparation. Separate
  `map-voxels` calls recompute just the voxel scores with `voxelCache: false`.
  Both use 1 m voxels and minimum occupancy five; evaluation uses a 0.5 m
  inlier threshold. These different operations are not a cache A/B comparison.
- C2M uses a full-input 2.5D Delaunay TIN made by the production mesh operation
  from the measured reference, with automatic edge rejection. Mesh construction
  is excluded. Every compared query is retained. No new vertex coordinates are
  created; duplicate XY handling and rejected triangles affect mesh topology.

Measurements ran sequentially on the existing Windows/i7-9750H machine,
Chromium 153 and a physical GPU. Adapter metadata contains only vendor,
architecture and fallback status. No build, test or repository scan ran during
timing. Other applications, CPU/GPU clocks, power, scheduling, GC and retained
heaps are uncontrolled; individual calls and diagnostic runs vary substantially.

## Uninstrumented operation times

Each entry is the median of ten complete production-worker RPCs in milliseconds.
These workers load the ordinary immutable production assets without timing
wrappers. This is a workload-size comparison, not an estimate of GPU savings.

| Operation | PandaSet | Autoware | Tokyo |
| --- | ---: | ---: | ---: |
| C2C, current warm CPU | 46.70 | 54.00 | 81.95 |
| C2C, WebGPU + four exact verifiers | 91.15 | 140.25 | 158.90 |
| Normals | 343.55 | 456.35 | 545.30 |
| SOR, including output indexing | 429.05 | 511.70 | 608.85 |
| M3C2, fresh normal fitting | 384.90 | 1,138.80 | 1,062.55 |
| M3C2, reused normals | 265.25 | 810.20 | 836.30 |
| ICP, including fitted transform/index | 346.70 | 353.70 | 143.85 |
| Map evaluation, reused voxel scores | 80.80 | 133.75 | 151.00 |
| Voxel scores, recomputed | 101.65 | 332.95 | 204.75 |
| C2M, including per-call mesh preparation | 1,952.05 | 5,235.60 | 8,242.00 |

ICP executes nine iterations on PandaSet, ten on Autoware and one on Tokyo.
The lower Tokyo ICP time therefore does not indicate better scaling with point
count. No percentage cache improvement should be inferred by dividing separate
M3C2 or Map rows. Historical reports use different builds/conditions and sometimes
uncached CPU preparation or serial GPU verification; their absolute times are
not controls for this measurement.

For the actual alternating C2C backend comparison, paired reduction is
`median(100 * (1 - gpuMillis / cpuMillis))`; negative means GPU is slower.

| Source | GPU paired reduction | GPU faster pairs |
| --- | ---: | ---: |
| PandaSet | -98.92% | 0/10 |
| Autoware | -160.62% | 0/10 |
| Tokyo | -107.28% | 0/10 |

GPU trials keep current snapshot import, validated imported-bound reuse, SIMD
and bounding-box verification, prefetch and adaptive 131,072-query batches.
Query staging, resident verification and double readback remain disabled.
Timed fallback or an unexpected verification lane count rejects the run.
Host GPU-wait medians are 12.95/20.50/7.80 ms and exact-verification medians are
61.40/93.10/116.85 ms in source order. These intervals overlap and exclude some
operation work. Host wait is not shader execution time; the intervals must not
be added or used to infer a shader-only speedup.

## Separate WASM boundary diagnostics

Fresh diagnostic contexts import the **same production worker and WASM bytes**
through a benchmark-only wrapper. After their own six excluded first/warm
rounds, three runs record synchronous JS-to-WASM export durations and nested
pool RPC windows. Every diagnostic output must match the ordinary context.
These runs are excluded from the uninstrumented wall medians above.

The following percentages are medians of the selected exports' summed duration
divided by all recorded export durations in each diagnostic run. They are not
percentages of whole-operation wall time or GPU-eligible work. Cross-worker
durations overlap, include possible preemption and allocation, and are not OS
thread CPU-time counters. RPC windows additionally include scheduling, transfers
and WASM work; they overlap and must not be added to export durations.

| Operation / selected export work | PandaSet | Autoware | Tokyo |
| --- | ---: | ---: | ---: |
| Normals: part-local neighbourhood + PCA/orientation | 99.88% | 99.89% | 99.85% |
| SOR: local and cross-part neighbourhood means | 96.46% | 97.13% | 96.93% |
| M3C2 reused: neighbourhood/projection/statistics kernel | 94.67% | 92.19% | 96.01% |
| ICP: correspondence/fitting loop | 83.61% | 79.34% | 40.69% |
| Map evaluation: prepared nearest distances | 98.96% | 98.74% | 98.82% |
| C2M: mesh preparation and nearest-distance kernel | 99.93% | 99.96% | 99.97% |

The normal export includes both exact nearest-k search and PCA. The M3C2 export
includes search, projection and statistics. The ICP export includes its whole
iteration loop. The wrapper does **not** separate internal WASM functions, so
these numbers cannot be presented as isolated search/PCA/reduction shares.
`WithBounds` in an export name also does not mean its optional bounds control
is enabled; the current default control remains unchanged.

## C2M preparation probe

The mesh workloads contain the following retained triangles after production
duplicate-XY handling and automatic edge rejection:

| Source | TIN vertices | Triangles | Source points used | Meshing voxel size |
| --- | ---: | ---: | ---: | ---: |
| PandaSet | 501,097 | 844,324 | 531,908 | 0 |
| Autoware | 873,771 | 1,688,821 | 878,921 | 0 |
| Tokyo | 893,804 | 1,466,179 | 941,933 | 0 |

A separate single-WASM control calls the exact production `meshDistances`
export with either an empty query list or the first production query slice.
Both copy and validate the full mesh and build/free its BVH. Empty calls do
no nearest queries. Five alternating warm-up pairs precede ten timed pairs.
Mesh indices match the production TIN and every queried f32 distance bit matches
the same slice of the ordinary C2M output. The unchanged input/source and
production-WASM hashes are also checked.

| Source | Queries in one slice | Empty median | Query median | Paired empty/query fraction |
| --- | ---: | ---: | ---: | ---: |
| PandaSet | 106,382 | 798.05 ms | 1,191.35 ms | 67.73% |
| Autoware | 109,865 | 2,141.30 ms | 2,827.25 ms | 76.06% |
| Tokyo | 117,742 | 2,143.65 ms | 4,291.80 ms | 53.76% |

These are preparation-cost probes, not directly subtractable components of the
full pool RPC. They include JS/WASM input copies and returned-array copies,
operate in one WASM instance without pool messages, and omit the pool's
concurrent load. Tokyo's timed query calls range from 3,549.40 to 7,201.40 ms;
empty calls range from 1,901.90 to 4,023.60 ms with a strong within-run trend.
This variability prevents a precise warm-cache speedup prediction. The empty
controls nevertheless establish substantial work before any distance query,
and the query controls leave substantial further work to investigate.

The measured pre-cache code in `parallelCloudToMesh` sends every worker the full mesh on
every call. `meshDistances` copies/validates it and `cloud_to_mesh` constructs
`MeshBvh::new` each time. Retaining one unchanged mesh/BVH with invalidation
and memory limits is therefore the first C2M experiment. GPU candidate search
should be compared against that improved CPU baseline, preserving the original
f64 distance and signed/tie semantics rather than treating all present C2M
time as GPU-convertible query work.

## Validation and next experiments

The three reports contain 300 timed ordinary RPCs, 90 timed diagnostic RPCs,
and 60 timed empty/query preparation calls, plus their excluded setup/warm-up
observations. Every call verifies applicable f32 normal/distance/LoD95/
significance bits, reported f64 distance statistics, ICP matrix/RMS/iteration/
convergence values and PLY output hashes. Map AWD/SCS retain the existing
1e-12 relative/absolute reduction tolerance. The reports contain 660 recorded
parity observations; an additional 180 diagnostic setup/warm calls run the same
checks before profiling.

Input, manifest, original source, worker/pool/WASM, diagnostic glue and benchmark
script hashes were checked against the finished reports. All three use the
same production artifacts and script. The ordinary/probe mesh hashes match.
Eight invalid-option/overwrite controls and script syntax checks passed.
This adds measurement tooling/documentation; production calculation code and
backend defaults are unchanged.

Next priorities supported by these measurements:

1. Measure a bounded retained CPU mesh/BVH against the present C2M preparation
   policy. It addresses the largest measured whole-operation cost.
2. Prototype shared GPU nearest-k candidates for normals/SOR, with exact
   original-coordinate membership and existing PCA/SOR arithmetic. Benchmark
   complete calls including transfer, verification and joining against current
   prepared CPU, not an obsolete uncached control.
3. Evaluate M3C2 neighbourhood/projection GPU work and, after CPU preparation
   reuse, C2M triangle candidates. Include variable-density neighbourhoods,
   signed/tie behaviour and required exact verification in comparisons.
4. Investigate ICP correspondences and fresh voxel scores if the earlier
   prototypes justify more GPU work. ICP's sequential iterations and Map's
   already cheap warm-cache path make isolated arithmetic throughput an
   insufficient adoption criterion.

The subsequent [ICP correspondence SIMD experiment](cpu-icp-simd.md) found
inconsistent whole-call gains on these same measured sources and retains its
scalar default. Correspondence search still warrants profiling; this CPU
experiment does not establish a benefit for GPU ICP.
The later [prior-iteration ICP radius experiment](cpu-icp-temporal.md) also
retains its default-off policy after checking the preceding production build.
Its [shared radius-state follow-up](cpu-icp-radius-context.md) also stays
disabled after slower real-source comparisons against both prior controls.

This ordering is an engineering inference from workload timings and existing
code. No new GPU kernel is implemented or accepted by this measurement, and
no percentage of all "effective GPU opportunities" is claimed.

## Reports and reproduction

- [PandaSet report](../benchmarks/gpu/opportunities/real-source/windows-pandaset019-opportunities.json)
- [Autoware report](../benchmarks/gpu/opportunities/real-source/windows-autoware-planning-opportunities.json)
- [Tokyo report](../benchmarks/gpu/opportunities/real-source/windows-tokyo-takanawadai-opportunities.json)
- [Benchmark script](../web/scripts/benchmark-gpu-opportunities.mjs)

Build the production SIMD WASM and app using the existing workflow, install the
project's Playwright Chromium, then run from `web` on a physical WebGPU device:

```powershell
node scripts/benchmark-gpu-opportunities.mjs --reference ../notes/c2c-real-pandaset/reference.ply --compared ../notes/c2c-real-pandaset/compared.ply --input-manifest ../notes/c2c-real-pandaset/inputs.json --output ../notes/NEW-pandaset-opportunities.json
```

Use the corresponding existing Autoware/Tokyo source preparation directories
for other runs. The output parent must exist and reports cannot be overwritten.
`--repeats` and `--profile-repeats` accept 1..100, defaulting to 10 and 3.
`--methods` accepts the operation names in the script; omitting `c2c` allows a
CPU-only diagnostic subset without requesting WebGPU. Measurements must remain
sequential, without builds/tests/scans running alongside them.
