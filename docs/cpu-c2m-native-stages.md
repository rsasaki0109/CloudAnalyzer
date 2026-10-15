# Native C2M stage profiling

Parallel nearest-triangle queries account for 57.07–84.71% of the measured
native C2M calls. The two passes agree on all three measured sources. BVH
construction accounts for 10.95–31.47%, Morton preparation for 3.09–7.99%,
and result allocation/scatter for 0.40–1.34%. Further experiments should
prioritize the query body, with construction/reuse as a secondary target.
This diagnostic adds no production acceleration or default-policy change.

These shares describe the frozen source before the subsequently adopted
[sibling-bound SIMD](cpu-c2m-native-bound-pair.md). They have not been remeasured
as a full two-pass profile of that new default. The current profiling example
follows the selected sibling policy, reading its compiled-in flags once
outside command clocks; the recorded archive preserves the earlier harness.

## Scope and clocks

The standalone [profiling example](../rust/crates/ca-core/examples/profile_c2m_stages.rs)
source-includes unchanged production distance/mesh modules under local symbols
to access their private helpers without modifying the library or public API.
Production Delaunay meshing uses automatic edge rejection. Original vertex
and triangle vectors move into the local mesh without coordinate copying,
transformation or reordering. Loading, meshing and pool creation occur outside
the measured calls.

The same process and fixed Rayon pool alternate two unsigned modes:

- `whole`: the source-included production `cloud_to_mesh_par` call.
- `stages`: five timed regions following the adopted native query policy.

| Report field | Timed work |
| --- | --- |
| `bvhBuild` | Fresh `MeshBvh::new`, including adopted preparation and parallel construction |
| `mortonPreparation` | Original serial bounds, keys, key-only unstable sorting and index extraction |
| `parallelQueries` | Original indexed 4,096-point chunks, chunk-local previous hit and cached traversal stack, exact unsigned distance/square root, result collection and chunk scratch destruction |
| `allocationScatter` | Final zeroed result allocation and original serial scatter, including consumed result-chunk destruction |
| `orderBvhDestruction` | Explicit destruction of the Morton order and BVH |

The outer clock includes pool installation and dispatch. Its remainder after
subtracting the five regions is recorded as unattributed time. The final
distance vector survives the clock in both modes. Statistics, serialization,
complete-output checks and cleanup run outside the clock. Each call builds a
fresh BVH; this is not a retained-mesh workload.

Source inclusion, local symbols, closure state, compiler inlining/code layout
and the five timers can change timing. The staged query closure differs from
the production generic state wrapper while preserving the selected query
policy. Output parity establishes tested correctness, not timing equivalence
to the archived production executable. The whole/staged difference is a
diagnostic comparison, not a speedup measurement. Individual box-distance,
triangle-projection, cache-miss or allocator costs within queries are not
isolated by these clocks.

## Measured results

[Measured-source preparation](c2c-real-source.md) supplies full unchanged f64
XYZ and original source-index subsets. No generated coordinates, displacement,
noise or additional thinning is introduced. Tokyo retains its preceding
original-point preparation. These subsets are not independent surveys or
accuracy ground truth. Native PLY order differs from browser LOD order.

| Source | Reference/query points | TIN vertices | TIN triangles |
| --- | ---: | ---: | ---: |
| PandaSet 019 | 531,908 / 531,908 | 501,097 | 844,324 |
| Autoware planning | 878,921 / 878,920 | 873,771 | 1,688,821 |
| Tokyo Takanawadai | 941,933 / 941,933 | 893,804 | 1,466,179 |

Windows/i7-9750H, 12 logical CPUs and 32 GiB RAM run frozen native release
executables with fixed 12-thread pools, Rust 1.98.1/x86_64-pc-windows-msvc,
optimization level 3, LTO and one codegen unit. Twelve production source/
configuration fingerprints match the preceding archived baseline and current
checkout. The new example and controller have separate fingerprints.
Adopted compact results, stack reuse/cached bounds, indexed preparation,
bottom-up bounds and parallel construction stay enabled; all previously
rejected native candidates stay disabled.

An independent archived production executable supplies the geometry, complete
f64 distances, six exact statistics and scalar PLY oracle, then exits before
the profiling process starts. One excluded first-use pair and five
alternating warm-up pairs precede ten alternating timed pairs per source/pass.
The initial pass runs PandaSet → Autoware → Tokyo; the final pass reverses
that order with identical build/controller fingerprints. All outliers remain.
No builds, tests, scans or edits run alongside timing.

| Source/pass | Whole median, ms | Staged median, ms | Median paired staged overhead |
| --- | ---: | ---: | ---: |
| PandaSet initial | 372.13 | 367.53 | −2.27% |
| PandaSet final | 350.03 | 349.98 | −1.79% |
| Autoware initial | 704.60 | 696.74 | −1.61% |
| Autoware final | 616.33 | 620.78 | +0.72% |
| Tokyo initial | 1,792.45 | 1,782.54 | −3.29% |
| Tokyo final | 1,685.63 | 1,638.10 | −0.74% |

Paired overhead is `median(100 * (staged_ms / whole_ms - 1))`, not a ratio of
marginal medians. Its negative values do not establish an optimization.
Clocks, power, scheduling, other applications, allocator behavior and retained
pages are uncontrolled. Absolute whole/staged times shift between passes.

Stage shares below are `median(100 * stage_ms / staged_ms)` over the ten
timed trials. Independent medians need not sum to 100%; they are not ratios
of the stage medians to the total median.

| Source/pass | BVH | Morton | Queries | Allocation/scatter | Order/BVH destruction |
| --- | ---: | ---: | ---: | ---: | ---: |
| PandaSet initial | 26.40% | 7.64% | 61.91% | 1.30% | 1.92% |
| PandaSet final | 25.56% | 7.99% | 63.12% | 1.34% | 1.93% |
| Autoware initial | 31.47% | 7.98% | 57.07% | 1.15% | 2.07% |
| Autoware final | 31.17% | 7.28% | 57.14% | 1.19% | 2.11% |
| Tokyo initial | 10.95% | 3.38% | 84.56% | 0.43% | 0.70% |
| Tokyo final | 11.08% | 3.09% | 84.71% | 0.40% | 0.69% |

Median unattributed time ranges from 0.0631 to 0.0751 ms across the six
reports. Query work dominates both passes. Allocation/scatter has limited
measured headroom compared with queries or construction; this agrees with
the preceding rejected [parallel result restoration](cpu-c2m-native-parallel-results.md)
experiment. The query fraction does not predict GPU speedup: transfer,
candidate quality, exact verification and end-to-end costs would still need
their own comparison. These native profiles establish no browser, WASM,
Python-conversion or GPU performance change.

## Validation and reproduction

All 96 first-use/warm-up/timed pairs match the independent production oracle:
192 complete f64 outputs, six statistics per output and full scalar PLY bytes,
plus six excluded oracle calls. Geometry bytes, query counts and thread counts
match. An independent audit recomputes all summaries, checks region clock sums
and pair order, and ties geometry/oracle fingerprints to the preceding
indexed-corner reports. Sources, executables, compiler, controller, example,
inputs/manifests and original PCD fingerprints are checked before/after.
No GPU model identifiers are collected.

The profiling release build, controller syntax and twelve invalid CLI
controls pass, alongside native all-target Clippy with warnings denied and
workspace formatting. The restored production source/executable correctness
proof still passes. Production code is unchanged during this experiment;
the preceding 412-test native suite was not rerun as part of profiling.

Build the unchanged release examples with the parallel feature:

```powershell
cargo build --manifest-path rust/Cargo.toml --release -p ca-core --features parallel --example bench_c2m_source
cargo build --manifest-path rust/Cargo.toml --release -p ca-core --features parallel --example profile_c2m_stages
```

Archive the first executable as `bench_c2m_source.exe` in a baseline root and
the second as `profile_c2m_stages.exe` in a separate profile root. Both roots
need the same twelve source/configuration files listed by the controller
under repository paths and identical `rustc -Vv` output as `compiler.txt`.
The profile root additionally needs the example source at its repository
path. Source guards cannot identify executable contents; matching builds
matter. From the repository root, using the measured archives:

```powershell
node web/scripts/profile-native-mesh-stages.mjs --baseline-root notes/native-mesh-indexed-corners-baseline-20261013 --profile-root notes/native-mesh-stages-profile-20261013 --reference notes/c2c-real-pandaset/reference.ply --compared notes/c2c-real-pandaset/compared.ply --input-manifest notes/c2c-real-pandaset/inputs.json --output notes/NEW-pandaset-native-stage-profile.json --threads 12 --repeats 10
```

`--threads` accepts 1..64 (default 12); `--repeats` accepts 1..100 (default
ten). Existing reports/run directories are not overwritten. Other checkouts
need matching archived builds and locally prepared inputs. Native output
artifacts remain under ignored `notes/`.

- PandaSet: [initial](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-stage-profile-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-pandaset019-native-stage-profile-final.json)
- Autoware: [initial](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-stage-profile-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-autoware-planning-native-stage-profile-final.json)
- Tokyo: [initial](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-stage-profile-initial.json), [final](../benchmarks/cpu/c2m/real-source/windows-tokyo-takanawadai-native-stage-profile-final.json)
- [Profiling controller](../web/scripts/profile-native-mesh-stages.mjs)
- [Production native query policy](../rust/crates/ca-core/src/distance.rs)
- [Production BVH construction and traversal](../rust/crates/ca-core/src/mesh.rs)
