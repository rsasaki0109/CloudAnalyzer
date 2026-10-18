# Fixed-state odometry stage costs

On this host, the **60 eligible real states** take 1.55 times as long to build
target-balanced H/b as fused baseline H/b with four Rayon workers. With one
worker the ratio is 1.25. The instrumented balanced decomposition spends about
**63% searching, 24% counting and 12% accumulating H/b** with four workers.
These fractions describe the staged diagnostic, with its own clocks and code
layout. Whole balanced calls are measured separately.

Four workers help the larger fine-sampling states, but make the small legacy
states slower. This identifies two useful follow-ups: reduce serial target-count
allocations/lookups and avoid unnecessary parallel work on small inputs. This
change adds measurement only; the production arithmetic and defaults are unchanged.

## Fixed-input results

Each ratio divides sums of per-state median evaluation times. Every state/thread
has ten measured blocks across two processes, with 32 identical-state evaluations
per block. Stage fractions instead pool all measured staged windows. A one/four
ratio above 1 means the four-worker run took less time.

| Workload | States | Baseline, 1/4 workers | Balanced, 1/4 workers | Balanced/baseline, 4 workers |
|---|---:|---:|---:|---:|
| All eligible | 60 | 1.795 | 1.447 | 1.547 |
| Legacy final states, eligible | 20 | 0.798 | 0.719 | 1.799 |
| Fine final states | 24 | 1.908 | 1.499 | 1.599 |
| First/last capped steps | 16 | 1.710 | 1.435 | 1.456 |

For the eligible suite, pooled staged fractions with one worker are search
77.58%, count 15.40%, accumulation 6.78%, destruction 0.15%, and clock/boundary
remainder 0.09%. With four workers they are 62.96%, 24.19%, 12.35%, 0.35% and
0.15%. The staged/whole-balanced mean ratio is 0.996 with one worker and 0.969
with four. Group-specific ratios and every block remain in `summary.json` and
the four compressed native outputs; none are filtered or overhead-corrected.

The one-worker run uses the **same parallel-feature executable** and a local
one-worker Rayon pool. It is not a separate serial-feature build. Thread placement
and clock frequency are not pinned. Small states, timer/code-layout effects,
cache residency and this single host limit generalization.

## States and scope

Keep all 48 predetermined public final snapshots from the
[target-balance trial](../flatwall-target-balance/README.md), then steps 0 and
499 from each of the eight capped traces in the
[dynamics audit](../flatwall-balance-dynamics/README.md). No sample is replaced.
Four final snapshots have only 8 or 9 thinned diagnostic points. Native begin
uses an empty ICP source below 10 points, so their original dispatch is retained
as four empty-source controls, separately from the 60 eligible states.

The first preparation incorrectly treated those diagnostic arrays as ICP inputs.
The untimed independent preflight rejected their nonzero H against the original
zero H. Original inputs, receipt, source, failing log and preflight source are
preserved. Preparation was corrected and all 64 original systems/pair rows were
verified **before the first timing process**; the compiled executable was unchanged.

Every timing evaluation reuses the same binary source/map/pose/center/threshold/
kernel. Restore maps in their stored point order and require no voxel capacity
loss. Final snapshots' ordered target coordinates match the original capture;
step inputs match the original attempted systems, including tail pairs at step
499. Search includes native transform/27-cell matching/ordered pair collection;
count includes temporary target coordinates, exact multiplicities and counts;
accumulation includes robust weights, J, H/b and Rayon reduction. Pair/count
destruction is separate. The outer staged clock retains the boundary remainder.

Use four fresh processes in order **1, 4, 4, 1** workers. Each runs one warmup
round and five measured rounds over all states. Rotate baseline, balanced and
staged method order by round plus state ordinal. Keep all 1,280 measured blocks,
256 warmup blocks and 256 untimed native validation outputs.

This measures a warm **fixed-state H/b construction**, excluding map reconstruction,
input decoding, deskew, sampling, solving, iteration-count changes, insertion,
validation, serialization and UI/GPU work. It does not establish SLAM/application
throughput or explain the earlier closed-loop runtime increase by itself.

## Verification and source preservation

```
python benchmarks/slam/flatwall-balance-stages/verify.py
python -m pytest cloudanalyzer/tests/test_flatwall_balance_stages_packet.py -q
```

The verifier independently exhausts the ordered 27-cell search on all 64 states,
checks exact coordinate multiplicities and original pair rows, and recomputes
baseline/balanced H/b with NumPy. All four processes' 256 untimed outputs and every
timing block's inventory/order/clock sums are checked. The timed batch last-result
error fields are native self-checks; individual inner-loop results are not retained.
Full raw timings, not just derived medians, are compressed losslessly.

The only change in the production odometry file is a test-only module declaration.
The ignored release test adds no library API or profiling clocks to normal builds.
Build and run details are in the frozen protocols and sources. The separate native
test executable is retained privately by hash; no installed extension is replaced.
`executed-source/` contains LF-normalized archives of all 83 local recursive Rust
source/config inputs. Raw source identities and private original bytes remain
preserved. Toolchain/dependency/test-fixture identity is not a hermetic rebuild
certificate. Historical native packets now resolve the changed odometry source to
its exact original `history/` bytes; original manifests, verifier and receipts are
also preserved. Future core work can retain these executed archives.

Derived from `flatwall_01.bag` and `flatwall_02.bag`, Kenji Koide (AIST),
[Zenodo record 7641866, version 2](https://zenodo.org/records/7641866),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The source recordings are not redistributed. References and earlier accuracy
results were already exposed; no new accuracy or held-out claim is made.
