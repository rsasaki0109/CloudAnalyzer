# One worker entry per automatic ICP loop

Automatic odometry registration runs its first ICP iteration on the caller.
Only when another iteration is needed, it enters the existing Rayon pool once
for the remaining iterations if the captured policy enables target balancing
or the source exceeds the existing 256-point reduction chunk. A caller already
on a Rayon worker keeps its current pool. Empty/ineligible scans and a default
source of at most 256 points stay on the caller. Scans that converge or fail
on the first attempted iteration also stay on the caller. Filtering and map updates stay
on the caller in every case. The serial build and the public stepped API retain
their execution behavior.

The worker entry is kept in a separate function so the small caller loop stays compact.
This removes repeated caller-to-pool handoffs during the loop. Correspondences,
exact multiplicities, pair order, weights, reduction chunks, H/b arithmetic,
solving, stopping, thresholds, diagnostics and defaults retain their definitions.
The change adds no pool, worker-count setting or public option.

## Real recording comparison

The [packet](../benchmarks/slam/flatwall-worker-dispatch/README.md) retains four
fresh native processes in reference/candidate/candidate/reference order. Each
process replays all 16 original cases in both modes, with one warmup and three
measured rounds. These are the same real flat-wall recordings, raw/gyro inputs,
motion policies and sampling modes as the preceding studies. They are reused
development data, with no blind holdout claim. An earlier native candidate
reduced balanced time 15.4% but increased legacy default time 6.6%; its
entire inputs, outputs and source snapshots remain preserved privately.
A second outlined-entry candidate reduced balanced time 12.4% but increased
legacy default time 8.3%; all of those records also remain preserved.
The final gate waits until repeated work is confirmed, with no case selection
or replacement of earlier measurements.

For each case/mode/implementation take the median of six measured complete
recording totals, then sum case medians. Ratios are candidate/reference elapsed
time; values below 1 mean less time. The default column means target balancing
is disabled; its fine-source dispatch also changes under the stated gate.

| Complete native registration | Cases | Balanced ratio | Default ratio |
| --- | ---: | ---: | ---: |
| All conditions | 16 | 0.880990 | 0.919128 |
| Legacy sampling | 8 | 0.886144 | 0.966613 |
| Fine sampling | 8 | 0.880176 | 0.911216 |

Balanced registration case ratios range from 0.843855
to 0.927997; 16 of 16 case medians
decrease. Default ratios range from 0.748385 to
1.106282; 12 of 16 decrease. All
case totals, warmups, individual frame times and slower outcomes are retained.

These are single-host, four-worker observations with uncontrolled CPU placement
and frequency. Clocks cover complete native register calls on preloaded scans,
including preparation, the gate and dispatch, ICP, solving and map insertion.
Bag decoding, gyro deskew/preparation of inputs, validation and output are outside
clocks. These results do not establish GPU, whole-application or accuracy gains.

## Numerical and source checks

All 81,920 replay frames preserve original pose and learned-threshold bits,
iterations and termination reasons. The independent verifier checks the original
checked-in real trajectory arrays bit for bit. Both binaries also match each
other's complete numerical frame records.

The baseline reuses the preserved preceding candidate executables after checking
all 84 native source files, toolchain versions and build commands. Original build
inputs, receipts and logs remain bound to the reused bytes. The new candidate is
built from a fresh 84-file snapshot. The only native source change is
`odometry.rs`; unchanged source archives are reused through explicit footprints.
Two line comments were corrected after measurement. The executed source remains
archived; `source-promotion.json` binds the final source to that exact comment
correction. It permits no statement changes.
All original private outputs, input recordings, prior trials and failures remain
preserved. No installed Python extension is replaced.

Native tests cover the 256/257 boundary, captured per-registration policy,
empty/ineligible sources, nested one/four-worker pools, ordinary and supplied
motion predictions, both correspondence policies and small/large sources.
Automatic calls match manually stepped pose bits, reports, maps, threshold
history and captured systems. Serial/parallel all-target clippy and rustfmt are
checked separately. This is a targeted release-test claim; the preceding study
retains three unchanged Windows release NaN bit-comparison failures in ICP/mesh.

Run `python benchmarks/slam/flatwall-worker-dispatch/verify.py` to check hashes,
executed source/build/invocation bindings, every original replay and every
derived timing summary. The regression tests also reject a one-bit pose or
threshold change, altered stopping results, omitted frames, relabeled warmups,
invalid clocks, changed summaries and a re-signed altered source archive.
