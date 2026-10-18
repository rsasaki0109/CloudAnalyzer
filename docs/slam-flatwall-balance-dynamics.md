# Target-balance recurrence and native runtime

The eight capped target-balanced registrations have repeating correspondence
patterns in their final 24 steps: four period-2, one period-3, two period-4 and
one period-6. At those lags, ordered source/target IDs and multiplicities repeat
exactly and poses return within roundoff. Consecutive changes also affect the
multiplicity of sources that keep their target. All 4,000 native systems and
steps were checked independently; the original 500-step endpoints are retained.
This supports investigating bounded recurrence handling, while leaving stopping
and convergence semantics unchanged in this change.

Three counterbalanced timing pairs on every original real condition show the
current target-balanced policy is slower: about 1.398 times baseline for the
sum of case medians, with case ratios of 1.124–1.669. The native calls include
sampling, ICP and map updates. Inputs are preloaded and diagnostics disabled;
bag I/O, gyro deskew and Python/UI work are excluded. Different closed-loop maps
and iteration counts mean this does not isolate target-counting overhead or
establish CPU-thread scaling. The earlier accuracy benefit remains conditional
and the default remains off.

See the [complete packet](../benchmarks/slam/flatwall-balance-dynamics/README.md)
for original snapshots, full selected maps, every native trace system, retained
tail pair rows, complete replay/timing poses, all timing samples, provenance,
source promotion and verification. Run
`python benchmarks/slam/flatwall-balance-dynamics/verify.py`.
