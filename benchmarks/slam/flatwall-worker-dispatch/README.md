# Automatic ICP worker dispatch: retained real recordings

See [the study](../../../docs/slam-flatwall-worker-dispatch.md) for the gate,
complete numerical/timing results, all controls and limitations.

`time/*.json.gz` are deterministic, lossless copies of four original native
JSON outputs (81,920 frames, including warmups). `catalog.json` binds their raw
hashes to original execution receipts, logs and binary identities. `summary.json`
contains every measured recording total and all case/subgroup summaries.
`source-footprints.json` resolves both 84-file native inventories to immutable
archives. The baseline's original build receipts and commands are retained.

The controller and analysis sources are archived as executed. Native validation
logs and invocation receipts are under `native-validation/`. This packet does
not include executable binaries or redistribute raw input bags. Raw/prepared
inputs and all earlier experiment artifacts remain preserved privately.

The original recordings are flatwall_01/02 from Kenji Koide (AIST),
[Zenodo 7641866, version 2](https://zenodo.org/records/7641866), CC BY 4.0.
Source acquisition and the original independent trajectory references are
documented in the [target-balance packet](../flatwall-target-balance/README.md).
The recordings are development data; no blind accuracy, full-application or
GPU performance claim follows from this CPU registration comparison.

Run `python benchmarks/slam/flatwall-worker-dispatch/verify.py` from the repository
root. It requires Python and NumPy, and does not rebuild or run native code.
Re-execution of the native controller requires the privately retained prepared
real inputs at the paths bound by `time/invocation.json`.
