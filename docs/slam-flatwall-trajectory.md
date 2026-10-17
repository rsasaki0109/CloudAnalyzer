# Conditional Flatwall trajectory evaluation

The fixed 185-scan nominal-motion frontend has a large mismatch with the
selected Flatwall reference under the declared body/time interpretation:
183 paired positions have 34.929240 m RMSE after one rigid position fit;
two tail scans lie beyond the reference. This adds an evaluation tool and
records a failed diagnostic baseline. It changes no estimator or defaults.

`scripts/flatwall_trajectory_reference.py` provides strict assumed-TUM parsing,
exact integer timestamp association, SO3 interpolation, proper rigid position
alignment and consecutive-original-scan SE3 errors. Parsing and interpolation
return independent pose buffers and do not mutate input arrays. Rank-deficient position alignment
is explicitly unavailable; first-pose alignment remains defined.

`scripts/audit_flatwall_trajectory.py` checks a frozen input protocol and source
manifest before parsing the reference, validates all original scan identities,
and writes a new output directory. It checks input hashes again on completion.
It records failures in place and rejects reuse of an output directory.

```console
python scripts/audit_flatwall_trajectory.py --protocol path/to/protocol.json --controller-freeze path/to/controller-freeze.json --out path/to/new-results
```

The protocol schema is `cloudanalyzer.flatwall_trajectory_exploratory.v1`.
Its `input_artifacts` entries are checkout-relative paths and SHA-256 hashes
for the archive, recording, original frontend trajectory/receipt, estimate
schema and reference-exposure record. The controller freeze supplies
`protocol_sha256` and `source_files_sha256`, including both evaluator modules.
The original frontend receipt also pins its per-scan and map output files.
Output must stay outside frozen input directories. The
[historical protocol](../benchmarks/slam/flatwall-trajectory/protocol.json)
shows the complete pinned policy and required private layout.

Decimal seconds must represent exact int64 nanoseconds. Reference timestamps
are converted from decimal digits without context-dependent arithmetic, so long
subnanosecond tokens cannot be silently rounded. An explicitly post-inspection
parser correction reproduces all original reports and paired/error arrays;
both source versions and execution records remain in the evidence packet.
Reference timestamps
strictly increase; quaternions must lie within the declared unit-norm tolerance.
An exact reference row or at most a 100 ms enclosing interval is required.
Every estimate has a coverage reason; unmatched poses have no numerical errors.
There is no extrapolation, clock fitting, scale correction, body-axis search,
window selection or pose repair. Read the [evidence packet](../benchmarks/slam/flatwall-trajectory/README.md)
for interpolation, alignment, conditioning, independent evo checks, source
provenance and the earlier reference-inspection deviation.

The text layout and reference body/clock mapping remain explicit assumptions.
The frontend input is an uncalibrated nominal IMU state. Therefore these
conditional arithmetic results establish neither calibrated physical accuracy
nor equivalence to GLIM. They identify a large baseline problem to investigate
before making estimator or performance claims.
