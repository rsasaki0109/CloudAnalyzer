# Mapping review improvements

Generated-map packages now show saved extent gates, held traces, estimator
agreement, Lanelet2 round-trip agreement, export warnings and unverified
assumptions beside the existing audit selector. Missing evidence is unavailable,
not a pass. The existing source-coverage viewer locates audit problems.

**Generation decision history** shows the verified package's recorded actions in
sequence, with status, reason, run candidate IDs and include/defer source
intervals. IDs belong to the recorded run, not current map lanes. History stays
available after map edits and when a project restores the original review ZIP;
it always describes the original generation, not subsequent workspace edits.
The original package remains downloadable for complete evidence.

**Show saved trajectory** overlays the package's verified KITTI pose translations
in cyan, with a green start and orange end. **Fit saved trajectory** frames the
bounds of all input poses. Coordinates stay in the package's original frame;
there is no alignment or map modification. Turning the overlay off removes it
from the scene and its framing bounds. Map edits or point-source changes clear
and disable it; reopening the original package restores availability. Restoring
a project retains the original archive but does not automatically apply its
trajectory to edited workspace geometry.

Trajectory display accepts 2–100,000 finite KITTI 3×4 poses, up to 16 MiB, with
proper orthonormal rotations (tolerance 0.001) and translations within ±1e12.
It displays at most 10,000 evenly indexed positions including both endpoints;
shown/source counts are explicit. The line joins those positions and does not
prove road connectivity, coverage or accuracy. Unsupported or malformed data
leaves the delivered maps accessible and shows why trajectory display is
unavailable. The original trajectory stays in the downloadable archive.

The display accepts `cloudanalyzer.mapping_run.v1`, at most 8 MiB of history and
128 actions. It shows at most 256 interval decisions per action, with shown/total
counts. Missing, unsupported or malformed history is explicitly unavailable;
it does not imply a successful generation or prevent opening an otherwise valid
map package. ZIP member hashes are still verified before any history is used.
Reasons render as text, including any markup in source records.

**Show source interval** links an include/defer decision to its original corridor
proposal. The action's `proposal_sha256` must match the verified packaged
`source_proposal` bytes. Candidate IDs alone never establish this link. Supported
proposals use `cloudanalyzer.corridor_proposals.v1` in `local_slam_metres`, within
8 MiB, 2,048 candidates and 100,000 total sections. Ordered sections must cover
the candidate endpoints. Missing revisions, unsupported frames and out-of-range
intervals disable the link with a reason; there is no extrapolation.

The viewer frames and highlights the original left/right source boundaries in
amber, retaining interior sections and linearly interpolating requested interval
endpoints. This is source geometry, not a validated road lane. **Clear source
interval** removes the highlight. Map/source edits clear it and disable links;
restored project archives require reopening the original map pair. Geometry is
constructed only when a link is selected, avoiding work for collapsed history.

Project metadata and browser recovery retain survey judgments and lane follow-up
snapshots. A restored survey can resume after the existing exact-content check
against a rerun comparison. Restored lane tasks remain locked until an explicit Resume saved task action verifies the comparison, map and complete recorded evidence. They do not silently attach to reused lane IDs. Saving includes committed task decisions;
press the task's Save button before leaving its editor. Old projects still load.

External SLAM producers can preflight their existing PLY/TUM exports without
creating a mapping job:

```python
from ca.mapping_import import inspect_slam_inputs
report = inspect_slam_inputs("map.ply", "trajectory.tum", provenance)
```

Use the same explicit provenance object as `ca mapping-import`. Preflight checks
finite geometry, strictly increasing timestamps, unit quaternions and import
size limits, and reports bounds, pose count and path length. It does not verify
common-frame truth, source coverage or independent accuracy. Import saves this
report in `slam-import.json`. The [GLIM preparation adapter](commands/mapping-prepare-glim.md) now preserves the documented global LiDAR export and binds its identities before import; it does not run GLIM.

Summarize recorded real-log mapping runs without regenerating their maps:

```sh
PYTHONPATH=cloudanalyzer python -m ca.mapping_scorecard benchmarks/vector-map/nclt-motion-pairs/april
PYTHONPATH=cloudanalyzer python -m ca.mapping_scorecard runs/new --baseline runs/old
```

Comparisons require the same recorded source and layout SHA-256 identities.
They report signed deltas, not an automatic quality verdict. Missing measurements
remain null; recorded action count is not human intervention count. This reads
saved metadata, not fresh physical input validation or an end-to-end timing run.

## Measure a fresh real-log run

Install `cloudanalyzer[benchmark]` (adds psutil), and run from the repository root
with the native core and recording readers available:

```sh
ca mapping-measure --run-dir runs/april -- \
  python scripts/mapping_e2e_coverage.py \
  web/public/samples/nclt-2012-04-29.mcap runs/april 2.5
ca mapping-scorecard runs/april
```

The output directory must be new. The synchronous child command inherits its
normal environment and streams its logs to the terminal. The wrapper does not
change its processing settings. The coverage script now finishes the run with
its audited draft, preserving quality/semantic holds; no audited draft exits
with code 2 and is finished as HD unavailable.

`measurement.json` records launch-to-observed-exit wall time (including child
imports), sampled process-tree RSS, logical output bytes before writing that
report, process exit status, command/script hashes, platform, and final run/job
hashes. The default sampling interval is 50 ms (`--interval .05`). RSS samples
can miss short-lived processes/peaks and count shared pages more than once.
Logical output bytes exclude external inputs and symlinks and are not peak disk
use. Final report hashing and inventory are outside the timed command. Use
synchronous commands; work detached beyond the parent lifetime is not measured.

Failures retain their available measurements and error/exit information, but
the scorecard only exposes performance fields for successful processes with
matching final run/job hashes and a supported protocol. Further run/job edits
mark the measurements stale. Runtime success does not assert map quality.
Performance deltas require matching generation status, sampling protocol and
interval as well as source/layout identity; compare the recorded commands and
platforms when interpreting differences. They are observations, not controlled
hardware-normalized speed claims.

Human intervention counts and independent accuracy stay null until explicitly
measured. Recorded actions are not a proxy for human work. Remaining work:
GLIM estimator execution and independent accuracy/intervention measures.

### GLIM execution entry point

Added `ca mapping-run-glim` with an explicit local Docker image and sensor-specific
configuration. It records immutable image identity, input hashes, a configuration
snapshot, process logs and failure status; timeouts remove the run container.
See [command details](commands/mapping-run-glim.md). Contract tests pass, but the
actual GLIM image could not be installed in this environment because layer
extraction exhausted disk capacity. Real estimator execution and independent
accuracy evaluation remain outstanding.

### GLIM sensor input preflight

Added `mapping-check-glim` for explicit MCAP points/IMU topics, availability
flags, finite IMU vectors, header timing/coverage, basic point structure and
frame IDs. Decoder failures block the report. The original NCLT demo bags failed standard rosbags 0.11.9 decoding;
this was fixed by adding complete ROS2 schemas. Their producer still marks
inertial measurements unavailable. Tests cover real demo files and correctly serialized
ROS2 IMU fixtures. This does not establish calibration or real GLIM accuracy.
See [diagnostic scope and findings](commands/mapping-check-glim.md).

### Standard ROS decoding for the NCLT demo bags

Replaced empty MCAP schemas with complete standard ROS2 message definitions in
both shipped samples and their generator. All 3,869 message payloads, topics,
channel metadata, sequences and timestamps were compared before/after and match.
Standard ROS decoding succeeds for every message; native scan/IMU decoding is
unchanged. The repair utility writes a new file and refuses nonempty or unknown
schemas. Container hashes change, so old benchmark hashes still refer to the
original containers and are not rewritten. GLIM input requirements remain unmet.

### Separate point/raw-IMU recording preparation

Added `mapping-prepare-glim-recording`: copies explicitly selected PointCloud2
and Imu channels into one ROS2 MCAP without changing payloads or timestamps,
records source/output hashes and runs the sensor preflight. Invalid IMU inputs
remain blocked with a retained report. Tests use real point payloads and
synthetic IMU fixtures; original raw NCLT inertial data is not present locally,
so real GLIM execution/accuracy is still outstanding. Frame/clock calibration
is not inferred by joining recordings.

### IMU gap detection

The GLIM input diagnostic and recording join now enforce an explicit maximum
IMU header interval (50 ms by default, configurable with `--max-imu-gap`).
Reports include effective integer-nanosecond policy, maximum observed interval,
complete excess count and up to 32 gap examples. This closes the case where
endpoints cover the point stream but intermediate inertial samples are missing.
Tests cover a 1 ns boundary at epoch timestamps, bounded examples, invalid
limits, CLI overrides and policy propagation through the recording join.

### GLIM output trajectory validation

A zero-exit GLIM process is no longer considered successful merely because it
wrote a nonempty trajectory. The runner and importer share bounded TUM checks
for finite poses, strictly increasing timestamps, normalized quaternions and
finite path length. The run receipt records the validated trajectory snapshot
and rejects changes between validation and final hashing. Malformed/oversized
outputs return `output_invalid` while preserving logs and failed-process status.
Tests use substituted processes and known trajectories; no real estimator run
or accuracy claim is implied.

### Execution-to-import traceability

`mapping-prepare-glim --run-report` now checks the successful runner report
against the exact global LiDAR trajectory and records the immutable image ID.
The report is copied byte-for-byte into the preparation directory, hashed in
provenance, and included in the mapping job's immutable dependencies. Failed or
inconsistent reports and modified report snapshots are rejected. This is local
content traceability, not proof of execution or same-state PLY generation.

### Portable external-SLAM provenance

Full and preview review exports now include available `slam_import`,
`source_trajectory` and `glim_execution_report` artifacts from the delivered
point-map owner. Existing hash checks cover these optional members in both
Python and browser readers. Integration coverage imports a linked GLIM fixture,
drafts/finishes, exports both ZIP variants and verifies them after removing
the external evidence files. No estimator is run and no original paths are
dereferenced during portable inspection. Older packages remain supported.

### Browser producer summary

The Original SLAM producer panel shows declared backend/version from a verified
import record. It shows the GLIM image ID and recorded success only when the
import's source/report identities and both run trajectory identities match the
packaged members. Missing, malformed, unsupported or inconsistent optional
evidence leaves maps usable with an unavailable notice. Each displayed JSON
record is limited to 4 MiB and producer strings to 2,048 characters; values render
as text. Summaries always describe the original archive, including after map
edits or project archive restoration, and do not claim calibration or accuracy.

### PointCloud2 input structure and coordinate checks

GLIM preflight now validates point strides, row padding, field bounds/types and
XYZ layout before inspecting finite coordinates. Inspection supports both byte
orders, organized clouds and FLOAT32/FLOAT64 XYZ within explicit message limits.
Malformed layouts, all-invalid clouds and dense-cloud nonfinite contradictions
block the report. Mixed nonfinite points in nondense clouds are counted and
warned about without rewriting data. Tests cover real demo recordings, padded
organized fixtures, limits, malformed layouts and continued inspection after a
bad message. This does not establish physical calibration or GLIM accuracy.

### Input preflight integrated with GLIM execution

`mapping-run-glim` now accepts paired points/IMU topic options and an IMU-gap
limit. It diagnoses MCAP inputs before contacting Docker, records blocked-input
results without launching a container, and binds a passing report to the exact
input inventory. Runs without the topic options explicitly say `not_requested`.
Runtime/image lookup failures retain their diagnostic receipt. Tests verify
that the real NCLT demo is blocked before any Docker call, while simulated
passing runs retain their policy and input hash. Sensor configuration and real
GLIM accuracy remain unverified.

### Review display of recorded input preflight

The original-producer summary now distinguishes preflight recorded as passed,
explicitly not requested, absent in older reports, and inconsistent/unavailable.
Pass summaries require an exact diagnostic-input/run-inventory hash match and
consistent saved gap thresholds/results; the UI shows topics, threshold, maximum
observed IMU interval and warning count. This is a recorded diagnostic summary,
not a new scan of raw data or a GLIM calibration/accuracy claim. Maps remain
usable when optional diagnostic details cannot be interpreted.

### Match diagnostic topics to GLIM configuration

Integrated preflight now reads GLIM's comment-bearing JSON and follows its
`global.config_ros` selector. Diagnostic point/IMU topics must match the selected
`glim_ros` settings (or documented bag-reader defaults), otherwise the runner
stops as `config_blocked` before contacting Docker. Matched config file hashes
are saved. The copied configuration and recording inventory are also checked
before container launch. Topic matching does not validate calibration, sensor
scaling, clock offsets or module compatibility. Tests cover comments/string
literals, duplicate keys, malformed configs, path bounds, mismatches and
configuration changes during image lookup.

### Read-only GLIM run inspection

Added `mapping-inspect-glim` to recheck recorded input hashes, the copied
configuration inventory and the validated output trajectory without Docker.
It reports changed/missing files while retaining the historical execution
outcome; matching files never convert a blocked/failed run into a success.
Original input paths remain required. Portable review ZIP inspection is the
separate workflow when those inputs are unavailable.

### Terminal records for GLIM preparation failures

Image lookup now has a 30-second timeout. Lookup interruption, configuration-copy
errors and late cleanup/verification interruption retain explicit terminal states
and failure stages. Run JSON uses atomic temporary-file replacement to avoid
truncating the previous receipt on update failure. The read-only inspector
exposes failure stage and error details separately from current hash checks.
Filesystem write failure or a hard kill can still leave the last saved state;
no recovery or power-loss guarantee is implied.


### Configuration-topic evidence in portable review

The producer panel now displays a recorded configuration-topic match only after
input-preflight validation, matching point/IMU names, and matching both config
file paths, hashes and sizes against the saved run inventory. Missing older
checks are labelled not recorded; inconsistent optional settings do not block
map loading or erase independently consistent input diagnostics. Original paths
are compared as strings and are never accessed. This does not verify calibration
or rerun GLIM. Regression cases cover topic/identity/schema mismatches, missing
and malformed inventories, and suppression after inconsistent input preflight.


### Bounded regular-file GLIM inventories

Input enumeration now stops at the entry limit rather than collecting the whole
tree first. It rejects special files and directory symlinks before hashing or
copying and propagates directory access errors. Tests cover FIFO rejection before
Docker/output creation, symlinks, socket/device stat types, directory-count limits,
stable nested-file ordering and unreadable traversal.


### Consistent immutable GLIM image identities

The runner now requires a complete Docker SHA-256 image ID, matching the import
and portable-review checks. Malformed, truncated, nonhex or multiple IDs retain
an image-lookup failure receipt before configuration copy or container launch.
Orchestration fixtures now use a complete synthetic image ID; they remain mock
contract tests, not evidence of actual GLIM execution.

### Reject numeric overflow in GLIM configuration JSON

Integrated topic checks now reject floating-point exponent overflow as well as
explicit NaN/Infinity in either selected configuration file, including nested
arrays and settings. Tests retain finite boundary values, numeric-looking text,
comments and original bytes, and verify a blocked receipt before Docker access.
This does not validate calibration or sensor-specific numeric ranges.

### Complete original input inventory inspection

New GLIM receipts record absolute input roots. Read-only inspection now compares
the complete current file inventories against the saved lists, detecting added
configuration/bag files as well as changed or missing files. Old receipts remain
readable and explicitly report that the complete inventory check was not recorded.
The inspector does not launch Docker or rewrite any evidence.

### Unambiguous execution receipt JSON

Inspection and GLIM export preparation now share the same bounded receipt reader.
It rejects duplicate object keys, nonfinite numbers and parser nesting failures,
so a conflicting duplicate execution status cannot silently override another.
Valid raw bytes are preserved for hashing and export. This applies to the Python
receipt workflows; it does not add authenticity guarantees to local records.

### Interrupted recording preparation receipts

MCAP preparation now records interruption and failure stages for writing,
input/output verification and sensor preflight. Receipts use atomic replacement;
exceptions still propagate and partial recordings remain available for diagnosis.
Tests cover write/preflight errors and Ctrl+C, unchanged inputs, retained completed
output identities and preservation of the previous receipt on replacement failure.

### Unavailable output size stays explicit

The measurement wrapper now returns a null logical output size with an error
when its output root is missing/invalid, traversal encounters an unreadable
subdirectory, or a special file is found. It no longer treats a partial directory
scan as a complete byte count. The 100,000-entry limit includes directories and
symlinks; symlink contents remain excluded. Other successful process metrics stay
available, while scorecards retain null size and omit numeric size deltas.

### Bounded cleanup after measurement failure

A monitoring error after process launch now stops the launched process instead
of waiting indefinitely for it. Ctrl+C uses the same cleanup path: SIGTERM to
its process group on POSIX (direct termination elsewhere), a five-second wait,
then SIGKILL/direct kill and one final five-second wait if needed. Signal errors,
timeouts and cleanup interruption are retained separately as `cleanup_error`;
failed/interrupted measurements remain excluded from successful scorecards.
The exit code describes the direct child; detached descendants or descendants
that outlive a terminated group leader are not independently confirmed stopped.

### Measurement diagnostics in scorecards

Scorecards now include `measurement_diagnostics` with the recorded exit code,
measurement error, cleanup error and output-size error. These fields are exposed
only when the measurement schema/protocol and run/job hashes match; stale or
missing measurements have null diagnostics. Text is capped at 4,096 characters
per field. The original receipt retains the full text.

A contradictory success receipt with a measurement/cleanup error does not supply
performance metrics. An output-size error suppresses size and its delta while
retaining independently available timing/memory metrics. Recorded process status
is preserved; diagnostics do not turn a failed run into successful evidence.

### Damaged optional measurement reports

Scorecards keep valid generation summaries available when `measurement.json` is
malformed, unreadable, oversized or too deeply nested. Such records report
`measurement_status: invalid_report` and a bounded `measurement_report_error`;
performance values and bound measurement diagnostics stay null. No report is
rewritten. Required run/job metadata still fails explicitly when invalid.
Metadata reads are bounded to 32 MiB even if the file grows after its size check.
Metadata nesting is limited to 128 container levels independently of the Python
JSON parser, with the root object counted as level one.

### Explicit performance comparison eligibility

Scorecard comparisons include `performance_comparison.eligible` and `reasons`.
Performance deltas require known matching generation stages, successful current
measurements, the supported measurement protocol and valid matching sampling
intervals. Two missing generation stages are no longer treated as an agreement.
Generation metrics remain available when performance comparison is unavailable;
individual unavailable performance values stay null even in eligible comparisons.
Eligibility does not establish identical hardware, system load or accuracy.

### Optional command timeout for measurements

Use `python -m ca.mapping_measure --run-dir runs/new --timeout 600 -- COMMAND`
to stop a still-running command after the configured elapsed duration. The option
is positive finite seconds; omitting it preserves unlimited execution. Receipts
record `timeout_seconds` and `timed_out`, and the CLI exits nonzero. Timed-out
measurements do not supply successful performance comparisons.

The monotonic deadline is checked between memory samples; sleeps are shortened
near the deadline. Process creation or a blocking monitoring call can delay its
observation, and bounded termination waits add up to ten seconds afterward.
This is not a strict wall-clock deadline. Cleanup errors remain explicit and the
existing process-group/descendant limitations still apply.


### Standard CLI for scorecards

`ca mapping-scorecard RUN [--baseline BASELINE]` now exposes the existing read-only
summary and comparison workflow through the main CLI. Successful JSON output
returns zero even when optional measurements are unavailable; missing required
metadata and incompatible baselines return nonzero. Integration tests exercise
real saved fixtures, comparisons, damaged optional evidence and unchanged files.
See [command documentation](commands/mapping-scorecard.md).


### Standard CLI for mapping measurements

`ca mapping-measure --run-dir RUN [--interval SECONDS] [--timeout SECONDS] -- COMMAND`
now exposes measurement through the main CLI. Child flags and paths pass through
unchanged without shell interpretation. Tests exercise actual child argument
receipt, failures, timeout, invalid options and existing-output preservation.
See [command documentation](commands/mapping-measure.md), including inherited
child logs and the machine-readable receipt file.

### Atomic measurement receipt publication

Measurement receipts now use a flushed same-directory temporary file and an
exclusive hard link, so readers see complete JSON and child-created receipts
cannot be replaced. Save failures and handled interruptions remove temporary
files. Regression coverage checks partial serialization, flush/publication
failures, interruption, existing files and dangling symlinks. Filesystems without
hard-link support fail explicitly; abrupt termination may leave a temporary file.
