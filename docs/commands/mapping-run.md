# Agent-driven generation of both maps

Use `start_mapping_run`, `inspect_mapping_run` and `advance_mapping_run` over
[MCP](mcp.md) to let Codex or another calling agent generate a point-cloud map
and a Lanelet2 road draft from a recording. Supply the log and **one explicit
layout hypothesis**. The agent examines candidates and chooses source ranges;
the runner binds that layout to every retained piece, executes geometry/lane
generation and returns diagnoses. The operator does not prepare candidate IDs,
per-piece lane JSON or individual processing commands.

The calling agent supplies reasoning. CloudAnalyzer does not embed an LLM or
make adoption decisions using a hidden ranking. No additional model API key is
required by the MCP server. The output is a draft with recorded unresolved
intervals and semantics, rather than a certified road network.

## Ask the agent to run

Install a current native core and register `ca mcp` with your agent. A request
can specify:

> Generate a point-cloud map and an HD road draft from `drive.mcap` into a new
> `runs/drive` directory. Use a single forward driving-lane hypothesis with
> one-way use, 40 km/h, fraction 1 and minimum profile width 2.5 m. Treat the
> source span and every traffic attribute as unverified. Use a six-attempt
> budget and keep the 90% full-input extent goal. Inspect candidate evidence,
> record your choices and retry autonomously within those fixed assumptions.
> Return the point map, editable IR, Lanelet2 and remaining review holds.

Use assumptions appropriate to the task. Direction, lane count, speed and
complete road width are not inferred from a trajectory or ground support.
There is no default legal traffic interpretation. If those assumptions cannot
be supplied, the [lane-free geometry workflow](mapping-job.md#adopt-source-geometry-while-lane-semantics-remain-unresolved)
can still preserve source curves for review.

## The loop

1. `start_mapping_run(source, out_dir, layout_hypothesis)` creates a new frozen
   job, generates the point map and corridor proposals and returns revision 0,
   a paged candidate index and decision guidance. Only raw inputs are accepted:
   MCAP, ROS1 bag or rosbag2 SQLite. Native processing and relevant recording
   readers must be installed. Initial point-map/proposal failures stay visible.
2. `inspect_mapping_run(job_dir, offset=0)` resumes saved observations without
   processing. Candidate index pages contain 16 entries. Page through them and
   issue `advance_mapping_run` inspect actions for relevant candidates.
3. An inspect action returns up to eight candidates with original-frame curves,
   surface evidence and observed stations. Previews contain up to 128 sections
   per candidate with explicit total/truncation flags. The inspection receipt
   persists. Draft ranges must have inspected endpoints; unreviewed tails stay
   unresolved. There is no automatic adoption or candidate ranking.
4. A draft action supplies **complete** include/defer decisions and reasons.
   It automatically saves source geometry, binds the fixed layout to each
   included piece, generates Lanelet2/IR/projector and reads all four source
   audits. It uses up to two shared HD attempts. Each draft is a replacement
   hypothesis; it does not append roads to an earlier map. Separate pieces stay
   disconnected. Previous maps and any explicit selection remain intact.
5. The agent reads width/structural failures, original input extent and both
   ground estimators, then decides another draft or finishes. The runner cannot
   change the layout or lower the extent goal through an action. Invalid
   choices spend no processing attempt. Failed native stages retain their
   reason and consume their attempt. Full source support is not accuracy truth.
6. Finish with a retained audited lane candidate, or `null` if none could be
   generated. The output includes point-map/trajectory paths, HD artifact and
   audit paths when available, a compact diagnosis, fixed assumptions and the
   complete decision-history path. It never calls `select_mapping_candidate`.
   A partial export remains `draft_needs_review`; no HD output is explicitly
   `hd_unavailable`. `deployment_ready=false` remains unconditional.

Every action supplies the inspected `expected_revision` and a reason. Stale
retries are rejected before execution. Inspection and finish actions spend no
HD attempts. The run allows 128 decisions; finishing remains possible after
that action budget is exhausted. The shared HD budget is 2–8 attempts, default
6, sufficient for up to three geometry/lane pairs when every stage succeeds.
Do not shrink required lane width, lane count or retained extent to improve a
source-support score. Semantic assumptions remain fixed hypotheses throughout.

## Inputs and actions

`layout_hypothesis` / `layout.json`:

```json
{
  "boundary_policy": "source_span_hypothesis",
  "reason": "Explicit unverified test layout; legal traffic semantics remain unresolved",
  "speed_limit_kmh": 40,
  "lanes": [
    {"direction": "forward", "kind": "driving", "one_way": true, "fraction": 1, "minimum_width_m": 2.5}
  ]
}
```

Lanes are left-to-right along original input stations; fractions sum to one.
Only driving lanes are accepted because the source-audit protocol assesses that
kind. [Lane export](mapping-job.md#export-explicit-lane-hypotheses-from-adopted-geometry)
describes supported values, profile-width constraints and virtual boundaries.
The initial policy file is hashed; editing it invalidates inspection/advancement.

Example action objects, using IDs read from the current run rather than fixed
dataset-specific IDs:

```json
{"type": "inspect", "candidate_ids": [12, 13]}
```

```json
{"type": "draft", "decisions": [{"candidate_id": 12, "action": "include", "reason": "Observed source band follows the input path; complete width remains unresolved"}]}
```

```json
{"type": "finish", "candidate_id": 2}
```

All tool calls use the `job_dir` and current revision returned by the runner.
The draft decisions use proposal IDs; finish uses the shared job attempt ID.
Geometry curve IDs and per-piece lane specifications are supplied automatically.

## Resume and inspect results

An agent can stop between actions and resume with `inspect_mapping_run`. Each
draft journals planned geometry/lane attempt IDs before processing. If a call
is interrupted after saving geometry or completing lane generation, inspect
the state and issue `{"type":"resume"}` with its current revision. Completed
stages are reused after checking their identity and hashes, without spending
attempts twice. Running native stages are not replayed blindly.

Hard process termination can leave `.mapping-lock` or `.mapping-run-lock` files.
Inspect the retained job/action and confirm its process has stopped before
manual recovery. Do not remove a live lock or retry startup into an existing
directory. Initial preparation interrupted before `run.json` exists is visible
through `inspect_mapping_job`; retain it and start a new run. This runner does
not reconstruct missing odometry/point-map work after a hard crash.

Response histories show the last eight actions, with explicit totals/limits.
Compact diagnoses retain full sample totals and at most 16 lane summaries per
audit; failed sample locations are omitted from these responses. Use
`diagnose_mapping_candidate` for the existing bounded problem-location preview,
or read the hashed complete source audit. The history file retains observations,
all action reasons, failed outcomes and the output choice. Full original station
dispositions remain in each draft report.

## CLI equivalents

```sh
ca mapping-run drive.mcap --out runs/drive --layout layout.json
ca mapping-run-inspect runs/drive
ca mapping-run-advance runs/drive --action action.json --revision 0 --reason "Inspect current source candidates"
```

These CLI commands execute one stage/agent decision and return JSON; the calling
agent continues the loop through MCP. Starting alone does not finish HD mapping.
A failed draft exits nonzero while preserving its run state and prior artifacts.
See the [two NCLT calling-agent runs](../../benchmarks/vector-map/nclt-agent-run/README.md)
for full live decision traces, unchanged layout priors and original-input extent.
