# mapping-inspect-glim

Recheck a saved GLIM run without Docker, GLIM or a new mapping attempt:

```bash
ca mapping-inspect-glim runs/glim-session > inspection.json
```

The command reads a regular `glim-run.json` of at most 4 MiB and checks the
SHA-256/size of every recorded bag/configuration input at its original path.
When a container command was recorded, the saved `config/` must also match the
complete recorded configuration inventory, including file names. A recorded
trajectory is checked at `dump/traj_lidar.txt` against its hash and the shared
bounded TUM validator. It never follows commands or launches subprocesses.

The JSON result separates current file integrity from historical execution:

- `status`: `recorded_files_match` or `changed_or_missing`.
- `recorded_execution_status`, `recorded_exit_code`, `recorded_image_id`: values
  from the original run, without relabelling a failed/blocked run as successful.
- `recorded_failure_stage`, `recorded_error`, `recorded_cleanup_error`: retained
  execution/preparation failure details when present.
- `verified_file_count`, `issue_count`, `issues`: matching files and discrepancies;
  all issues are counted and the first 32 are shown.
- `trajectory_check`: current pose count, time span and path length when valid.
- `report_sha256`: identity of the report being inspected, not an authenticity
  signature. The report is checked again for changes during inspection.

Exit 0 means the recorded files checked by this command match, even if the
original execution was blocked or failed. Exit 1 means missing/changed files or
invalid report metadata. No files are rewritten. Missing original inputs remain
visible failures; for portable review without original paths, export the finished
mapping review ZIP and use `mapping-bundle-inspect` instead.

This command checks individually recorded original input files. New unrecorded
files added to original directories, process logs, unrecorded/invalid partial
trajectories, other dump files and Docker image contents are not verified.
Hash agreement is not proof of estimator execution, calibration, common-state
PLY export or accuracy. A failed run with no recorded trajectory can still have
matching input files; its failure status and null trajectory check remain explicit.
