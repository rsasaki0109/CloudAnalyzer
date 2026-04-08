# CloudAnalyzer Perception Artifact Comparison

This static report compares a geometry-first non-deep baseline and a higher-fidelity deep baseline on the same public RELLIS-3D frame.

Both candidates are deterministic demo artifacts derived from the reference scene for documentation purposes; they are not archived model outputs.

## Summary
- Reference: reference_scene.pcd
- Files: 2
- Mean AUC: 0.8081
- Mean Chamfer: 0.0673
- Best AUC: candidates/deep_baseline.pcd (0.9515)
- Worst AUC: candidates/nondeep_baseline.pcd (0.6648)
- Best Chamfer: candidates/deep_baseline.pcd (0.0252)
- Worst Chamfer: candidates/nondeep_baseline.pcd (0.1094)

## Quality Gate
- Min AUC: 0.9000
- Max Chamfer: 0.0500
- Pass: 1
- Fail: 1

## Results

| Path | Points | Chamfer | AUC | Best F1 | Threshold | Status |
|---|---:|---:|---:|---:|---:|---|
| candidates/deep_baseline.pcd | 34370 | 0.0252 | 0.9515 | 1.0000 | 0.10 | PASS |
| candidates/nondeep_baseline.pcd | 16906 | 0.1094 | 0.6648 | 0.9735 | 0.30 | FAIL |

## Inspection Commands

- candidates/deep_baseline.pcd: `ca web docs/demo/perception/candidates/deep_baseline.pcd docs/demo/perception/reference_scene.pcd --heatmap`
  - Snapshot: `ca heatmap3d docs/demo/perception/candidates/deep_baseline.pcd docs/demo/perception/reference_scene.pcd -o deep_baseline_vs_reference_scene_heatmap.png`
- candidates/nondeep_baseline.pcd: `ca web docs/demo/perception/candidates/nondeep_baseline.pcd docs/demo/perception/reference_scene.pcd --heatmap`
  - Snapshot: `ca heatmap3d docs/demo/perception/candidates/nondeep_baseline.pcd docs/demo/perception/reference_scene.pcd -o nondeep_baseline_vs_reference_scene_heatmap.png`