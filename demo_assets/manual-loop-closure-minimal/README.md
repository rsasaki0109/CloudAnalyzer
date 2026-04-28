# Manual Loop-Closure Minimal Fixture

Synthetic fixture for dogfooding `ca loop-closure-report` without bundling external SLAM data.

The fixture contains:

- `before/map.pcd`: cube corners shifted by `+0.25 m` on x
- `after/map.pcd`: cube corners aligned to the reference
- `reference/map.pcd`: reference cube corners
- `before/pose_graph.g2o` and `after/pose_graph.g2o`: tiny connected pose graphs
- `before/optimized_poses_tum.txt`, `after/optimized_poses_tum.txt`, and `reference/trajectory.tum`: three-pose trajectories

Example:

```bash
ca loop-closure-report \
  demo_assets/manual-loop-closure-minimal/before/map.pcd \
  demo_assets/manual-loop-closure-minimal/after/map.pcd \
  demo_assets/manual-loop-closure-minimal/reference/map.pcd \
  --before-session-root demo_assets/manual-loop-closure-minimal/before \
  --after-session-root demo_assets/manual-loop-closure-minimal/after \
  --before-traj demo_assets/manual-loop-closure-minimal/before/optimized_poses_tum.txt \
  --after-traj demo_assets/manual-loop-closure-minimal/after/optimized_poses_tum.txt \
  --ref-traj demo_assets/manual-loop-closure-minimal/reference/trajectory.tum \
  --thresholds 0.05,0.1,0.2,0.5 \
  --min-auc-gain 0.01 \
  --min-ate-gain 0.05 \
  --require-posegraph-ok \
  --format-json
```
