# RELLIS-3D Public Seed Frame

This directory stores the minimal public input needed to regenerate the checked-in
perception demo under `docs/demo/perception` without downloading from Google Drive.

Included files:

- `os1_cloud_node_kitti_bin/000001.bin`
- `os1_cloud_node_semantickitti_label_id/000001.label`

These files come from the public RELLIS-3D "Ouster LiDAR with Annotation Examples"
bundle and are used only as a reproducibility seed for the static perception demo.

Regenerate the demo with:

```bash
python3 scripts/build_perception_demo.py \
  --output docs/demo/perception \
  --example-root demo_assets/public/rellis3d-frame-000001 \
  --frame 000001
```
