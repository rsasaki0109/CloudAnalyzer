# Public Object Evaluation Examples

This directory stores checked-in JSON examples for `ca detection-evaluate` and
`ca tracking-evaluate`.

## What These Files Are

- `detection_reference.json`
  Axis-aligned boxes derived from semantic label clusters on the public
  RELLIS-3D frame `000001`.
- `detection_estimated_good.json`
  A near-match detection estimate that should score well against the same
  reference.
- `detection_estimated_regressed.json`
  A deliberately degraded detection estimate with misses, localization error,
  and a false positive.
- `tracking_reference.json`
  A deterministic 3-frame synthetic track sequence seeded from the same public
  frame boxes.
- `tracking_estimated_good.json`
  A near-match tracking estimate with stable IDs.
- `tracking_estimated_regressed.json`
  A degraded tracking estimate with a miss and an ID switch.

The detection files are derived from public data. The tracking files are
**synthetic temporal examples** built from those same public boxes so the
tracking JSON contract can be demonstrated without bundling a full public MOT
dataset in-repo.

## Regenerate

```bash
python3 scripts/build_object_eval_examples.py \
  --output demo_assets/public/rellis3d-frame-000001/object_eval \
  --example-root demo_assets/public/rellis3d-frame-000001 \
  --frame 000001
```

## Example Commands

```bash
ca detection-evaluate \
  demo_assets/public/rellis3d-frame-000001/object_eval/detection_estimated_good.json \
  demo_assets/public/rellis3d-frame-000001/object_eval/detection_reference.json \
  --iou-thresholds 0.25,0.5 --min-map 0.9

ca tracking-evaluate \
  demo_assets/public/rellis3d-frame-000001/object_eval/tracking_estimated_regressed.json \
  demo_assets/public/rellis3d-frame-000001/object_eval/tracking_reference.json \
  --iou-threshold 0.5 --min-mota 0.8 --max-id-switches 1
```
