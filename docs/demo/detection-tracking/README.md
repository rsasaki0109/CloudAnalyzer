# Detection & Tracking Demo

Public demo comparing good and regressed candidates for 3D object detection
and multi-object tracking evaluation.

## Contents

- `index.html` — Overview page with comparison tables
- `detection_good.html` / `detection_regressed.html` — Detection reports
- `tracking_good.html` / `tracking_regressed.html` — Tracking reports
- `results.json` — Raw evaluation results

## Rebuild

```bash
python scripts/build_detection_tracking_demo.py --output docs/demo/detection-tracking
```

## Data Source

All examples are derived from the public RELLIS-3D dataset.
See ATTRIBUTION.md for license details.
