# Cloudini Benchmark Tutorial

This workflow treats Cloudini as the compression engine and CloudAnalyzer as the QA / benchmarking layer.

## Goal

Measure both:

- geometry quality after compression / decompression
- compressed artifact size

in one repeatable report.

## Expected directory layout

```text
experiment/
├── original/      # original point clouds before compression
├── compressed/    # Cloudini artifacts (.cloudini, .bin, etc.)
└── decoded/       # point clouds decoded back from Cloudini
```

Files are matched by relative path first, then by stem.

Example:

```text
original/map_a.pcd
compressed/map_a.cloudini
decoded/map_a.pcd
```

## Run the benchmark

```bash
ca batch experiment/decoded --evaluate reference_map.pcd \
  --compressed-dir experiment/compressed \
  --baseline-dir experiment/original \
  --report experiment/cloudini_report.html
```

This produces:

- quality metrics: `Chamfer`, `Hausdorff`, `AUC`, `Best F1`
- size metrics: `Size Ratio`, `Space Saving`
- `Quality vs Size` scatter plot with Pareto frontier
- Pareto candidate list in the report summary
- Recommended operating point based on the smallest acceptable Pareto candidate
- HTML report count-badged summary rows, quick actions, failed-first / recommended-first sorting, and filters for pass-only, failed-only, pareto-only, and recommended-only review
- inspection commands for `ca web --heatmap` and `ca heatmap3d`

## Add a quality gate

```bash
ca batch experiment/decoded --evaluate reference_map.pcd \
  --compressed-dir experiment/compressed \
  --baseline-dir experiment/original \
  --min-auc 0.95 \
  --max-chamfer 0.02 \
  --report experiment/cloudini_report.html
```

If any file fails the thresholds, the command exits with code `1`.

## Interpretation

- low `Size Ratio` is better: smaller compressed artifacts
- high `AUC` is better: less quality loss
- low `Chamfer` is better: less geometric drift
- Pareto candidates are the non-dominated operating points: no other result is both smaller and higher-quality
- the recommended point is the smallest Pareto candidate that still satisfies the quality gate, if one is active

The useful operating point is the best tradeoff between size and quality, not the best value on either axis alone.
