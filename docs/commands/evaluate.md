# ca evaluate

Evaluate point cloud similarity using F1, Chamfer Distance, Hausdorff Distance, and AUC metrics.

## Usage

```bash
ca evaluate SOURCE TARGET [OPTIONS]
```

## Arguments

| Argument | Description |
|---|---|
| `SOURCE` | Source (estimated) point cloud file |
| `TARGET` | Target (reference) point cloud file |

## Options

| Option | Default | Description |
|---|---|---|
| `-t`, `--thresholds` | `0.05,0.1,0.2,0.3,0.5,1.0` | Comma-separated distance thresholds |
| `--plot` | — | Output path for F1/P/R curve plot (png) |
| `--output-json` | — | Dump result as JSON file |
| `--format-json` | `false` | Print JSON to stdout |

## Metrics

| Metric | Description |
|---|---|
| **Precision** | Fraction of source points within threshold distance of target |
| **Recall** | Fraction of target points within threshold distance of source |
| **F1 Score** | Harmonic mean of Precision and Recall: `2*P*R/(P+R)` |
| **Chamfer Distance** | Mean of bidirectional mean nearest-neighbor distances |
| **Hausdorff Distance** | Max of bidirectional max nearest-neighbor distances |
| **AUC** | Area Under the F1 curve (trapezoidal integration, normalized) |

## Examples

```bash
# Basic evaluation
ca evaluate estimated.pcd reference.pcd

# Custom thresholds with plot
ca evaluate estimated.pcd reference.pcd \
  -t 0.01,0.05,0.1,0.2,0.5,1.0 --plot f1_curve.png

# JSON output for automation
ca evaluate a.pcd b.pcd --format-json | jq '.auc'

# Evaluate downsampling quality
ca downsample map.pcd -o map_v02.pcd -v 0.2
ca evaluate map_v02.pcd map.pcd --plot quality.png
```

## Output Example

```
Source: 1597449 pts | Target: 1784475 pts

Chamfer Distance:  0.0083
Hausdorff Distance: 0.1809
AUC (F1):          0.9852

F1 Scores:
  d=0.05  P=0.9471  R=0.8833  F1=0.9141
  d=0.10  P=0.9971  R=0.9899  F1=0.9935
  d=0.20  P=1.0000  R=1.0000  F1=1.0000

S->T  mean=0.0053  median=0.0000  max=0.1473
T->S  mean=0.0113  median=0.0063  max=0.1809
```

## Python API

```python
from ca.evaluate import evaluate, plot_f1_curve

result = evaluate("estimated.pcd", "reference.pcd", thresholds=[0.05, 0.1, 0.5])
print(f"AUC: {result['auc']:.4f}")
print(f"Chamfer: {result['chamfer_distance']:.4f}")

# Plot F1 curve
plot_f1_curve(result, "f1_curve.png")

# Compare multiple
from ca.plot import plot_multi_f1
results = [evaluate(f"v{v}.pcd", "ref.pcd") for v in [0.1, 0.2, 0.5]]
plot_multi_f1(results, ["v0.1", "v0.2", "v0.5"], "comparison.png")
```
