# ca pipeline

Run a complete processing and evaluation pipeline in one command: filter → downsample → evaluate.

## Usage

```bash
ca pipeline INPUT REFERENCE [OPTIONS]
```

## Arguments

| Argument | Description |
|---|---|
| `INPUT` | Input point cloud to process |
| `REFERENCE` | Reference point cloud for evaluation |

## Options

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | *required* | Output file path |
| `-v`, `--voxel-size` | `0.1` | Voxel size for downsampling |
| `-n`, `--neighbors` | `20` | Neighbors for outlier filter |
| `-s`, `--std-ratio` | `2.0` | Std deviation ratio threshold |
| `-t`, `--thresholds` | `0.05,0.1,...,1.0` | Comma-separated thresholds for evaluation |
| `--output-json` | — | Dump result as JSON |

## Steps

1. **Filter**: Remove statistical outliers (SOR)
2. **Downsample**: Voxel grid downsampling
3. **Evaluate**: F1/Chamfer/Hausdorff against reference

## Examples

```bash
# Default parameters
ca pipeline noisy.pcd reference.pcd -o clean.pcd

# Custom parameters
ca pipeline noisy.pcd reference.pcd -o clean.pcd \
  -v 0.2 -n 30 -s 1.5 -t 0.05,0.1,0.5

# Save result for CI
ca pipeline input.pcd ref.pcd -o out.pcd --output-json result.json
```

## Output Example

```
Filter:     1784475 -> 1742057 pts (removed 42418)
Downsample: 1742057 -> 1550910 pts (11.0%)
Chamfer:    0.0557
AUC (F1):   0.9819
Saved:      clean.pcd
```
