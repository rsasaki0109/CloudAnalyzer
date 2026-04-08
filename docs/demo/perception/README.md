# Perception Artifact Comparison Demo

This demo compares two candidate perception artifacts against the same public reference frame.
It uses the RELLIS-3D Ouster LiDAR example and evaluates both candidates with `ca batch`.

## What The Candidates Mean

- `nondeep_baseline.pcd`: geometry-first non-deep baseline with coarse voxelization, long-range thinning, and a larger rigid bias.
- `deep_baseline.pcd`: higher-fidelity deep baseline with denser sampling and a smaller rigid bias.
- `reference_scene.pcd`: non-void labeled points from the official public RELLIS-3D example frame.

These are deterministic demo artifacts, not archived model outputs. The point is to show how
CloudAnalyzer compares a non-deep baseline and a deep baseline against the same public reference artifact.

## Source Data

- Dataset: [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)
- Example bundle: [Ouster LiDAR with Annotation Examples](https://drive.google.com/uc?export=download&id=1QikPnpmxneyCuwefr6m50fBOSB2ny4LC)
- Label ontology: [rellis.yaml](https://github.com/unmannedlab/RELLIS-3D/blob/main/benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml)
- Frame: `000001`

## Label Counts In The Reference Frame

- `3` grass: 18,569 points
- `4` tree: 18,351 points
- `17` person: 520 points
- `18` fence: 522 points
- `19` bush: 2,532 points
- `23` concrete: 19,919 points
- `31` puddle: 1,286 points
- `33` mud: 82 points

## Batch Metrics

| Candidate | Points | Retained vs Ref | AUC | Chamfer | Best F1 | Gate |
|---|---:|---:|---:|---:|---:|---|
| `deep_baseline.pcd` | 34370 | 55.6% | 0.9515 | 0.0252 | 1.0000 | PASS |
| `nondeep_baseline.pcd` | 16906 | 27.4% | 0.6648 | 0.1094 | 0.9735 | FAIL |

Gate settings: `min_auc=0.90`, `max_chamfer=0.05`

Interpretation:
The non-deep baseline intentionally drops long-range geometry and fails the gate.
The deep baseline keeps denser geometry and passes the same gate on the same frame.

## Files

| File | Description |
|---|---|
| `reference_scene.pcd` | Public reference artifact derived from the official labels |
| `candidates/nondeep_baseline.pcd` | Geometry-first non-deep baseline artifact |
| `candidates/deep_baseline.pcd` | Higher-fidelity deep baseline artifact |
| `index.html` | HTML batch report for GitHub Pages |
| `report.md` | Markdown version of the same batch report |
| `results.json` | Raw batch results, summary, and demo metadata |
| `ATTRIBUTION.md` | Dataset provenance and license note |

## Reproduce

```bash
python3 scripts/build_perception_demo.py --output docs/demo/perception --frame 000001

ca batch docs/demo/perception/candidates \
  --evaluate docs/demo/perception/reference_scene.pcd \
  --thresholds 0.02,0.05,0.1,0.2,0.3 \
  --min-auc 0.90 --max-chamfer 0.05 \
  --report docs/demo/perception/index.html
```
