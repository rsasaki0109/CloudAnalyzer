# `ca image-evaluate`

Score a set of rendered images against a reference (ground-truth) set on
**PSNR + SSIM** — the first photometric evaluation surface in
CloudAnalyzer. Pairs images by filename across the two directories and
emits per-pair metrics plus a per-metric summary (mean / median / min /
max).

## Why it exists

CloudAnalyzer's existing evaluation surface is *geometric*:

- `ca evaluate` / `ca map-evaluate` — Chamfer / F1 / AUC of point clouds
- `ca geometry-evaluate` — 3DGS / mesh → point cloud → same geometric metrics
- `ca traj-evaluate` — ATE / RPE of trajectories

`ca image-evaluate` opens a parallel **photometric** axis. For 3DGS /
NeRF outputs the rendered image at known camera poses is often the
real QA target, not the underlying geometry. This command is the
foundation; a future `ca rendered-evaluate` will fold rendering and
scoring into one command (driving a 3DGS PLY into images at given
camera poses, then piping through this scoring layer).

## Usage

```bash
ca image-evaluate <rendered_dir> <reference_dir> [options]
```

`<rendered_dir>` and `<reference_dir>` are directories of images that
share filenames. Pairs are matched by basename + extension; files
present in `<rendered_dir>` but missing from `<reference_dir>` are
counted in `summary.pairs_missing_in_reference` and skipped. Pairs
whose shapes don't match are counted in `summary.pairs_size_mismatch`
and skipped (drop the renderer's resize step if you are seeing these).

Supported formats: PNG, JPEG. Alpha channels are dropped; grayscale
inputs are promoted to RGB by replicating the single channel.

## Common options

| Option | Description |
|---|---|
| `--metrics psnr,ssim` | Comma-separated metric list. Defaults to both. |
| `--extensions .png,.jpg,.jpeg` | Image extensions discovered under `<rendered_dir>`. |
| `--ssim-window 11` | Gaussian-window side length used in SSIM (Wang & Bovik 2004 use 11). |
| `--ssim-sigma 1.5` | Gaussian-window sigma used in SSIM. |
| `--max-pairs N` | Cap on number of pairs evaluated. Useful when smoke-testing a large render set. |
| `--output-json path.json` | Write the full result (per-pair + summary + metadata) as JSON. |
| `--format-json` | Print the same JSON on stdout instead of the human-readable summary. |

## Metric definitions

- **PSNR** — Peak signal-to-noise ratio. Identical inputs return `+inf`;
  identical-shape inputs differing by Gaussian noise of standard
  deviation `σ` produce roughly `−20·log10(σ)` dB on unit-range data.
  The aggregate `psnr_*` stats drop `+inf` rows so a single bit-identical
  pair doesn't poison the mean.
- **SSIM** — Structural similarity index over a Gaussian window
  (Wang, Bovik et al. 2004). Identical inputs return exactly `1.0`.
  Color inputs are scored per-channel and averaged. Implementation uses
  `scipy.ndimage.gaussian_filter` (no new dependency).

## Examples

### Score every PNG in two folders

```bash
ca image-evaluate renders/seq00 references/seq00
```

Prints something like:

```
image-evaluate: 200 pair(s) scored (0 missing in reference, 0 size-mismatch)
  PSNR  mean=28.1432 dB median=28.0961 dB min=24.1102 dB max=33.5570 dB
  SSIM  mean=0.8927 median=0.8941 min=0.7521 max=0.9645
```

### Pipe into another tool via JSON

```bash
ca image-evaluate renders/seq00 references/seq00 --format-json \
  | jq '.summary.psnr_mean'
```

### Score only PSNR, capped at 20 pairs (CI smoke)

```bash
ca image-evaluate renders/seq00 references/seq00 \
  --metrics psnr --max-pairs 20 --output-json qa/image_eval.json
```

## CI quality gate

`image-evaluate` is also exposed as a config-driven check kind, so photometric
quality rides the same regression gate as geometric/trajectory checks:

```yaml
checks:
  - id: rendered-views
    kind: image
    rendered_dir: outputs/renders/seq00
    reference_dir: baselines/references/seq00
    gate:
      min_psnr: 28.0   # dB
      min_ssim: 0.85
```

`ca check` gates on the aggregate **mean** PSNR / SSIM, emits a per-check
report, ranks failures in triage, and `ca report-pr-comment` shows the means
with up/down deltas against a baseline. See [../ci.md](../ci.md) § *Config-Driven QA*.

## Future directions

- `ca rendered-evaluate <3dgs.ply> <reference_dir>` — render a 3DGS PLY
  at the camera poses implied by `reference_dir`'s filename metadata
  (or a separate camera JSON), then drive the result through the
  scoring path here.
- LPIPS support behind an optional `[photometric]` extra (requires
  `torch` + a pretrained network — not landed yet); once it lands,
  `max_lpips` joins the `image` check gate.

## Related

- `ca evaluate` / `ca map-evaluate` — geometric scoring of point clouds.
- `ca geometry-evaluate` — 3DGS / mesh sampled to point cloud, scored geometrically.
- `ca traj-evaluate` — trajectory scoring (ATE / RPE).
