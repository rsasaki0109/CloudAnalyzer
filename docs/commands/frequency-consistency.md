# Frequency Consistency Metric (FCM)

CloudAnalyzer exposes the Frequency Consistency Metric as
`frequency_consistency` in both `ca image-evaluate` and
`ca rendered-evaluate`. It targets high-frequency 3D Gaussian Splatting
artifacts that can be understated by PSNR or SSIM. Lower is better.
The implementation follows Equation 3 of Zhou et al.,
[*Reducing Closeup Frequency Artifacts for Level-of-Detail 3D Gaussian
Splatting*](https://openaccess.thecvf.com/content/CVPR2026W/3DMV/html/Zhou_Reducing_Closeup_Frequency_Artifacts_for_Level-of-Detail_3D_Gaussian_Splatting_CVPRW_2026_paper.html).

```bash
ca image-evaluate renders/ references/ \
  --metrics frequency_consistency --format-json
```

## Definition

For each same-named candidate/reference pair, CloudAnalyzer:

1. clamps decoded RGB values to `[0, 1]` and converts them to grayscale with
   `0.299 R + 0.587 G + 0.114 B`;
2. applies a sampled `5 x 5` Laplacian-of-Gaussian filter with `sigma = 1`
   and zero padding;
3. computes the normalized Frobenius error
   `||LoG(candidate) - LoG(reference)||_F / ||LoG(reference)||_F`;
4. clamps the score to `[0, 1]`.

Identical edge maps score `0`. If the reference edge map is flat, the normal
denominator is zero: CloudAnalyzer defines both maps flat as `0`, and a flat
reference with a non-flat candidate as `1`. The JSON metadata records the
kernel, padding, normalization, direction, and this flat-reference policy so
results remain auditable.

FCM is reference-based and frame-local. It does not claim to measure temporal
flicker, optical-flow consistency, opacity, or false transparency.

## CI gate

The aggregate is the arithmetic mean of finite per-pair scores. Enable the
metric explicitly or let the gate enable it automatically:

```yaml
checks:
  - id: splat-frequency
    kind: rendered
    splat: outputs/scene.ply
    cameras: outputs/transforms.json
    reference_dir: baselines/renders
    metrics: [psnr, ssim, frequency_consistency]
    gate:
      max_frequency_consistency: 0.12
```

`max_frequency_consistency` is available for both `kind: image` and
`kind: rendered`. Calibrate the threshold on an accepted baseline; a gate with
zero matched pairs or an unavailable aggregate fails rather than passing
silently.
