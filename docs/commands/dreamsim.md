# DreamSim perceptual distance

Install the optional learned-perceptual dependencies:

```bash
pip install "cloudanalyzer[perceptual]"
ca image-evaluate renders/ references/ --metrics psnr,ssim,dreamsim_distance
```

DreamSim is a holistic perceptual distance trained to align with human visual
similarity judgments. Lower values mean more similar images. CloudAnalyzer
uses the official API: `dreamsim(pretrained=True, device=...)`, its returned
PIL preprocessing transform, and `model(image_a, image_b)`. The first use may
download pretrained model weights. See the official
[paper](https://arxiv.org/abs/2306.09344) and
[repository](https://github.com/ssundaram21/dreamsim).

```yaml
checks:
  - id: perceptual-render
    kind: image
    rendered_dir: renders
    reference_dir: references
    metrics: [psnr, ssim, dreamsim_distance]
    gate:
      max_dreamsim_distance: 0.25
```

The threshold is illustrative. Calibrate it from accepted baseline renders
using the same DreamSim package/checkpoint and preprocessing environment.
