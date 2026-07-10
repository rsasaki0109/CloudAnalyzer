# `ca uncertainty-evaluate`

Evaluate whether reported position covariance is statistically consistent with
ground-truth trajectory error:

```bash
ca uncertainty-evaluate estimated-covariance.json reference.tum --align-mode rigid
```

The JSON input has explicit conventions:

```json
{
  "metadata": {
    "covariance_frame": "estimate_world",
    "error_convention": "estimated_minus_reference"
  },
  "states": [
    {"timestamp": 0.0, "position": [0, 0, 0], "covariance": [[0.01,0,0],[0,0.01,0],[0,0,0.02]]}
  ]
}
```

This phase supports position state and 3x3 covariance (DoF=3). With rigid
alignment, both errors and covariance are rotated into the reference frame.
Covariance must be finite, symmetric, positive definite, and acceptably
conditioned; singular matrices are rejected rather than pseudo-inverted.

Outputs include **position NEES**, normalized position NEES (`NEES / 3`),
χ² confidence coverage, and support count. NIS is intentionally unsupported:
it requires filter innovations and their
innovation covariance, which cannot be reconstructed from poses and state
covariance.

With `align-mode: none`, per-state χ² coverage is descriptive and assumes the
reference trajectory is exact; temporal correlation means samples are not an
independent χ² sequence. Origin/rigid alignment is fitted from the evaluated
data, so results are labeled `aligned_proxy`, not a formal consistency test.
For rigid alignment, `P_ref = R P_est Rᵀ` and the translation error is evaluated
in that same reference frame.

```yaml
checks:
  - id: estimator-consistency
    kind: uncertainty
    estimated: estimated-covariance.json
    reference: reference.tum
    alignment: rigid
    gate:
      max_mean_position_nees: 5.0
      min_normalized_mean_position_nees: 0.2
      min_coverage_95: 0.9
```

These thresholds are illustrative, not universal. Expected mean position NEES
depends on DoF, sample count, temporal correlation, alignment, and GT
uncertainty. Calibrate CI policy for the dataset and desired false-alarm rate;
the lower bound can detect an excessively conservative covariance estimator.
