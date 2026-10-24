# SalsaNext architecture

`salsanext.py` is adapted from [TiagoCortinhal/SalsaNext](https://github.com/TiagoCortinhal/SalsaNext/blob/7548c124b48f0259cdc40e98dfc3aeeadca6070c/train/tasks/semantic/modules/SalsaNext.py),
commit `7548c124b48f0259cdc40e98dfc3aeeadca6070c`, under the accompanying MIT
license. Only two unused imports (`imp`, `__init__ as booger`) were removed for
modern Python compatibility; the network and forward pass are unchanged.
Upstream file SHA256: `7af43381eafa11f3124a8403cdba9880bf4026545518b5615a58328a0475b378`.

Tiago Cortinhal, George Tzelepis and Eren Erdal Aksoy,
*SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving* (2020).
[Paper](https://arxiv.org/abs/2003.03653).

Pretrained weights and example scans are not bundled. Their upstream model and
dataset terms apply independently; the source record accompanies local preparation.
