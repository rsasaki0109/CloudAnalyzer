# Real target-normal SLAM diagnostic

A fixed target-normal objective has been evaluated on 2,560 preserved frames from both real flatwall recordings, across two prediction policies and four input/sampling arms. It uses parallel CPU neighborhood queries and batched PCA, without changing the running estimator.

Sparse legacy cases frequently show weak plane systems. Fine-sampling cases show no weak directions at the frozen 1e-3 diagnostic threshold on either objective, despite previously observed translation drift. This result does not support adopting a plane objective as a complete drift fix. Normal acceptance retains a median of roughly 35–54% of matches in the fine cases; the comparator uses that same subset, weights and center.

The [complete packet](../benchmarks/slam/flatwall-plane-objective/README.md) provides all 16 case results, all frame matrices/spectra, explicit normal rejections, 48 predetermined full-map samples, independent arithmetic verification and semantic tamper tests. It documents the exact perturbation convention, PCA policy, sample coverage and limits of public versus private replay. Original estimates, native artifacts and previous trajectory scores remain unchanged.

Run `python benchmarks/slam/flatwall-plane-objective/verify.py` from the repository root. No state feedback, accuracy improvement, calibrated observability, timing/GPU speedup or GLIM equivalence is claimed. Future updates should freeze their solver/correspondence policy before evaluating both complete recordings.
