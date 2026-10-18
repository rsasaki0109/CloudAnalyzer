# Target-balanced odometry trial

An opt-in `LidarOdometry(target_balance=True)` path now divides each accepted pair's robust weight by the number of pairs selecting the same map coordinate, recomputed at every ICP iteration. It keeps all accepted pairs and parallelizes search and H/b accumulation on the CPU. Default behavior remains off.

On the frozen two-recording trial, conditional rigid-aligned position RMSE improves in 11 of 16 conditions and worsens in 5. Fine-sampling + gyro compensation improves by about 31–64% across both prediction policies and both recordings. Raw/fine repeat-motion cases worsen, and a sparse case approximately doubles translation RPE despite a tiny position improvement. The complete mixed result is retained, with no default adoption or runtime/GLIM claim.

All 2,560 new frames closed before scoring, alongside 2,560 same-build default replays that reproduce previous pose and diagnostic-array bits. The [complete packet](../benchmarks/slam/flatwall-target-balance/README.md) contains all case metrics, both author reference members, trajectories, frame systems, 48 predetermined raw map/match samples and an independent verifier. It documents exact associations, physical provenance assumptions, public/private replay limits, native source identities and preserved failed attempts.

Run `python benchmarks/slam/flatwall-target-balance/verify.py`. To run a retained-input trial, use `scripts/run_flatwall_target_balance.py`; its entry normalizes paths before invoking the frozen controller. A full trial requires the original recordings and retained baseline snapshots; the public verifier requires neither the bags nor a native extension.
