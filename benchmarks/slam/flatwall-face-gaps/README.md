# Real voxel-face bound reuse comparison

The [study](../../../docs/slam-flatwall-face-gaps.md) reports all conditions and
limitations. `time-summary.json` and `fixed-summary.json` are recomputed from
four complete trajectory replays and eight fixed-state processes. Lossless gzip
outputs retain every original native JSON byte; invocations, clocks, logs,
receipts and executed source archives remain bound by the manifest.

Run from a checkout with NumPy and SciPy installed:

```sh
python benchmarks/slam/flatwall-face-gaps/verify.py
```

This offline command checks saved original real arrays and native outputs;
it does not rerun or approximate the native timings. Archived `.py.raw` files
preserve the exact executed controller/analysis bytes. Fixed-state analysis is
archived separately as `fixed-analysis-controller.py.raw`. Existing native
source archives are reused only after
checking their hashes against each new build inventory. The candidate source is
archived here so later native development cannot rewrite this measurement.

Source data: Kenji Koide / AIST, [Real-world datasets for LiDAR-based SLAM](https://doi.org/10.5281/zenodo.7641866), version 2, CC BY 4.0. The original
flat-wall provenance, array hashes and attribution remain in the earlier
packets. Recordings are reused development data. No raw bags are included.
