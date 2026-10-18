# Exact target-count capacity comparison

The [study](../../../docs/slam-flatwall-count-reserve.md) reports every condition,
regression and limitation. Both summaries are recomputed from the lossless native
outputs. Build/execution receipts, raw controller bytes and archived sources
remain bound by the manifest. This is reused real development data.

With NumPy and SciPy installed, run:

```sh
python benchmarks/slam/flatwall-count-reserve/verify.py
```

This offline command checks saved original real arrays and native outputs. It
does not rerun or approximate the native timing experiments. Existing archived
native sources are reused only after checking all hashes against the new build
inventories. The new candidate source is archived here for future development.

Source data: Kenji Koide / AIST, [Real-world datasets for LiDAR-based SLAM](https://doi.org/10.5281/zenodo.7641866), version 2, CC BY 4.0. Original provenance,
array hashes and attribution remain in preceding packets. No raw bags included.
