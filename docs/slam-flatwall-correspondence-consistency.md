# Real SLAM correspondence consistency

All 1,062,382 preserved forward matches across 2,560 frames and 16 real recording/policy/input cases now have independently checked reverse nearest-source diagnostics. Queries cover every original source point, including unmatched points, and resolve ties by original index. Exact target deduplication and four CPU workers reduce repeated reverse-query work without changing query results.

Fine-sampling cases save a median 57.9–60.8% of reverse query points through deduplication. Strict reciprocal filtering retains a median 27.6–30.4% of pairs and 28.2–31.4% of robust weight. These are density and support observations, not evidence that rejected pairs are outliers or that filtering improves drift. No estimator update or default changes.

The [complete packet](../benchmarks/slam/flatwall-correspondence-consistency/README.md) includes all 16 summaries, all original/subset frame systems, the full private snapshot inventory and 48 predetermined public source/match samples. Its independent public verifier exhaustively checks every reverse query and H/b on the samples, plus all frame spectra and accounting. It states the narrower scope of public raw replay versus the full private audit.

Run `python benchmarks/slam/flatwall-correspondence-consistency/verify.py`. The next candidate is target-multiplicity weight balancing with all pairs retained; it requires a separate frozen closed-loop trial and inherited reference scoring. No accuracy, calibrated observability, GLIM equivalence or end-to-end speedup is claimed here.
