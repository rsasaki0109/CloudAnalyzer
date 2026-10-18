# Fixed-state odometry cost and worker scaling

The fixed real-state experiment separates target-balanced H/b construction into
search/collection, exact target counts, matrix accumulation and destruction.
With four local Rayon workers, the eligible suite's staged time is approximately
63% search, 24% count and 12% accumulation. Whole balanced construction takes
1.55 times fused baseline on identical inputs. With one worker the ratio is 1.25.

On the 24 fine final states, moving from one to four workers reduces balanced
construction time by a factor of 1.50, and baseline by 1.91. The 20 eligible
legacy states become slower. Serial count allocation/lookups and small-input
parallel scheduling are therefore useful next targets. No production arithmetic,
stopping policy, weighting rule, map update or default changes here.

All 48 previously selected final snapshots remain, along with the first/last
attempted step of every capped frame. Four insufficient-points snapshots retain
their original empty registration source and are reported separately. A failed
untimed preparation check and its original inputs/source remain preserved.

These are warm fixed-state native kernel measurements, not whole-registration,
application, WASM, GPU or general throughput results. The staged decomposition
has its own clocks/code layout; its fractions do not predict removable time from
the whole call. The [complete packet](../benchmarks/slam/flatwall-balance-stages/README.md)
retains all samples, independent checks, frozen protocols and executed source.
