# Packed ordinary GPU C2C nodes

The experimental ordinary WebGPU search can store each node in two u32 words (8 bytes) instead of five (20 bytes). The first word holds the two-bit axis/leaf tag and a 30-bit right-child or leaf-start index. The second preserves every split-coordinate bit or the leaf end. Left children remain implicit. Point coordinates, traversal order, visit limits, 32 stack slots and authoritative f64 CPU verification are unchanged. Leaf, box and parent entry points use the original nodes; packed ordinary traversal takes precedence over the separate compact-stack control.

A new GPU reference uploads only its selected node representation. A later format switch lazily adds the other representation while sharing the point, query and readback buffers. Packing uses a temporary one-byte-per-node connectivity array and rejects malformed trees before upload. Failed allocation/upload preparation cleans up and permits retry. Selected node bytes and total cached node bytes are reported separately. The 60% node-buffer reduction does not measure total device memory or memory traffic; a reference that has used both formats retains 28 bytes per node.

The [raw packet](../benchmarks/wasm-gpu-packed-nodes/manifest.json) fixes four reports in PandaSet, Autoware, Autoware, PandaSet order, four exact CPU workers, 131072-query batches, six warm-up pairs and twelve alternating production RPC pairs. Diagnostics, inherited bounds, boxes, parent traversal, compact stacks and alternative query copies are off in the primary comparison. Each report also retains fresh-reference cold samples and two separate diagnostic pairs after two diagnostic warm-up pairs. Parsing/indexing are outside the RPC; cold RPCs include reference export, packing, upload and exact verification with the device, pipelines and pool warm. Input specifications retain source hashes and attribution and reproduce the same even/odd real-point subsets without added noise.

| Real input | Flat median ms | Packed median ms | Median paired change | Faster pairs | Cold flat / packed ms |
|---|---:|---:|---:|---:|---:|
| pandaset 1 | 126.60 | 127.80 | -0.99% | 7/12 | 308.8 / 595.2 |
| autoware 1 | 173.35 | 174.50 | -1.42% | 7/12 | 454.5 / 432.3 |
| autoware 2 | 168.25 | 169.90 | -1.54% | 7/12 | 832.6 / 905.8 |
| pandaset 2 | 118.65 | 114.80 | -0.22% | 7/12 | 282.0 / 682.0 |

Paired changes are medians of within-pair ratios; they differ from the ratio of the separately computed medians. Every tail is retained, including the 677.7 ms flat Autoware trial. The candidate won 28 of 48 pairs. Both Autoware reports passed the warmed gate, while neither PandaSet report reached the required 1% paired reduction. Only one report passed every gate. Three single-sample cold guards failed; these samples do not establish that packing caused the delay.

Separate diagnostic GPU pass totals fell from roughly 66.1 to 60.4–60.5 ms for PandaSet and 91.5–92.0 to 83.3–83.4 ms for Autoware (about 8–9%). These instrumented results explain the candidate's potential and are excluded from adoption. Host/GPU phases overlap and use distinct clocks; subtracting them cannot isolate transfer cost. CPU certificate totals remain identical and are not candidate-accuracy measurements.

Fresh benchmark references retain 646200 vs 1615500 node bytes for PandaSet and 1048568 vs 2621420 for Autoware. Warm mixed-format references retain 2261700 and 3669988 node bytes respectively, with shared points. Ordinary runs request no optional box/parent buffers. The unchanged Rust source set and four byte-identical WASM artifacts are verified before frontend reuse; CI also performs a fresh full WASM build.

Adoption required all four reports to achieve at least 1% median paired RPC reduction and 7/12 faster pairs, every cold packed/flat ratio at most 1.05, and the predeclared buffer guards. The overall gate failed, so flat nodes remain the default. `gpuPackedNodes` is an internal experimental control; backend selection is unchanged.

Correctness checks passed 26 software-GPU browser cases and four physical-GPU cases. Thirty full f64/brute comparisons cover 258 real queries, an exact hit and two overflow guards. Eight ordinary workgroup/inherited variants directly compare 2088 raw candidate slots, including 2072 valid non-overflow slots, before CPU correction. Tests also cover exact needed-field packing, 40 codec edge/malformed cases, independent uploads, lazy shared preparation, format switches, mutation during concurrent buffered maps, ignored flags, allocation/upload failure cleanup, pipeline fallback/retry and zero live buffers after disposal. Slot parity is scoped to this fixture; f64 verification remains authoritative for every result.

A sample compatibility run uses a new output path:

```sh
node web/scripts/build-query-export.mjs --output packed-build.json
node web/scripts/benchmark-packed-nodes.mjs --output packed-smoke.json --build-info packed-build.json --reference web/public/samples/lidar_reference.pcd --compared web/public/samples/lidar_candidate.pcd --threads 2 --warmups 0 --repeats 1 --gpu software --cold-first flat
```

The sample run checks compatibility and does not replace the frozen real-data timing protocol. No adapter identity, physical occupancy or spill data is collected.
