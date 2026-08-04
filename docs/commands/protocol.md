# Versioned evaluation protocols

CloudAnalyzer can attach a validated, versioned comparison contract to an
evaluation result. This keeps coordinate conventions, alignment, thresholds,
dataset provenance, input hashes, and runtime information next to the metrics.

## Validate a protocol

```bash
ca protocol validate protocol.yaml
ca protocol validate protocol.yaml --format-json
```

The current schema is `cloudanalyzer.protocol.v1`. A minimal protocol is:

```yaml
schema_version: cloudanalyzer.protocol.v1
name: tum-rgbd-ate-rpe
kind: trajectory

conventions:
  coordinate_frame: world
  length_unit: m
  time_unit: s
  alignment: none
  association: timestamp_interpolation
  quaternion_order: xyzw

parameters:
  max_time_delta_s: 0.05
  rpe_distance_m: [1.0, 10.0]

provenance:
  dataset: TUM RGB-D
  dataset_version: rgbd_dataset_freiburg1
  hash_inputs: true
```

`name`, `kind`, and the five core convention fields are required. Unknown
fields are preserved for forward-compatible metadata. The CLI computes a
stable `protocol_sha256`/`protocol_id` from the declared document; run-specific
details live under `evaluation_protocol.run` and do not change that identity.

## Attach it to an evaluation

The main evaluation commands accept the same option:

```bash
ca evaluate estimated.pcd reference.pcd \
  --protocol protocol.yaml --output-json qa/evaluate.json

ca traj-evaluate estimated.tum reference.tum \
  --protocol protocol.yaml --output-json qa/trajectory.json

ca map-evaluate estimated.pcd reference.pcd \
  --protocol protocol.yaml --output-json qa/map.json
```

The same option is available on `image-evaluate`, `rendered-evaluate`,
`geometry-evaluate`, `run-evaluate`, and `benchmark eval`. The result gains an additive
`evaluation_protocol` object containing:

- the normalized protocol and stable identity;
- the exact command and effective options;
- resolved input paths, sizes, and SHA-256 hashes (or a directory manifest);
- CloudAnalyzer, Python, and platform versions.

Input hashing is enabled by default for protocol runs. Set
`provenance.hash_inputs: false` when a large remote dataset is already
content-addressed elsewhere; the manifest will still record paths and sizes.
