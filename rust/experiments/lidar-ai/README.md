# LiDAR AI on wgpu — native experiment

This separate package runs the **pretrained SalsaNext SemanticKITTI CNN** with
Burn/wgpu. It is an opt-in experiment, not an installed `ca` command or a browser
feature. The normal app, workspace build and visibility-based dynamic removal
are unchanged. Parked and moving cars share the model's `car` class.

The first supported input is **one KITTI-format sensor-frame HDL-64 scan**:

- Little-endian float32 `[x,y,z,remission]`, 1..150,000 points; metres, X forward,
  Y left, Z up; remission already in 0..1. No inferred intensity normalization.
- Fixed 64×2048 range image, elevation −25°..+3°, range up to 100 m.
- At a pixel collision, only the nearest source return gets a class. All other,
  invalid, out-of-view and out-of-range returns remain in the retained output.
- Selected class + softmax threshold (default **car, 0.9**) chooses a candidate
  mask. No k-NN spreading, object-box enlargement or geometry heuristics.

**Do not pass an accumulated map as a scan.** A fused map needs retained scan
origins and per-frame inference before fusion, or a separately validated
map-segmentation model. These weights do not establish cross-sensor accuracy.

## Prepare the model locally

Download `pretrained.zip` through the [official SalsaNext pretrained-model link](https://github.com/TiagoCortinhal/SalsaNext#pretrained-model).
Weights and scans are not included in Git. The exporter checks the archive and
checkpoint SHA256 before loading, keeps PyTorch's weights-only unpickler, ignores
unused training state and exports the unchanged inference network to ONNX.

In an environment with PyTorch 2.6+, NumPy 2 and ONNX already installed, from repo root:

```sh
python rust/experiments/lidar-ai/prepare_salsanext.py --archive /path/to/pretrained.zip --out notes/my-lidar-model
```

Use a new output directory. This step does not install packages or download data.
The experiment's [third-party notice](third_party/NOTICE.md) records the pinned
architecture, paper and license. Model/dataset terms apply independently.

## Build and run

Rust 1.95+ is required by Burn 0.22. An example PowerShell session:

```powershell
$env:CA_LIDAR_ONNX = (Resolve-Path notes/my-lidar-model/salsanext.onnx).Path
$env:CARGO_TARGET_DIR = Join-Path $PWD notes/lidar-ai-target
$env:CARGO_BUILD_JOBS = '2'
cargo build --manifest-path rust/experiments/lidar-ai/Cargo.toml --features gpu
```

The build-time ONNX importer writes `salsanext.bpk` next to generated Rust in
`notes/lidar-ai-target/debug/build/ca-lidar-ai-*/out/model/`. Pass that exact local
weight pack explicitly (the binary checks its hash against its imported pack):

```powershell
notes/lidar-ai-target/debug/ca-lidar-ai.exe --scan scan.bin --sensor-frame --weights /path/to/salsanext.bpk --out notes/new-lidar-result
```

A discrete GPU is explicitly requested; a missing/failed adapter is an error.
No CPU/software fallback is labelled as GPU execution. The build checks the
export manifest, pinned checkpoint, model shape and ONNX digest. The first call includes
model loading, shader compilation and readback, so its time is not a warmed
inference benchmark. Model, shape and probability failures select no removal.

Outputs are staged and published to a new directory after validation. A file-I/O
failure may leave a clearly named pending directory; it does not replace an
existing result or input. The result contains:

- `retained.bin`: original records outside the selected mask.
- `candidates.bin`: original records selected as the requested object class.
- `candidate-mask.u8`: one 0/1 byte per original input record, in source order.
- `report.json`: counts, class names, source/weight hashes, profile and limits.

The input stays unchanged. Open the retained and candidate `.bin` files in the
browser to review them. `--classes car,truck,other-vehicle` and `--threshold 0.95`
are explicit alternatives. `--debug-tensors` writes input/output tensors for
numerical comparison; it is off by default.

## Checks

Default tests require no model or GPU and keep inference dependencies disabled:

```sh
cargo test --manifest-path rust/experiments/lidar-ai/Cargo.toml --locked
```

They cover sensor axes/FOV, range/point limits, nearest-only ownership, target and
confidence selection, malformed probability tensors, and byte-exact source
record splitting (including retained NaN payloads). A supported machine can also
run `cargo clippy --all-targets --features gpu -- -D warnings` with the manifest
and model environment above.

A one-frame prediction and CPU/GPU numerical agreement do **not** demonstrate
whole-dataset detection accuracy, whole-car removal, improved map quality or
real-time processing. Unknown/occluded returns are deliberately retained.

References: [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext),
[Burn ONNX import](https://burn.dev/books/burn/onnx-import.html).

For a numerical reproduction (requires NumPy and ONNX Runtime on CPU), run the
GPU command with `--debug-tensors`, then:

```sh
python rust/experiments/lidar-ai/verify_result.py --scan scan.bin --result notes/new-lidar-result --onnx notes/my-lidar-model/salsanext.onnx
```

This separately projects the original scan, compares CPU/GPU model probabilities
and masks, and verifies both output files against exact original source records.
