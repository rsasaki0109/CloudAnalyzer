# LiDAR AI on wgpu — native experiment

This separate package runs the **pretrained SalsaNext SemanticKITTI CNN** with
Burn/wgpu. It is an opt-in experiment, not an installed `ca` command or a browser
inference feature. The browser can review its exported results. The normal Rust
workspace build and visibility-based dynamic removal are unchanged. Parked and
moving cars share the model's `car` class.

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

The input stays unchanged. `--classes car,truck,other-vehicle` and `--threshold 0.95`
are explicit alternatives. `--debug-tensors` writes input/output tensors for
numerical comparison; it is off by default.

Native source and weight hash reads enforce their byte budgets while reading,
including the source recheck before publication. Oversized or growing inputs
fail without publishing a result. Completion uses the same bounded reader for
the source, baseline records and debug tensors.

## Review in the browser

Open **Clouds → LiDAR AI scan review (experimental) → Open AI result…**, then
select the **original scan `.bin`**, `report.json` and `candidate-mask.u8` together.
The browser verifies the source SHA256, nearest-only projection, mask counts,
source record order and both output hashes before loading anything into the
scene. The source-order mask is never applied to reordered viewer points.

The comparison adds three clouds in one Undo step: original, retained (gray)
and candidates (red). Switch between original, highlighted, retained and
candidate-only views while orbiting the same camera. Existing clouds stay as
they were; Undo removes this comparison and Redo restores it. Removing or moving
comparison clouds disables the comparison controls until reopened/restored.
Normal cloud tools and exports remain available.

**Download exact retained/candidates .bin** saves the original inference records,
including their intensity and float bits. These downloads remain the original
result after later viewer edits. Save edited clouds using the cloud list's normal
export instead. Verification establishes matching files, not detection accuracy.

Browser comparison currently requires finite XYZI records, a history budget
large enough for one complete comparison, and at most 150,000 source points.
The native path and CPU verifier retain invalid returns unchanged. Empty result
parts have empty downloads and are skipped in the viewer. Inference and threshold
changes still run in the native command; no browser inference or fused-map removal
is implied. Reopen the three files to restore the review controls after reload;
workspace snapshots can separately retain the visible comparison clouds.

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

[Recorded native run and numerical checks](../../../benchmarks/lidar-ai/salsanext-wgpu/README.md) retain source/model hashes and the one-scan result.

## Audit removal against annotations

`evaluate_removal.py` separately scores a **frozen native result** against an
explicitly paired SemanticKITTI `.label` file. It uses only the Python standard
library and never runs inference, changes thresholds, propagates labels or writes
point files. Annotation bytes are not model inputs. Run its contract tests with:

```sh
python -m unittest discover -s rust/experiments/lidar-ai -p test_evaluate_removal.py -v
```

Create a pairing JSON from the original scan/annotation provenance. The two
SHA256 values must identify the exact source-order files, not just their names
or point counts. For example (replace the hash placeholders):

```json
{
  "app": "CloudAnalyzer SemanticKITTI pairing",
  "version": 1,
  "sequence": "08",
  "frame": "000000",
  "split": "validation",
  "point_order": "original-kitti-xyzi",
  "label_format": "semantickitti-uint32-le",
  "scan_sha256": "SHA256 of the original scan.bin",
  "labels_sha256": "SHA256 of its original .label file",
  "scan_member": "dataset/sequences/08/velodyne/000000.bin",
  "labels_member": "dataset/sequences/08/labels/000000.label"
}
```

```sh
python rust/experiments/lidar-ai/evaluate_removal.py --scan 000000.bin --labels 000000.label --pairing pairing.json --result notes/native-result --out notes/new-removal-audit.json
```

The audit verifies the native source/mask/output hashes, exact record split and
nearest-only projection before computing binary TP/FP/FN/TN, precision, recall,
IoU and false-removal classes. **Full-scan recall counts retained occluded and
out-of-view car points as misses**; projected-owner recall is shown separately,
together with the maximum full-scan recall that this ownership policy permits.
Per-instance retained fractions expose incomplete removal. Empty denominators
produce `null`, not perfect scores. Raw IDs 0,1,52,99 are ignored by the official
single-scan mapping; their removed counts and a worst-case precision bound remain
visible. Packed instance IDs are decoded separately from semantic IDs.

Pairing hashes prevent accidental file substitution; their annotation provenance
must still come from the corresponding dataset frame. A few validation frames
cannot establish general accuracy. In particular, validation sequence 08 is not
an independent held-out test for a public checkpoint selected using that split.
Dataset/model terms remain separate; no scan or annotation files are distributed.

Label format and factual ID mapping follow [SemanticKITTI](https://semantic-kitti.org/dataset)
and its [pinned API configuration](https://github.com/PRBonn/semantic-kitti-api/blob/a9c749e8124b2243b6eef1b8bcf971a9f1173a2d/config/semantic-kitti.yaml).

[Three annotated validation frames](../../../benchmarks/lidar-ai/semantic-kitti-smoke/README.md)
record actual car-point precision, full-scan recall, ignored labels, ownership
limits and the threshold trade-off. These measurements leave the inference and
default threshold unchanged.

## Optional near-surface completion experiment

The separate CPU-only `ca-lidar-complete` command uses an existing native result
with `--debug-tensors`. It does not run a model or change the original files.
The nearest-only inference command and browser review keep their existing
contract. This experiment emits **`completion-report.json` and
`completed-mask.u8`**, distinct from a normal native inference report/mask.

```sh
cargo build --manifest-path rust/experiments/lidar-ai/Cargo.toml --no-default-features --features completion --bin ca-lidar-complete
# Use the matching target directory's executable:
ca-lidar-complete --scan scan.bin --base notes/frozen-native-debug-result --out notes/new-completion
```

The fixed rule considers only valid in-range/FOV returns that lost a pixel
collision. The original owner and another distinct original owner must predict
the same target class at score ≥0.99 and lie within **15 cm** of the return;
the two seeds must be at least **2 cm** apart. A different-class owner at score
≥0.90 within 15 cm vetoes the addition. Neighborhoods over 256 owners abstain.
Original owner decisions never change, and added points never become seeds.
These geometric guards are an experiment, not a guarantee against false removal.

Before completing, the command checks the native model/profile, source identity,
exact input tensor, normalized probabilities, original mask and exact output
records. It publishes a new directory with `retained.bin`, `candidates.bin`,
`added-mask.u8`, `completed-mask.u8` and `completion-report.json`. The source and
base results remain unchanged. In **Clouds → LiDAR AI scan review**, select six
files together: the original `.bin`, baseline `report.json` / `candidate-mask.u8`,
and `completion-report.json` / `completed-mask.u8` / `added-mask.u8`. Red marks
baseline candidates, orange marks additions, and gray marks the completed
retained scan. Compare all candidates, baseline only or added only; focus the
selected candidates, Undo/Redo the complete comparison, or download exact
completed retained/candidate records and a separate `added.bin`.

**Nearest-only retained scan** shows the baseline retained set using the existing
gray completed-retained cloud plus orange additions. Orange points are kept by
the baseline; completion removes them. This view adds no extra cloud or point
load. Its two **Download nearest-only** buttons save `baseline-retained.bin` and
`baseline-candidates.bin` in exact original source order, separately from the
completed downloads. They preserve the verified baseline after view changes,
Undo/Redo or later edits, just like the completed downloads. No completion or
inference is rerun when switching views.

The three-file nearest-only reader remains strict. Completion review checks
the baseline, source/report/mask hashes, additive mask consistency, original
ownership and direct-owner distance, plus exact completed output records. It
does **not** load probability tensors or rerun the full score/two-seed/class-veto
policy. Use the native command or independent Python oracle for those checks.
The browser runs no inference and receives no annotations. Completion can
increase false removals; the nearest-only default remains unchanged.

[Completion review examples and failure traces](../../../benchmarks/lidar-ai/completion-review/README.md)
show actual browser comparisons and why the earlier two failed cases remain
experimental.

An independent standard-library Python projector/grid oracle checks the complete
mask, rejection counts, hashes and exact source-record split:

```sh
python rust/experiments/lidar-ai/verify_completion.py --scan scan.bin --base notes/frozen-native-debug-result --result notes/new-completion --out notes/new-completion-check.json
```

Verification without annotation files also checks the native model/profile,
projection and class counts, exact baseline records, completion model/selection
bindings, and the debug input reconstructed from the scan with native f32 rounding.
Rehashing a substituted debug input does not make it valid. This checks a frozen
result's consistency; it does not run inference or measure semantic accuracy.

The CPU-only Rust/Python integration test needs no model, GPU, NumPy or scan
download. Build `ca-lidar-complete` as above, then set `CA_LIDAR_COMPLETE_BIN` to
its absolute executable path when running the Python unit tests. CI runs this
test against the actual CPU binary and synthetic records; without that variable,
only this integration case is skipped.

Optionally supply **both** `--labels frame.label --pairing pairing.json` to score
the frozen completed mask against annotations, using the pairing format above.
The original `evaluate_removal.py` command still rejects non-nearest removal;
the completion verifier explicitly validates this separate selection policy.
Its nearest-only recall ceiling is a counterfactual baseline, not a bound on a
completed mask. Default inference and thresholds remain unchanged.

[Six-frame completion comparison](../../../benchmarks/lidar-ai/near-surface-completion/README.md)
records improvements and two failed diagnostic gates. Added non-car removals
keep this policy experimental; it is not adopted as the default.
