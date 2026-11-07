"""Check one debug result against independent CPU inference; no semantic labels used."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import onnxruntime as ort
    if args.scan.stat().st_size > 150000 * 16:
        parser.error("single-scan budget exceeded")
    source = args.scan.read_bytes()
    report = json.loads((args.result / "report.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(source).hexdigest() == report["input_sha256"]
    assert hashlib.sha256(args.onnx.read_bytes()).hexdigest() == report["onnx_sha256"]
    points = np.frombuffer(source, dtype="<f4").reshape(-1, 4)
    owners = np.full(64 * 2048, -1, dtype=np.int64)
    ranges = np.full(64 * 2048, np.inf)
    low, up = math.radians(-25), math.radians(3)
    for i, point in enumerate(points):
        if not np.isfinite(point).all():
            continue
        x, y, z, intensity = map(float, point)
        assert 0 <= intensity <= 1
        r = math.hypot(math.hypot(x, y), z)
        if not 1e-6 <= r <= 100:
            continue
        pitch = math.asin(z / r)
        if not low <= pitch <= up:
            continue
        col = min(2047, math.floor(0.5 * (-math.atan2(y, x) / math.pi + 1) * 2048))
        row = min(63, math.floor((up - pitch) / (up - low) * 64))
        pixel = row * 2048 + col
        if r < ranges[pixel]:
            owners[pixel], ranges[pixel] = i, r
    pixels = np.flatnonzero(owners >= 0)
    inputs = np.zeros((1, 5, 64 * 2048), dtype=np.float32)
    mean = np.array([12.12, 10.88, 0.23, -1.04, 0.21], dtype=np.float32)
    std = np.array([12.32, 11.47, 6.91, 0.86, 0.16], dtype=np.float32)
    raw_values = np.column_stack([ranges[pixels].astype(np.float32), points[owners[pixels]]])
    inputs[0, :, pixels] = (raw_values - mean) / std
    inputs = inputs.reshape(1, 5, 64, 2048)
    native_input = np.fromfile(args.result / "input.f32", dtype="<f4").reshape(inputs.shape)
    assert np.array_equal(inputs, native_input), "CPU/native preprocessing differs"
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    cpu = session.run(None, {"range_image": inputs})[0].reshape(20, -1)
    gpu = np.fromfile(args.result / "probabilities.f32", dtype="<f4").reshape(cpu.shape)
    assert np.isfinite(gpu).all() and np.isfinite(cpu).all()
    difference = np.abs(cpu - gpu)
    # A bounded numerical check, not an accuracy/precision threshold for detection.
    assert np.allclose(cpu, gpu, atol=1e-4, rtol=1e-4), "model backends disagree"
    labels, best = cpu.argmax(0), cpu.max(0)
    classes = [report["class_names"].index(name) for name in report["selection"]["target_classes"]]
    expected = np.zeros(len(points), dtype=np.uint8)
    expected[owners[pixels]] = np.isin(labels[pixels], classes) & (best[pixels] >= np.float32(report["selection"]["threshold"]))
    mask = (args.result / "candidate-mask.u8").read_bytes()
    assert expected.tobytes() == mask, "candidate mask differs from CPU inference"
    records = np.frombuffer(source, dtype="V16")
    assert (args.result / "retained.bin").read_bytes() == records[expected == 0].tobytes()
    assert (args.result / "candidates.bin").read_bytes() == records[expected == 1].tobytes()
    record = {"reference_role": "ONNX Runtime CPU, same pretrained CNN; not semantic ground truth",
        "cpu_runtime_version": ort.__version__, "preprocessing_byte_exact": True,
        "probability_max_absolute_difference": float(difference.max()),
        "probability_mean_absolute_difference": float(difference.mean()),
        "all_pixel_argmax_agreement": float(np.mean(cpu.argmax(0) == gpu.argmax(0))),
        "candidate_mask_byte_exact": True, "retained_and_candidate_source_records_byte_exact": True,
        "source_points": len(points), "candidate_points": int(expected.sum()),
        "unclassified_retained": len(points) - len(pixels), "source_unchanged": True}
    print(json.dumps(record, indent=2))

if __name__ == "__main__":
    main()
