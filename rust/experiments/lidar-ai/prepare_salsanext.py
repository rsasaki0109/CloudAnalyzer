"""Export the pinned upstream checkpoint locally; never download or install implicitly.

Requires existing torch, numpy and onnx. Training-state objects are replaced by
inert placeholders while retaining PyTorch's weights-only unpickler.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

ARCHIVE_SHA = "cfbbb5403f9438ccc1ff0fac6b64c3f5f21a1881e76e4172a45a0e077624e0c5"
WEIGHTS_SHA = "1757b8917e81b39058feb78dd17d690d74a878cab272a2c7b47bd8e5869ee811"
UPSTREAM = "https://drive.google.com/file/d/1utfzooTDAlV5M6XGvCE0-L-vbdLe_2rD/view"

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

class _IgnoredTrainingState:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        pass

def _training_getattr(instance, name):
    if instance is not _IgnoredTrainingState and not isinstance(instance, _IgnoredTrainingState):
        raise ValueError("unexpected training method reference")
    return _IgnoredTrainingState()

def _plain_defaultdict(*args):
    return {}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="upstream pretrained.zip; see README")
    parser.add_argument("--out", type=Path, required=True, help="NEW local output directory")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be a new directory")
    if not args.archive.is_file() or args.archive.stat().st_size > 128 * 1024**2:
        parser.error("missing archive or archive exceeds128MiB")
    if digest(args.archive) != ARCHIVE_SHA:
        parser.error("checkpoint archive differs from the pinned upstream release")
    # Imported only on explicit export; ordinary Rust builds do not need Python ML.
    import numpy as np
    import onnx
    import torch
    args.out.mkdir()
    with zipfile.ZipFile(args.archive) as archive:
        entry = archive.getinfo("pretrained/SalsaNext")
        if entry.file_size > 128 * 1024**2:
            raise ValueError("checkpoint exceeds128MiB")
        # Only this named entry is copied; no archive paths are extracted.
        weights = archive.read(entry)
    if hashlib.sha256(weights).hexdigest() != WEIGHTS_SHA:
        raise ValueError("unexpected checkpoint bytes")
    allowed = [
        dict, (_plain_defaultdict, "collections.defaultdict"),
        (_training_getattr, "builtins.getattr"), (_training_getattr, "__builtin__.getattr"),
        (_IgnoredTrainingState, "torch.optim.lr_scheduler.CyclicLR"),
        (_IgnoredTrainingState, "torch.optim.sgd.SGD"),
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype, type(np.dtype("float64")), type(np.dtype("float32")),
    ]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
    architecture = Path(__file__).parent / "third_party" / "salsanext.py"
    spec = importlib.util.spec_from_file_location("salsanext_pinned", architecture)
    if spec is None or spec.loader is None:
        raise RuntimeError("missing architecture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = checkpoint["state_dict"]
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model = module.SalsaNext(20)
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(2)
    output = args.out / "salsanext.onnx"
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 5, 64, 2048), str(output),
            input_names=["range_image"], output_names=["class_probabilities"],
            opset_version=18, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(output)))
    record = {"onnx_sha256": digest(output), "source_weight_sha256": WEIGHTS_SHA,
        "archive_sha256": ARCHIVE_SHA, "source_url": UPSTREAM,
        "architecture_revision": "7548c124b48f0259cdc40e98dfc3aeeadca6070c",
        "input_shape": [1, 5, 64, 2048], "output_shape": [1, 20, 64, 2048],
        "exporter": "torch " + torch.__version__, "weights_only_load": True,
        "redistributed": False}
    (args.out / "onnx-export.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))

if __name__ == "__main__":
    main()
