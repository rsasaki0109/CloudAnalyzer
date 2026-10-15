"""Verify saved evidence, optionally the ZIP and its raw-record subset provenance."""

import argparse
import hashlib
import json
import runpy
import zipfile
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source and not args.bundle:
        parser.error("--source requires --bundle")
    root = Path(__file__).resolve().parent
    hashes = json.loads((root / "files-sha256.json").read_text())
    assert set(hashes) == {
        p.name for p in root.iterdir() if p.is_file() and p.name != "files-sha256.json"
    }
    for name, value in hashes.items():
        assert digest(root / name) == value, name
    receipt = json.loads((root / "receipt.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    preview = receipt["preview_pointcloud"]
    assert manifest["preview_pointcloud"] == preview
    assert manifest["schema"] == "cloudanalyzer.mapping_review_bundle.v2"
    assert "map" not in manifest["roles"] and "preview_map" in manifest["roles"]
    assert receipt["state_before"] == receipt["state_after"]
    assert receipt["mapping_attempts_spent"] == 0
    assert not receipt["native_processing_performed"]
    assert not receipt["geometry_regenerated"]
    assert receipt["exact_hd_and_saved_audits"]
    assert receipt["retained_record_bytes_exact"]
    assert not receipt["full_original_map_included"]
    assert receipt["standalone_result"]["integrity_verified"]
    assert preview["source_count"] == 646309
    assert preview["preview_count"] == 161578 and preview["every_nth_record"] == 4
    assert (
        preview["source"]["sha256"]
        == "944865739c31dd07e44c7c02f59f0f462dd5eef03d1eada6b0cce7d8cec8c7ad"
    )
    assert preview["source"] == manifest["review"]["artifacts"]["map"]
    assert not receipt["extent"]["passes_requested_extent"]
    assert receipt["audit_summaries"][0]["source_support"]["height_mismatches"] == 363
    if args.bundle:
        assert args.bundle.stat().st_size == receipt["bundle"]["bytes"]
        assert digest(args.bundle) == receipt["bundle"]["sha256"]
        module = runpy.run_path(
            str(root.parents[2] / "cloudanalyzer/ca/mapping_bundle.py")
        )
        result = module["inspect_mapping_bundle"](str(args.bundle))
        assert result["preview_pointcloud"] == preview
        assert result["review"] == manifest["review"]
        with zipfile.ZipFile(args.bundle) as archive:
            assert json.loads(archive.read("manifest.json")) == manifest
            saved = json.loads(archive.read(result["roles"]["hd_source_audits"]))
            previous = json.loads(
                (root.parent / "nclt-review-bundle/manifest.json").read_text()
            )
            for role in (
                "hd_map",
                "hd_editable_map",
                "hd_projector",
                "hd_source_audits",
            ):
                assert (
                    manifest["review"]["artifacts"][role]["sha256"]
                    == previous["review"]["artifacts"][role]["sha256"]
                )
            for i, audit in enumerate(
                [
                    saved["editable"],
                    saved["reopened_osm"],
                    *saved["ground_consensus"].values(),
                ]
            ):
                lanes = audit["quality"]["lanes"]
                for key in (
                    "samples",
                    "supported",
                    "height_mismatches",
                    "insufficient_returns",
                ):
                    assert (
                        sum(
                            l[t][key]
                            for l in lanes
                            for t in ("left", "right", "center")
                        )
                        == receipt["audit_summaries"][i]["source_support"][key]
                    )
            if args.source:
                assert args.source.stat().st_size == preview["source"]["bytes"]
                assert digest(args.source) == preview["source"]["sha256"]

                def skip_header(stream):
                    for _ in range(32):
                        if stream.readline(16384) == b"end_header\n":
                            return
                    raise AssertionError("incomplete PLY header")

                with args.source.open("rb") as source, archive.open(
                    result["roles"]["preview_map"]
                ) as subset:
                    skip_header(source)
                    skip_header(subset)
                    kept = 0
                    for i in range(preview["source_count"]):
                        row = source.read(32)
                        assert len(row) == 32
                        if i % 4 == 0:
                            assert subset.read(32) == row, i
                            kept += 1
                    assert kept == 161578 and not source.read(1) and not subset.read(1)
    print(
        "Preview evidence verified"
        + ("; ZIP verified" if args.bundle else "")
        + ("; all retained original records matched" if args.source else "")
    )


if __name__ == "__main__":
    main()
