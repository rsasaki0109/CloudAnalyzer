"""Verify the small receipt, and optionally the exact portable review ZIP."""

import argparse
import hashlib
import json
import runpy
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    expected = json.loads((root / "files-sha256.json").read_text())
    assert set(expected) == {
        p.name for p in root.iterdir() if p.is_file() and p.name != "files-sha256.json"
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    receipt = json.loads((root / "receipt.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert receipt["state_before"] == receipt["state_after"]
    assert (
        receipt["exact_delivered_artifacts"] and receipt["mapping_attempts_spent"] == 0
    )
    assert (
        not receipt["geometry_regenerated"]
        and not receipt["native_processing_performed"]
    )
    assert receipt["standalone_interpreter_flags"] == ["-I", "-S"]
    assert receipt["standalone_result"]["integrity_verified"]
    assert receipt["verified_files"] == len(manifest["files"]) == 15
    assert manifest["pointcloud_summary"]["map_points"] == 646309
    assert (
        manifest["review"]["artifacts"]["map"]["sha256"]
        == "944865739c31dd07e44c7c02f59f0f462dd5eef03d1eada6b0cce7d8cec8c7ad"
    )
    assert manifest["review"]["hd_repair_decision"]["adopted"]
    assert not receipt["extent"]["passes_requested_extent"]
    assert len(receipt["audit_summaries"]) == 4
    assert receipt["audit_summaries"][0]["source_support"]["height_mismatches"] == 363
    for value in (receipt, manifest):
        assert not value["raw_logs_included"] and not value["resumable_mapping_job"]
        assert (
            not value["independent_accuracy_established"]
            and not value["deployment_ready"]
        )
    if args.bundle:
        descriptor = receipt["bundle"]
        assert args.bundle.stat().st_size == descriptor["bytes"]
        digest = hashlib.sha256()
        with args.bundle.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        assert digest.hexdigest() == descriptor["sha256"]
        module = runpy.run_path(
            str(root.parents[2] / "cloudanalyzer/ca/mapping_bundle.py")
        )
        result = module["inspect_mapping_bundle"](str(args.bundle))
        assert result["review"] == manifest["review"]
        with zipfile.ZipFile(args.bundle) as archive:
            assert json.loads(archive.read("manifest.json")) == manifest
            saved = json.loads(archive.read(result["roles"]["hd_source_audits"]))
        four = [
            ("legacy.editable", saved["editable"]),
            ("legacy.reopened_osm", saved["reopened_osm"]),
            *[
                (f"ground_consensus.{k}", v)
                for k, v in saved["ground_consensus"].items()
            ],
        ]
        summaries = []
        for name, audit in four:
            lanes = audit["quality"]["lanes"]
            summaries.append(
                {
                    "name": name,
                    "source_support": {
                        k: sum(
                            lane[t][k]
                            for lane in lanes
                            for t in ("left", "right", "center")
                        )
                        for k in (
                            "samples",
                            "supported",
                            "height_mismatches",
                            "insufficient_returns",
                        )
                    },
                    "needs_review_lane_ids": [
                        lane["lane"] for lane in lanes if lane["needs_review"]
                    ],
                }
            )
        assert summaries == receipt["audit_summaries"]
    print(
        "Portable review packet verified"
        + ("; ZIP verified without native processing" if args.bundle else "")
    )


if __name__ == "__main__":
    main()
