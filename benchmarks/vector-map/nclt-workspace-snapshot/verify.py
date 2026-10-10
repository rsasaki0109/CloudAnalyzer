"""Recheck the saved NCLT workspace evidence with the Python standard library."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def records(data):
    header, body = data.split(b"end_header\n", 1)
    lines = header.decode("ascii").splitlines()
    assert lines[:2] == ["ply", "format binary_little_endian 1.0"]
    assert [line for line in lines if line.startswith("property ")] == [
        "property double x",
        "property double y",
        "property double z",
        "property float intensity",
        "property float correction",
    ]
    count = int(
        next(line.split()[2] for line in lines if line.startswith("element vertex "))
    )
    assert len(body) == count * 32
    return count, Counter(body[i : i + 32] for i in range(0, len(body), 32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source and not args.bundle:
        parser.error("--source requires --bundle")
    root = Path(__file__).resolve().parent
    for name, expected in json.loads((root / "files-sha256.json").read_text()).items():
        assert sha256((root / name).read_bytes()) == expected, name
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["schema"] == "cloudanalyzer.point_workspace_evidence.v1"
    assert receipt["root_child_state_before"] == receipt["root_child_state_after"]
    assert receipt["mapping_attempts_spent"] == 0
    assert receipt["hd_geometry_regenerated"] is False
    assert receipt["new_source_audits_run"] is False
    assert receipt["independent_accuracy_established"] is False
    assert receipt["deployment_ready"] is False
    result = {
        "evidence_hashes_verified": True,
        "bundle_verified": False,
        "source_verified": False,
    }
    if args.bundle:
        data = args.bundle.read_bytes()
        assert len(data) == receipt["snapshot"]["bytes"]
        assert sha256(data) == receipt["snapshot"]["sha256"]
        with zipfile.ZipFile(args.bundle) as archive:
            assert archive.testzip() is None
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest == json.loads((root / "manifest.json").read_text())
            assert manifest["schema"] == "cloudanalyzer.project_snapshot.v1"
            paths = [entry["path"] for entry in manifest["files"]]
            assert len(paths) == len(set(paths))
            assert len(archive.namelist()) == len(set(archive.namelist()))
            assert set(archive.namelist()) == {"manifest.json", *paths}
            contents = []
            for entry in manifest["files"]:
                content = archive.read(entry["path"])
                assert len(content) == entry["bytes"]
                assert sha256(content) == entry["sha256"]
                contents.append(content)
            project = json.loads(contents[0])
            clouds = project["session"]["clouds"]
            assert len(clouds) == len(contents) - 1 == 2
            assert [cloud["visible"] for cloud in clouds] == receipt["saved_visibility"]
            assert [cloud["transforms"] for cloud in clouds] == receipt[
                "snapshot_source_transforms"
            ]
            assert len(json.loads(project["vectorMap"])["lanes"]) == receipt["hd_lanes"]
            for cloud, entry, content in zip(
                clouds, manifest["files"][1:], contents[1:]
            ):
                source = cloud["source"]
                assert source["kind"] == "file" and source["name"] == entry["name"]
                assert cloud["loadMaxPoints"] == 0 and cloud["transforms"] == []
                assert source["size"] == len(content)
                assert source["algorithm"] == "sha256-chunks-v1"
                chunk_size = 8 * 1024 * 1024
                chunks = b"".join(
                    hashlib.sha256(content[start : start + chunk_size]).digest()
                    for start in range(0, len(content), chunk_size)
                )
                prefix = f"sha256-chunks-v1:{chunk_size}:{len(content)}:".encode()
                assert sha256(prefix + chunks) == source["digest"]
            parsed = [records(content) for content in contents[1:]]
            assert [count for count, _ in parsed] == receipt["point_counts"]
            assert parsed[1][1] <= parsed[0][1]
        result["bundle_verified"] = True
        if args.source:
            source_data = args.source.read_bytes()
            assert sha256(source_data) == receipt["original_source"]["sha256"]
            count, original = records(source_data)
            assert count == parsed[0][0] and original == parsed[0][1]
            assert parsed[1][1] <= original
            result["source_verified"] = True
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
