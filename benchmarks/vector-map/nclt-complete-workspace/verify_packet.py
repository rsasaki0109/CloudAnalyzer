"""Verify the complete-workspace packet and optional original binary inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--zip", type=Path)
parser.add_argument("--input", type=Path)
parser.add_argument("--review", type=Path)
args = parser.parse_args()


def read(name):
    return json.loads((root / name).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


hashes = read("files-sha256.json")
assert set(hashes) == {
    str(p.relative_to(root))
    for p in root.rglob("*")
    if p.is_file() and p.name != "files-sha256.json" and "__pycache__" not in p.parts
}
for name, expected in hashes.items():
    assert digest((root / name).read_bytes()) == expected, name
manifest, project, receipt, browser = [
    read(n)
    for n in (
        "manifest.json",
        "project.json",
        "verification.json",
        "browser-receipt.json",
    )
]
assert manifest["schema"] == "cloudanalyzer.project_snapshot.v2"
assert (
    manifest["cloud_count"] == 2
    and manifest["pose_graph_sources_included"]
    and manifest["review_archive_included"]
)
assert not manifest["unloaded_original_density_included"]
assert receipt["zip_sha256"] == browser["zipSha256"]
assert receipt["zip_bytes"] == browser["zipBytes"] < 64 * 1024**2
assert (
    browser["nodes"]
    == receipt["graph_nodes"]
    == len(json.loads(project["poseGraph"]["snapshot"])["graph"]["nodes"])
    == 199
)
assert (
    browser["poseGraphExact"]
    and browser["hdMapExact"]
    and browser["archivedAuditsRemainOriginal"]
)
assert (
    not receipt["cross_drive_alignment_established"]
    and not receipt["frozen_audits_validate_current_edits"]
)
assert project["session"]["clouds"][1]["displayPreview"]
assert len(manifest["files"]) == 5


def source_digest(data):
    size = 8 * 1024**2
    chunks = [
        hashlib.sha256(data[i : i + size]).digest() for i in range(0, len(data), size)
    ]
    return digest(f"sha256-chunks-v1:{size}:{len(data)}:".encode() + b"".join(chunks))


if args.zip:
    assert args.zip.stat().st_size == receipt["zip_bytes"]
    assert digest(args.zip.read_bytes()) == receipt["zip_sha256"]
    with zipfile.ZipFile(args.zip) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            "manifest.json",
            *(d["path"] for d in manifest["files"]),
        }
        assert archive.read("manifest.json") == (root / "manifest.json").read_bytes()
        assert archive.read("project.json") == (root / "project.json").read_bytes()
        members = {}
        for descriptor in manifest["files"]:
            data = archive.read(descriptor["path"])
            assert (
                len(data) == descriptor["bytes"]
                and digest(data) == descriptor["sha256"]
            )
            members[descriptor["name"]] = data
        refs = [c["source"] for c in project["session"]["clouds"]]
        refs.extend(
            ref
            for s in project["poseGraph"]["sources"]
            for ref in [*([] if s["graph"] is None else [s["graph"]]), *s["scans"]]
        )
        refs.append(project["reviewArchive"])
        for ref in refs:
            data = members[ref["name"].split("/")[-1]]
            assert len(data) == ref["size"] and source_digest(data) == ref["digest"]
        for description in receipt["points"]:
            data = archive.read(description["path"])
            header = data[:8192].split(b"end_header\n")[0].decode().splitlines()
            assert f"element vertex {description['count']}" in header
            assert [
                line for line in header if line.startswith("property ")
            ] == description["properties"]
        for source, name, expected in [
            (args.input, "nclt-2012-04-29.mcap", browser["inputSha256"]),
            (
                args.review,
                "nclt-hd-preflight-display-preview.zip",
                browser["originalArchiveSha256"],
            ),
        ]:
            assert digest(members[name]) == expected
            if source:
                assert members[name] == source.read_bytes()
print(
    "Complete-workspace packet verified"
    + (
        " with exact original binary inputs"
        if args.zip
        else " (binary ZIP requires --zip)"
    )
)
