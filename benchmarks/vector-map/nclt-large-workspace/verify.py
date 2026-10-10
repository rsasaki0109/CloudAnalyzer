"""Verify recorded workspace evidence and, optionally, every original ZIP member."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
CHUNK = 8 * 1024**2
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--zip", type=Path)
parser.add_argument("--legacy", type=Path)
parser.add_argument("--candidate", type=Path)
args = parser.parse_args()
assert all((args.zip, args.legacy, args.candidate)) or not any(
    (args.zip, args.legacy, args.candidate)
), "Supply all three binary inputs together"


def read(name):
    return json.loads((ROOT / name).read_text())


def stream_identity(stream, size):
    chunks, whole, actual = [], hashlib.sha256(), 0
    while data := stream.read(CHUNK):
        actual += len(data)
        whole.update(data)
        chunks.append(hashlib.sha256(data).digest())
    assert actual == size
    root = hashlib.sha256(
        f"sha256-chunks-v1:{CHUNK}:{size}:".encode() + b"".join(chunks)
    ).hexdigest()
    return root, whole.hexdigest()


hashes = read("files-sha256.json")
assert set(hashes) == {
    str(p.relative_to(ROOT))
    for p in ROOT.rglob("*")
    if p.is_file() and p.name != "files-sha256.json" and "__pycache__" not in p.parts
}
for name, expected in hashes.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name

manifest, project, receipt = [read(n) for n in (
    "manifest.json", "project.json", "large-workspace-receipt.json"
)]
assert receipt["schema"] == "cloudanalyzer.large_workspace_receipt.v1"
assert 64 * 1024**2 < receipt["zipBytes"] < 256 * 1024**2
assert receipt["legacyZipBytes"] == 54815611
assert receipt["legacyZipSha256"] == "c918ad9dfd2a9972ee385d9e9cf2c2956a3a7cf2fc5d4ece348b7a65177b7db4"
assert receipt["candidateInputBytes"] == 18824718
assert receipt["candidateInputSha256"] == "ce475e7e6b70de9474fd34a6d3c117535ab784b405d57f0cdd9c1854e422a3c8"
assert all(receipt[k] is True for k in (
    "cloudExportsExact", "originalAssetsExact", "poseGraphExact", "hdMapExact",
    "reviewsExact", "originalArchiveExact", "savedAuditsDisabled",
))
assert manifest["schema"] == "cloudanalyzer.project_snapshot.v3"
assert manifest["cloud_count"] == len(project["session"]["clouds"]) == len(receipt["clouds"]) == 3
assert manifest["pose_graph_sources_included"] and manifest["review_archive_included"]
assert manifest["point_data"] == "current_loaded_records"
assert manifest["coordinate_frame"] == "current_transformed_coordinates"
assert manifest["unloaded_original_density_included"] is False
assert receipt["nodes"] == len(json.loads(project["poseGraph"]["snapshot"])["graph"]["nodes"]) == 199
members = {m["name"]: m for m in manifest["files"]}
assert len(members) == len(manifest["files"]) == 6
assert sum(m["bytes"] for m in manifest["files"]) < 256 * 1024**2
for cloud, result in zip(project["session"]["clouds"], receipt["clouds"]):
    descriptor = members[cloud["source"]["name"]]
    assert cloud["transforms"] == [] and cloud["loadMaxPoints"] == 0
    assert result["name"] == cloud["name"] and result["bytes"] == descriptor["bytes"]
    assert cloud["source"] == descriptor["identity"]
assert project["session"]["clouds"][1]["displayPreview"]
for result in receipt["assets"]:
    assert members[result["name"]]["bytes"] == result["bytes"]

if args.zip:
    for path, size, digest in (
        (args.zip, receipt["zipBytes"], receipt["zipSha256"]),
        (args.legacy, receipt["legacyZipBytes"], receipt["legacyZipSha256"]),
        (args.candidate, receipt["candidateInputBytes"], receipt["candidateInputSha256"]),
    ):
        with path.open("rb") as stream:
            assert path.stat().st_size == size
            assert stream_identity(stream, size)[1] == digest
    with zipfile.ZipFile(args.zip) as archive, zipfile.ZipFile(args.legacy) as legacy:
        assert archive.testzip() is None  # Independent CRC32 verification.
        assert json.loads(archive.read("manifest.json")) == manifest
        assert json.loads(archive.read("project.json")) == project
        assert set(archive.namelist()) == {"manifest.json", *(m["path"] for m in manifest["files"])}
        actual = {}
        for m in manifest["files"]:
            info = archive.getinfo(m["path"])
            assert info.compress_type == zipfile.ZIP_STORED and info.file_size == m["bytes"]
            with archive.open(info) as stream:
                root, digest = stream_identity(stream, info.file_size)
            assert m["identity"]["algorithm"] == "sha256-chunks-v1"
            assert m["identity"]["kind"] == "file" and m["identity"]["name"] == m["name"]
            assert m["identity"]["size"] == info.file_size and m["identity"]["digest"] == root
            actual[m["name"]] = digest
        for cloud, result in zip(project["session"]["clouds"], receipt["clouds"]):
            assert actual[cloud["source"]["name"]] == result["sha256"]
        for asset in receipt["assets"]:
            assert actual[asset["name"]] == asset["sha256"]
        old = json.loads(legacy.read("project.json"))
        assert json.loads(project["poseGraph"]["snapshot"]) == json.loads(old["poseGraph"]["snapshot"])
        assert json.loads(project["vectorMap"]) == json.loads(old["vectorMap"])
        assert project["reviews"] == old["reviews"]
        # Old point records and both original inputs remain exact, independent of the browser receipt.
        old_manifest = json.loads(legacy.read("manifest.json"))
        for i, m in enumerate(old_manifest["files"][1:]):
            digest = receipt["clouds"][i]["sha256"] if i < 2 else actual[m["name"]]
            assert digest == m["sha256"]
        # Browser indexing reorders input points and adds its PLY comment. Compare
        # all complete XYZ/intensity/correction records, including duplicates.
        header, records = archive.read("clouds/002.ply").split(b"end_header\n", 1)
        original_header, original_records = args.candidate.read_bytes().split(b"end_header\n", 1)
        expected_properties = [b"property double x", b"property double y", b"property double z",
                               b"property float intensity", b"property float correction"]
        for h in (header, original_header):
            assert b"element vertex 588267" in h.splitlines()
            assert [line for line in h.splitlines() if line.startswith(b"property ")] == expected_properties
        assert len(records) == len(original_records) == 588267 * 32
        assert sorted(records[i:i + 32] for i in range(0, len(records), 32)) == sorted(
            original_records[i:i + 32] for i in range(0, len(original_records), 32)
        ), "Imported candidate record multiset differs"
        refs = [ref for s in project["poseGraph"]["sources"]
                for ref in ([s["graph"]] if s["graph"] else []) + s["scans"]]
        refs.append(project["reviewArchive"])
        for ref in refs:
            m = members[ref["name"].split("/")[-1]]
            assert ref["size"] == m["bytes"] and ref["digest"] == m["identity"]["digest"]
print("Large workspace metadata and receipts verified" + (
    "; complete ZIP, CRC32, chunk identities and original inputs independently verified"
    if args.zip else "; binary verification requires --zip --legacy --candidate"
))
