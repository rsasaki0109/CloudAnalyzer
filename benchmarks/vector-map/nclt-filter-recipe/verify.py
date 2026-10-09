"""Check exact recipe records, source membership and immutable archive identity."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rows(data):
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    assert "format binary_little_endian 1.0" in header
    assert [line for line in header.splitlines() if line.startswith("property ")] == [
        "property double x",
        "property double y",
        "property double z",
        "property float intensity",
        "property float correction",
    ]
    count = int(re.search(r"element vertex (\d+)", header)[1])
    assert len(data) - end == count * 32
    return count, Counter(data[pos : pos + 32] for pos in range(end, len(data), 32))


if len(sys.argv) not in (4, 5):
    raise SystemExit("verify.py workspace.zip original_full_map.ply original_review.zip [wasm]")
root = Path(__file__).resolve().parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    sha, name = line.split("  ", 1)
    assert digest((root / name).read_bytes()) == sha, name
receipt = json.loads((root / "recipe-receipt.json").read_text())
for key in (
    "individualFiltersExact", "allOutputRecordsFromSource", "hdMapUnchanged",
    "previewFlagRetained", "oneUndoRedo", "workspaceRoundtripExact",
):
    assert receipt[key] is True, key
archive = Path(sys.argv[1]).read_bytes()
assert len(archive) == receipt["zipBytes"]
assert digest(archive) == receipt["zipSha256"]
original = Path(sys.argv[2]).read_bytes()
assert digest(original) == receipt["originalInputSha256"]
review = Path(sys.argv[3]).read_bytes()
assert digest(review) == receipt["originalReviewSha256"]
if len(sys.argv) == 5:
    assert digest(Path(sys.argv[4]).read_bytes()) == receipt["wasmSha256"]
with zipfile.ZipFile(sys.argv[3]) as z:
    review_manifest = json.loads(z.read("manifest.json"))
    preview = z.read(review_manifest["roles"]["preview_map"])
    hd_map = json.loads(z.read(review_manifest["roles"]["hd_editable_map"]))
with zipfile.ZipFile(sys.argv[1]) as z:
    manifest = json.loads(z.read("manifest.json"))
    members = {m["name"]: m for m in manifest["files"]}
    for member in manifest["files"]:
        data = z.read(member["path"])
        assert len(data) == member["bytes"], member["path"]
        assert digest(data) == member["sha256"], member["path"]
    project = json.loads(z.read("project.json"))
    clouds = project["session"]["clouds"]
    assert len(clouds) == 4
    assert clouds[1]["displayPreview"] and clouds[3]["displayPreview"]
    assert json.loads(project["vectorMap"]) == hd_map
    assert z.read(members[project["reviewArchive"]["name"]]["path"]) == review
    for i, raw in enumerate((original, preview)):
        raw_count, raw_records = rows(raw)
        source = z.read(members[clouds[i]["source"]["name"]]["path"])
        source_count, source_records = rows(source)
        assert source_count == raw_count == receipt["sourcePoints"][i]
        assert source_records == raw_records
        del source_records
        output = z.read(members[clouds[i + 2]["source"]["name"]]["path"])
        output_count, output_records = rows(output)
        assert not (output_records - raw_records), "Duplicated or changed source records"
        processing = clouds[i + 2]["processing"]
        assert processing == receipt["records"][i]
        assert processing["recipe"] == receipt["recipe"]
        assert processing["wasmSha256"] == receipt["wasmSha256"]
        assert processing["input"] == {
            "name": clouds[i]["name"], "points": source_count, "sha256": digest(source),
        }
        assert processing["output"] == {"points": output_count, "sha256": digest(output)}
print("Original records, exact output membership, map, preview flag and provenance verified.")
